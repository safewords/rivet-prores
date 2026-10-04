//! The per-block arithmetic, with SIMD versions chosen at run time.
//!
//! Three kernels carry nearly all of the arithmetic of a frame:
//!
//! - [`idct_put`]: inverse quantisation (§7.3), the IDCT (§7.4) and the
//!   conversion to samples (§7.5.1) of one block;
//! - [`fdct_load`]: the inverse of §7.5.1 and the forward DCT of one block;
//! - [`quantise`]: the encoder's quantisation of one slice component, with
//!   a bitmap of the non-zero coefficients for the entropy coder.
//!
//! Each has a scalar version (the reference, written as plainly as the
//! formulas) and versions for SSE4.1, AVX2 and NEON that compute *exactly*
//! the same numbers: the same single-precision operations in the same
//! order, a multiply and an add where the scalar code has `a * b + c` —
//! never a fused multiply-add, whose single rounding would make the
//! decoded picture depend on the CPU. Every SIMD kernel is tested
//! bit-for-bit against the scalar one.
//!
//! The level is detected once per process. `PRORES_FORCE_SCALAR=1` in the
//! environment forces the scalar code (CI runs the tests both ways).

#[cfg(target_arch = "aarch64")]
mod neon;
pub(crate) mod scalar;
#[cfg(target_arch = "x86_64")]
mod x86;

use std::sync::OnceLock;

/// An instruction set the kernels have a version for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Isa {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Sse41,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

impl Isa {
    /// Every level this CPU can run, scalar first.
    pub(crate) fn available() -> Vec<Isa> {
        #[allow(unused_mut)]
        let mut v = vec![Isa::Scalar];
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                v.push(Isa::Sse41);
            }
            if std::is_x86_feature_detected!("avx2") {
                v.push(Isa::Avx2);
            }
        }
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("neon") {
            v.push(Isa::Neon);
        }
        v
    }

    /// The best level available, unless `PRORES_FORCE_SCALAR` is set.
    fn detect() -> Isa {
        if std::env::var_os("PRORES_FORCE_SCALAR").is_some_and(|v| !v.is_empty() && v != "0") {
            return Isa::Scalar;
        }
        *Isa::available().last().expect("scalar is always available")
    }

    /// The level in use, detected on first call.
    #[inline]
    pub(crate) fn get() -> Isa {
        static ISA: OnceLock<Isa> = OnceLock::new();
        *ISA.get_or_init(Isa::detect)
    }
}

/// Sample conversion of §7.5.1 for `bit_depth`: `s = floor(v · scale +
/// bias)` clamped to `0..=max`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Output {
    pub(crate) scale: f32,
    pub(crate) bias: f32,
    pub(crate) max: f32,
}

impl Output {
    pub(crate) fn new(bit_depth: u32) -> Output {
        Output {
            scale: (1u32 << bit_depth) as f32 / 512.0,
            bias: (1u32 << (bit_depth - 1)) as f32 + 0.5,
            max: ((1u32 << bit_depth) - 1) as f32,
        }
    }
}

/// One block of a decoded slice: `coef` (quantised, raster order `8 · v +
/// u`) times `scale` (W · qScale ÷ 8 per position), inverse transformed and
/// converted to samples, raster order.
#[inline]
pub(crate) fn idct_put(isa: Isa, coef: &[i32; 64], scale: &[f32; 64], out: Output) -> [u16; 64] {
    match isa {
        Isa::Scalar => scalar::idct_put(coef, scale, out),
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `Isa::Sse41` / `Isa::Avx2` are only produced by
        // `Isa::available`, after the CPU reported the feature.
        Isa::Sse41 => unsafe { x86::idct_put_sse41(coef, scale, out) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: as above.
        Isa::Avx2 => unsafe { x86::idct_put_avx2(coef, scale, out) },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: `Isa::Neon` is only produced after the CPU reported NEON.
        Isa::Neon => unsafe { neon::idct_put(coef, scale, out) },
    }
}

/// One block of samples (raster order) at `scale` = 512 ÷ 2^bit_depth:
/// `v = s · scale − 256` (the inverse of §7.5.1), forward transformed.
#[inline]
pub(crate) fn fdct_load(isa: Isa, pix: &[u16; 64], scale: f32) -> [f32; 64] {
    match isa {
        Isa::Scalar => scalar::fdct_load(pix, scale),
        #[cfg(target_arch = "x86_64")]
        // SAFETY: see `idct_put`.
        Isa::Sse41 => unsafe { x86::fdct_load_sse41(pix, scale) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: see `idct_put`.
        Isa::Avx2 => unsafe { x86::fdct_load_avx2(pix, scale) },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: see `idct_put`.
        Isa::Neon => unsafe { neon::fdct_load(pix, scale) },
    }
}

/// Quantises one slice component held in scanned order (`f[n · s + b]` is
/// coefficient `s` of the scan of block `b`, `n` blocks):
/// `QF = sign(x) · floor(|x| + r)` with `x = F · inv[s]`, `r` 0.5 for the DC
/// and `ac_rounding` otherwise. `out` gets the result in the same order and
/// `mask` (`n` words) a bit per non-zero value.
#[inline]
pub(crate) fn quantise(
    isa: Isa,
    f: &[f32],
    n: usize,
    inv: &[f32; 64],
    ac_rounding: f32,
    out: &mut [i32],
    mask: &mut [u64],
) {
    assert!(f.len() == 64 * n && out.len() == 64 * n && mask.len() == n);
    // The vector versions take whole vectors of blocks per scan position.
    match isa {
        Isa::Scalar => scalar::quantise(f, n, inv, ac_rounding, out, mask),
        #[cfg(target_arch = "x86_64")]
        // SAFETY: see `idct_put`; the lengths were checked above.
        Isa::Sse41 if n.is_multiple_of(4) => unsafe { x86::quantise_sse41(f, n, inv, ac_rounding, out, mask) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: as above.
        Isa::Avx2 if n.is_multiple_of(8) => unsafe { x86::quantise_avx2(f, n, inv, ac_rounding, out, mask) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: as above.
        Isa::Avx2 if n.is_multiple_of(4) => unsafe { x86::quantise_sse41(f, n, inv, ac_rounding, out, mask) },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: as above.
        Isa::Neon if n.is_multiple_of(4) => unsafe { neon::quantise(f, n, inv, ac_rounding, out, mask) },
        #[allow(unreachable_patterns)]
        _ => scalar::quantise(f, n, inv, ac_rounding, out, mask),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn range(&mut self, lo: i64, hi: i64) -> i64 {
            lo + (self.next() % (hi - lo + 1) as u64) as i64
        }
    }

    /// `PRORES_REQUIRE_SIMD=1` turns a missing SIMD level into a failure:
    /// a kernel test that silently covers only scalar code looks the same
    /// as one that passes.
    #[test]
    fn simd_levels_present_when_required() {
        let avail = Isa::available();
        eprintln!("SIMD levels: {avail:?}; in use: {:?}", Isa::get());
        if std::env::var_os("PRORES_REQUIRE_SIMD").is_some_and(|v| v == "1") {
            #[cfg(target_arch = "x86_64")]
            assert!(avail.contains(&Isa::Sse41) && avail.contains(&Isa::Avx2), "{avail:?}");
            #[cfg(target_arch = "aarch64")]
            assert!(avail.contains(&Isa::Neon), "{avail:?}");
            assert_ne!(Isa::get(), Isa::Scalar, "PRORES_REQUIRE_SIMD with PRORES_FORCE_SCALAR");
        }
    }

    fn weights(rng: &mut Rng) -> [u8; 64] {
        std::array::from_fn(|_| rng.range(2, 63) as u8)
    }

    #[test]
    fn idct_put_is_bit_exact() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for case in 0..40_000 {
            let w = weights(&mut rng);
            let q = rng.range(1, 224) as u32;
            let q = if q <= 128 { q } else { 128 + 4 * (q - 128) };
            let scale: [f32; 64] = std::array::from_fn(|k| (w[k] as u32 * q) as f32 / 8.0);
            // Mostly zeros, small values, and now and then extremes.
            let density = rng.range(0, 64);
            let coef: [i32; 64] = std::array::from_fn(|_| {
                if rng.range(0, 63) >= density {
                    return 0;
                }
                match rng.range(0, 20) {
                    0 => rng.range(-(1 << 30), 1 << 30) as i32,
                    1 => [i32::MAX, i32::MIN, 1 << 30, -(1 << 30)][rng.range(0, 3) as usize],
                    2..=5 => rng.range(-2048, 2047) as i32,
                    _ => rng.range(-8, 8) as i32,
                }
            });
            let depth = [8, 10, 12, 16][case % 4];
            let out = Output::new(depth);
            let want = scalar::idct_put(&coef, &scale, out);
            for isa in Isa::available() {
                assert_eq!(idct_put(isa, &coef, &scale, out), want, "{isa:?} case {case}");
            }
        }
    }

    #[test]
    fn fdct_load_is_bit_exact() {
        let mut rng = Rng(42);
        for case in 0..40_000 {
            let depth = [8, 10, 12, 16][case % 4];
            let max = (1i64 << depth) - 1;
            let pix: [u16; 64] = std::array::from_fn(|_| match rng.range(0, 9) {
                0 => 0,
                1 => max as u16,
                _ => rng.range(0, max) as u16,
            });
            let scale = 512.0 / (1u32 << depth) as f32;
            let want = scalar::fdct_load(&pix, scale);
            for isa in Isa::available() {
                let got = fdct_load(isa, &pix, scale);
                assert_eq!(got.map(f32::to_bits), want.map(f32::to_bits), "{isa:?} case {case}");
            }
        }
    }

    #[test]
    fn quantise_is_bit_exact() {
        let mut rng = Rng(7);
        for case in 0..4_000 {
            let n = [1usize, 2, 4, 8, 16, 32][case % 6];
            let w = weights(&mut rng);
            let q = rng.range(1, 512) as u32;
            let inv: [f32; 64] = std::array::from_fn(|k| 8.0 / (w[k] as u32 * q) as f32);
            let f: Vec<f32> = (0..64 * n)
                .map(|_| match rng.range(0, 9) {
                    0 => 0.0,
                    1 => -0.0,
                    2 => rng.range(-2048 * 64, 2048 * 64) as f32 / 64.0,
                    // Exactly half-way and the dead-zone edges.
                    3 => (rng.range(-40, 40) as f32 + 0.5) / inv[0],
                    _ => (rng.range(-20000, 20000) as f32) / 997.0,
                })
                .collect();
            let mut want = vec![0; 64 * n];
            let mut want_mask = vec![0; n];
            scalar::quantise(&f, n, &inv, 0.42, &mut want, &mut want_mask);
            for (i, &v) in want.iter().enumerate() {
                assert_eq!(want_mask[i / 64] >> (i % 64) & 1 == 1, v != 0);
            }
            for isa in Isa::available() {
                let mut got = vec![0; 64 * n];
                let mut mask = vec![!0; n];
                quantise(isa, &f, n, &inv, 0.42, &mut got, &mut mask);
                assert_eq!(got, want, "{isa:?} case {case}");
                assert_eq!(mask, want_mask, "{isa:?} case {case}");
            }
        }
    }

    /// The scalar kernels are the transforms of `dct.rs` (which run the
    /// accuracy qualification of Annex A) plus the sample conversions.
    #[test]
    fn scalar_kernels_are_the_reference_transforms() {
        let mut rng = Rng(3);
        for _ in 0..2000 {
            let coef: [i32; 64] = std::array::from_fn(|_| rng.range(-300, 300) as i32);
            let scale = [4.0f32; 64];
            let out = Output::new(10);
            let f: [f32; 64] = std::array::from_fn(|k| coef[k] as f32 * scale[k]);
            let v = crate::dct::idct(&f);
            let want: [u16; 64] =
                std::array::from_fn(|k| (v[k] * out.scale + out.bias).floor().clamp(0.0, out.max) as u16);
            assert_eq!(scalar::idct_put(&coef, &scale, out), want);
        }
    }
}
