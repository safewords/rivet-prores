//! The 8×8 DCT of RDD 36 §7.4, as a separable product with the orthonormal
//! DCT-II basis: `f[y][x] = Σv Σu T[y][v] T[x][u] F[v][u]` with
//! `T[x][u] = C(u)/2 · cos((2x+1)uπ/16)`, which is the formula of §7.4
//! (its 1/4 and C(u)C(v) split between the two factors). Single-precision
//! floating point, which meets Annex A's accuracy criteria by orders of
//! magnitude (the test below runs the qualification).

use std::sync::LazyLock;

/// `BASIS[8 * x + u]` = T[x][u].
pub(crate) static BASIS: LazyLock<[f32; 64]> = LazyLock::new(|| {
    let mut t = [0f32; 64];
    for x in 0..8 {
        for u in 0..8 {
            let c = if u == 0 {
                std::f64::consts::FRAC_1_SQRT_2
            } else {
                1.0
            };
            let a = ((2 * x + 1) * u) as f64 * std::f64::consts::PI / 16.0;
            t[8 * x + u] = (c / 2.0 * a.cos()) as f32;
        }
    }
    t
});

/// `BASIS_T[8 * u + x]` = T[x][u]: the basis transposed, for the SIMD
/// kernels' row pass of the IDCT (a vector across `x` per `u`).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) static BASIS_T: LazyLock<[f32; 64]> =
    LazyLock::new(|| std::array::from_fn(|i| BASIS[8 * (i % 8) + i / 8]));

/// Inverse DCT. `coef` holds F[v][u] at `8 * v + u`; the result holds
/// f[y][x] at `8 * y + x`.
#[inline]
pub(crate) fn idct(coef: &[f32; 64]) -> [f32; 64] {
    let t = &*BASIS;
    // tmp[v][x] = Σu F[v][u] T[x][u]
    let mut tmp = [0f32; 64];
    for v in 0..8 {
        let row = &coef[8 * v..8 * v + 8];
        if row.iter().all(|&c| c == 0.0) {
            continue;
        }
        for x in 0..8 {
            let b = &t[8 * x..8 * x + 8];
            let mut s = 0.0;
            for u in 0..8 {
                s += row[u] * b[u];
            }
            tmp[8 * v + x] = s;
        }
    }
    // f[y][x] = Σv T[y][v] tmp[v][x]
    let mut out = [0f32; 64];
    for y in 0..8 {
        let b = &t[8 * y..8 * y + 8];
        for v in 0..8 {
            let w = b[v];
            for x in 0..8 {
                out[8 * y + x] += w * tmp[8 * v + x];
            }
        }
    }
    out
}

/// Forward DCT, the inverse of [`idct`]: `F[v][u] = Σy Σx T[y][v] T[x][u] f[y][x]`.
#[inline]
pub(crate) fn fdct(pix: &[f32; 64]) -> [f32; 64] {
    let t = &*BASIS;
    // tmp[y][u] = Σx f[y][x] T[x][u]
    let mut tmp = [0f32; 64];
    for y in 0..8 {
        for x in 0..8 {
            let p = pix[8 * y + x];
            for u in 0..8 {
                tmp[8 * y + u] += p * t[8 * x + u];
            }
        }
    }
    // F[v][u] = Σy T[y][v] tmp[y][u]
    let mut out = [0f32; 64];
    for y in 0..8 {
        for v in 0..8 {
            let w = t[8 * y + v];
            for u in 0..8 {
                out[8 * v + u] += w * tmp[8 * y + u];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// §7.4's formula, term by term, in double precision.
    fn reference_idct(f: &[f64; 64]) -> [f64; 64] {
        let c = |n: usize| if n == 0 { 1.0 / 2f64.sqrt() } else { 1.0 };
        let mut out = [0f64; 64];
        for y in 0..8 {
            for x in 0..8 {
                let mut s = 0.0;
                for v in 0..8 {
                    for u in 0..8 {
                        s += c(u)
                            * c(v)
                            * f[8 * v + u]
                            * (((2 * x + 1) * u) as f64 * PI / 16.0).cos()
                            * (((2 * y + 1) * v) as f64 * PI / 16.0).cos();
                    }
                }
                out[8 * y + x] = s / 4.0;
            }
        }
        out
    }

    /// The orthogonal FDCT of IEEE 1180 eq. 1, in double precision.
    fn reference_fdct(p: &[f64; 64]) -> [f64; 64] {
        let c = |n: usize| if n == 0 { 1.0 / 2f64.sqrt() } else { 1.0 };
        let mut out = [0f64; 64];
        for v in 0..8 {
            for u in 0..8 {
                let mut s = 0.0;
                for y in 0..8 {
                    for x in 0..8 {
                        s += p[8 * y + x]
                            * (((2 * x + 1) * u) as f64 * PI / 16.0).cos()
                            * (((2 * y + 1) * v) as f64 * PI / 16.0).cos();
                    }
                }
                out[8 * v + u] = c(u) * c(v) * s / 4.0;
            }
        }
        out
    }

    /// The pseudo-random generator of the IEEE 1180-1990 appendix:
    /// integers uniformly in `-l..=h`.
    struct Ieee1180Rand(i32);
    impl Ieee1180Rand {
        fn next(&mut self, l: i32, h: i32) -> i32 {
            self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
            let i = self.0 & 0x7fff_fffe;
            let x = i as f64 / 0x7fff_ffff as f64 * (l + h + 1) as f64;
            x as i32 - l
        }
    }

    /// RDD 36 Annex A, steps 1–7, with its five criteria.
    #[test]
    fn annex_a_idct_accuracy_qualification() {
        for (l, h) in [(2048, 2047), (40, 40), (2400, 2400)] {
            for sign in [1.0f64, -1.0] {
                let mut rng = Ieee1180Rand(1);
                let mut err_sum = [0f64; 64];
                let mut err_sq = [0f64; 64];
                let mut peak = 0f64;
                let blocks = 10_000;
                for _ in 0..blocks {
                    // Step 1: pixels with three fraction bits.
                    let mut p = [0f64; 64];
                    for v in p.iter_mut() {
                        *v = sign * rng.next(l, h) as f64 / 8.0;
                    }
                    // Steps 2-3: FDCT, round to quarter-integers, clip.
                    let mut f = reference_fdct(&p);
                    for c in f.iter_mut() {
                        *c = ((*c * 4.0).round() / 4.0).clamp(-2048.0, 2047.75);
                    }
                    // Step 4: the reference IDCT, clipped, full precision.
                    let reference = reference_idct(&f);
                    // Step 5: the implementation under test.
                    let mut ff = [0f32; 64];
                    for (d, s) in ff.iter_mut().zip(f.iter()) {
                        *d = *s as f32;
                    }
                    let test = idct(&ff);
                    // Step 6: errors.
                    for k in 0..64 {
                        let r = reference[k].clamp(-256.0, 256.0);
                        let t = (test[k] as f64).clamp(-256.0, 256.0);
                        let e = t - r;
                        err_sum[k] += e;
                        err_sq[k] += e * e;
                        peak = peak.max(e.abs());
                    }
                }
                let n = blocks as f64;
                assert!(peak <= 0.15, "ppe {peak}");
                for k in 0..64 {
                    assert!(err_sq[k] / n <= 0.002, "pmse");
                    assert!((err_sum[k] / n).abs() <= 0.0015, "pme");
                }
                let omse: f64 = err_sq.iter().sum::<f64>() / (64.0 * n);
                let ome: f64 = err_sum.iter().sum::<f64>() / (64.0 * n);
                assert!(omse <= 0.001, "omse {omse}");
                assert!(ome.abs() <= 0.00015, "ome {ome}");
            }
        }
    }

    #[test]
    fn fdct_matches_the_reference_and_inverts() {
        let mut rng = Ieee1180Rand(7);
        for _ in 0..500 {
            let mut p = [0f64; 64];
            let mut pf = [0f32; 64];
            for k in 0..64 {
                p[k] = rng.next(256, 255) as f64;
                pf[k] = p[k] as f32;
            }
            let want = reference_fdct(&p);
            let got = fdct(&pf);
            for k in 0..64 {
                assert!(
                    (want[k] - got[k] as f64).abs() < 2e-3,
                    "{} {}",
                    want[k],
                    got[k]
                );
            }
            let back = idct(&got);
            for k in 0..64 {
                assert!((back[k] as f64 - p[k]).abs() < 2e-3);
            }
        }
    }
}
