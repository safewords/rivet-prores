//! NEON versions of the kernels: a block row is two `float32x4_t`. Like
//! the x86 versions they multiply and add separately (no `vfmaq`), in the
//! scalar loops' order, so the results are identical to the bit.
//!
//! Memory is touched only through the `ld*` / `st*` helpers, which take
//! fixed-size arrays.

use super::Output;
use crate::dct::{BASIS, BASIS_T};
use std::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
fn ld4(a: &[f32; 4]) -> float32x4_t {
    // SAFETY: `a` is 4 readable f32.
    unsafe { vld1q_f32(a.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn st4(a: &mut [f32; 4], v: float32x4_t) {
    // SAFETY: `a` is 4 writable f32.
    unsafe { vst1q_f32(a.as_mut_ptr(), v) }
}

#[inline]
#[target_feature(enable = "neon")]
fn ld4i(a: &[i32; 4]) -> int32x4_t {
    // SAFETY: `a` is 4 readable i32.
    unsafe { vld1q_s32(a.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn st4i(a: &mut [i32; 4], v: int32x4_t) {
    // SAFETY: `a` is 4 writable i32.
    unsafe { vst1q_s32(a.as_mut_ptr(), v) }
}

#[inline]
#[target_feature(enable = "neon")]
fn ld8w(a: &[u16; 8]) -> uint16x8_t {
    // SAFETY: `a` is 8 readable u16.
    unsafe { vld1q_u16(a.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn st8w(a: &mut [u16; 8], v: uint16x8_t) {
    // SAFETY: `a` is 8 writable u16.
    unsafe { vst1q_u16(a.as_mut_ptr(), v) }
}

#[inline(always)]
fn rows<T>(a: &[T; 64]) -> &[[T; 8]; 8] {
    a.as_chunks::<8>().0.try_into().expect("64 = 8 × 8")
}

#[inline(always)]
fn rows_mut<T>(a: &mut [T; 64]) -> &mut [[T; 8]; 8] {
    a.as_chunks_mut::<8>().0.try_into().expect("64 = 8 × 8")
}

#[inline(always)]
fn halves<T>(a: &[T; 8]) -> (&[T; 4], &[T; 4]) {
    let (lo, hi) = a.split_at(4);
    (lo.try_into().expect("4"), hi.try_into().expect("4"))
}

#[inline(always)]
fn halves_mut<T>(a: &mut [T; 8]) -> (&mut [T; 4], &mut [T; 4]) {
    let (lo, hi) = a.split_at_mut(4);
    (lo.try_into().expect("4"), hi.try_into().expect("4"))
}

#[inline]
#[target_feature(enable = "neon")]
fn samples4(v: float32x4_t, out: Output) -> uint32x4_t {
    let x = vaddq_f32(vmulq_f32(v, vdupq_n_f32(out.scale)), vdupq_n_f32(out.bias));
    let x = vminq_f32(
        vmaxq_f32(vrndmq_f32(x), vdupq_n_f32(0.0)),
        vdupq_n_f32(out.max),
    );
    vcvtq_u32_f32(x)
}

#[target_feature(enable = "neon")]
pub(super) fn idct_put(coef: &[i32; 64], scale: &[f32; 64], out: Output) -> [u16; 64] {
    let (t, bt) = (rows(&BASIS), rows(&BASIS_T));
    let mut f = [0f32; 64];
    let (c, s) = (rows(coef), rows(scale));
    for (r, row) in rows_mut(&mut f).iter_mut().enumerate() {
        let ((c0, c1), (s0, s1)) = (halves(&c[r]), halves(&s[r]));
        let (f0, f1) = halves_mut(row);
        st4(f0, vmulq_f32(vcvtq_f32_s32(ld4i(c0)), ld4(s0)));
        st4(f1, vmulq_f32(vcvtq_f32_s32(ld4i(c1)), ld4(s1)));
    }
    let btv: [(float32x4_t, float32x4_t); 8] = std::array::from_fn(|u| {
        let (a, b) = halves(&bt[u]);
        (ld4(a), ld4(b))
    });
    let zero = vdupq_n_f32(0.0);
    let mut tmp = [(zero, zero); 8];
    for (v, row) in rows(&f).iter().enumerate() {
        let (mut a0, mut a1) = (zero, zero);
        for u in 0..8 {
            let w = vdupq_n_f32(row[u]);
            a0 = vaddq_f32(a0, vmulq_f32(w, btv[u].0));
            a1 = vaddq_f32(a1, vmulq_f32(w, btv[u].1));
        }
        tmp[v] = (a0, a1);
    }
    let mut res = [0u16; 64];
    for (y, row) in rows_mut(&mut res).iter_mut().enumerate() {
        let (mut a0, mut a1) = (zero, zero);
        for v in 0..8 {
            let w = vdupq_n_f32(t[y][v]);
            a0 = vaddq_f32(a0, vmulq_f32(w, tmp[v].0));
            a1 = vaddq_f32(a1, vmulq_f32(w, tmp[v].1));
        }
        // The values are already clamped to 0..=65535: narrowing is exact.
        st8w(
            row,
            vcombine_u16(vmovn_u32(samples4(a0, out)), vmovn_u32(samples4(a1, out))),
        );
    }
    res
}

#[target_feature(enable = "neon")]
pub(super) fn fdct_load(pix: &[u16; 64], scale: f32) -> [f32; 64] {
    let (t, pr) = (rows(&BASIS), rows(pix));
    let mut p = [0f32; 64];
    let (sc, off) = (vdupq_n_f32(scale), vdupq_n_f32(256.0));
    for (y, row) in rows_mut(&mut p).iter_mut().enumerate() {
        let w = ld8w(&pr[y]);
        let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w)));
        let hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w)));
        let (r0, r1) = halves_mut(row);
        st4(r0, vsubq_f32(vmulq_f32(lo, sc), off));
        st4(r1, vsubq_f32(vmulq_f32(hi, sc), off));
    }
    let tv: [(float32x4_t, float32x4_t); 8] = std::array::from_fn(|x| {
        let (a, b) = halves(&t[x]);
        (ld4(a), ld4(b))
    });
    let zero = vdupq_n_f32(0.0);
    let mut tmp = [(zero, zero); 8];
    for (y, row) in rows(&p).iter().enumerate() {
        let (mut a0, mut a1) = (zero, zero);
        for x in 0..8 {
            let w = vdupq_n_f32(row[x]);
            a0 = vaddq_f32(a0, vmulq_f32(w, tv[x].0));
            a1 = vaddq_f32(a1, vmulq_f32(w, tv[x].1));
        }
        tmp[y] = (a0, a1);
    }
    let mut res = [0f32; 64];
    for (v, row) in rows_mut(&mut res).iter_mut().enumerate() {
        let (mut a0, mut a1) = (zero, zero);
        for y in 0..8 {
            let w = vdupq_n_f32(t[y][v]);
            a0 = vaddq_f32(a0, vmulq_f32(w, tmp[y].0));
            a1 = vaddq_f32(a1, vmulq_f32(w, tmp[y].1));
        }
        let (r0, r1) = halves_mut(row);
        st4(r0, a0);
        st4(r1, a1);
    }
    res
}

#[target_feature(enable = "neon")]
pub(super) fn quantise(
    f: &[f32],
    n: usize,
    inv: &[f32; 64],
    ac_rounding: f32,
    out: &mut [i32],
    mask: &mut [u64],
) {
    mask.fill(0);
    let zero = vdupq_n_f32(0.0);
    let lanes: uint32x4_t = {
        let b = [1u32, 2, 4, 8];
        // SAFETY: `b` is 4 readable u32.
        unsafe { vld1q_u32(b.as_ptr()) }
    };
    let (fc, oc) = (f.as_chunks::<4>().0, out.as_chunks_mut::<4>().0);
    for (s, &inv) in inv.iter().enumerate() {
        let iv = vdupq_n_f32(inv);
        let r = vdupq_n_f32(if s == 0 { 0.5 } else { ac_rounding });
        for c in 0..n / 4 {
            let i = n / 4 * s + c;
            let x = vmulq_f32(ld4(&fc[i]), iv);
            // floor of a non-negative value, then truncation: exact.
            let m = vcvtq_s32_f32(vrndmq_f32(vaddq_f32(vabsq_f32(x), r)));
            let q = vbslq_s32(vcltq_f32(x, zero), vnegq_s32(m), m);
            st4i(&mut oc[i], q);
            let nz = vaddvq_u32(vandq_u32(vtstq_s32(q, q), lanes));
            let bit = 4 * i;
            mask[bit / 64] |= (nz as u64) << (bit % 64);
        }
    }
}
