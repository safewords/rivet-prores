//! SSE4.1 and AVX2 versions of the kernels. A block row is eight `f32`:
//! one AVX2 register or two SSE ones. Both transforms are written as
//! broadcast-multiply-add over whole rows, accumulating in exactly the
//! order of the scalar loops in `dct.rs` (from 0.0, a multiply then an
//! add, never fused), so the results are identical to the bit.
//!
//! Memory is touched only through the `ld*` / `st*` helpers, which take
//! fixed-size arrays: the one `unsafe` per helper is a load or store of
//! exactly that many elements.

use super::Output;
use crate::dct::{BASIS, BASIS_T};
use std::arch::x86_64::*;

// ---- memory helpers -------------------------------------------------------

#[inline]
#[target_feature(enable = "avx2")]
fn ld8(a: &[f32; 8]) -> __m256 {
    // SAFETY: `a` is 8 readable f32; the load is unaligned.
    unsafe { _mm256_loadu_ps(a.as_ptr()) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn st8(a: &mut [f32; 8], v: __m256) {
    // SAFETY: `a` is 8 writable f32; the store is unaligned.
    unsafe { _mm256_storeu_ps(a.as_mut_ptr(), v) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn ld8i(a: &[i32; 8]) -> __m256i {
    // SAFETY: `a` is 32 readable bytes; the load is unaligned.
    unsafe { _mm256_loadu_si256(a.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn st8i(a: &mut [i32; 8], v: __m256i) {
    // SAFETY: `a` is 32 writable bytes; the store is unaligned.
    unsafe { _mm256_storeu_si256(a.as_mut_ptr().cast(), v) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn st16w(a: &mut [u16; 16], v: __m256i) {
    // SAFETY: `a` is 32 writable bytes; the store is unaligned.
    unsafe { _mm256_storeu_si256(a.as_mut_ptr().cast(), v) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn ld8w(a: &[u16; 8]) -> __m128i {
    // SAFETY: `a` is 16 readable bytes; the load is unaligned.
    unsafe { _mm_loadu_si128(a.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn ld4(a: &[f32; 4]) -> __m128 {
    // SAFETY: `a` is 4 readable f32; the load is unaligned.
    unsafe { _mm_loadu_ps(a.as_ptr()) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn st4(a: &mut [f32; 4], v: __m128) {
    // SAFETY: `a` is 4 writable f32; the store is unaligned.
    unsafe { _mm_storeu_ps(a.as_mut_ptr(), v) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn ld4i(a: &[i32; 4]) -> __m128i {
    // SAFETY: `a` is 16 readable bytes; the load is unaligned.
    unsafe { _mm_loadu_si128(a.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn st4i(a: &mut [i32; 4], v: __m128i) {
    // SAFETY: `a` is 16 writable bytes; the store is unaligned.
    unsafe { _mm_storeu_si128(a.as_mut_ptr().cast(), v) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn ld8w_sse(a: &[u16; 8]) -> __m128i {
    // SAFETY: `a` is 16 readable bytes; the load is unaligned.
    unsafe { _mm_loadu_si128(a.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn st8w(a: &mut [u16; 8], v: __m128i) {
    // SAFETY: `a` is 16 writable bytes; the store is unaligned.
    unsafe { _mm_storeu_si128(a.as_mut_ptr().cast(), v) }
}

/// Rows of a 64-element block.
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

// ---- AVX2 -----------------------------------------------------------------

/// `v · scale + bias`, floored and clamped to `0..=max`, as integers.
#[inline]
#[target_feature(enable = "avx2")]
fn samples8(v: __m256, out: Output) -> __m256i {
    let x = _mm256_add_ps(
        _mm256_mul_ps(v, _mm256_set1_ps(out.scale)),
        _mm256_set1_ps(out.bias),
    );
    let x = _mm256_min_ps(
        _mm256_max_ps(_mm256_floor_ps(x), _mm256_setzero_ps()),
        _mm256_set1_ps(out.max),
    );
    _mm256_cvttps_epi32(x)
}

#[target_feature(enable = "avx2")]
pub(super) fn idct_put_avx2(coef: &[i32; 64], scale: &[f32; 64], out: Output) -> [u16; 64] {
    let (t, bt) = (rows(&BASIS), rows(&BASIS_T));
    // Inverse quantisation, kept in memory for the broadcasts below.
    let mut f = [0f32; 64];
    let (c, s) = (rows(coef), rows(scale));
    for (r, row) in rows_mut(&mut f).iter_mut().enumerate() {
        st8(
            row,
            _mm256_mul_ps(_mm256_cvtepi32_ps(ld8i(&c[r])), ld8(&s[r])),
        );
    }
    // tmp[v] = Σu F[v][u] · T[·][u], a row across x.
    let btv: [__m256; 8] = std::array::from_fn(|u| ld8(&bt[u]));
    let mut tmp = [_mm256_setzero_ps(); 8];
    for (v, row) in rows(&f).iter().enumerate() {
        let mut acc = _mm256_setzero_ps();
        for u in 0..8 {
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(row[u]), btv[u]));
        }
        tmp[v] = acc;
    }
    // out[y] = Σv T[y][v] · tmp[v], two rows at a time into 16 samples.
    let mut res = [0u16; 64];
    let pairs: &mut [[u16; 16]; 4] = res.as_chunks_mut::<16>().0.try_into().expect("64 = 4 × 16");
    for (p, pair) in pairs.iter_mut().enumerate() {
        let row = |y: usize| {
            let mut acc = _mm256_setzero_ps();
            for v in 0..8 {
                acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(t[y][v]), tmp[v]));
            }
            samples8(acc, out)
        };
        let (a, b) = (row(2 * p), row(2 * p + 1));
        // packus works per 128-bit lane: a0-3 b0-3 a4-7 b4-7, reordered.
        st16w(
            pair,
            _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packus_epi32(a, b)),
        );
    }
    res
}

#[target_feature(enable = "avx2")]
pub(super) fn fdct_load_avx2(pix: &[u16; 64], scale: f32) -> [f32; 64] {
    let (t, pr) = (rows(&BASIS), rows(pix));
    // The samples as values, kept in memory for the broadcasts.
    let mut p = [0f32; 64];
    for (y, row) in rows_mut(&mut p).iter_mut().enumerate() {
        let s = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(ld8w(&pr[y])));
        st8(
            row,
            _mm256_sub_ps(
                _mm256_mul_ps(s, _mm256_set1_ps(scale)),
                _mm256_set1_ps(256.0),
            ),
        );
    }
    // tmp[y] = Σx p[y][x] · T[x][·], a row across u.
    let tv: [__m256; 8] = std::array::from_fn(|x| ld8(&t[x]));
    let mut tmp = [_mm256_setzero_ps(); 8];
    for (y, row) in rows(&p).iter().enumerate() {
        let mut acc = _mm256_setzero_ps();
        for x in 0..8 {
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(row[x]), tv[x]));
        }
        tmp[y] = acc;
    }
    // out[v] = Σy T[y][v] · tmp[y].
    let mut res = [0f32; 64];
    for (v, row) in rows_mut(&mut res).iter_mut().enumerate() {
        let mut acc = _mm256_setzero_ps();
        for y in 0..8 {
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(t[y][v]), tmp[y]));
        }
        st8(row, acc);
    }
    res
}

#[target_feature(enable = "avx2")]
pub(super) fn quantise_avx2(
    f: &[f32],
    n: usize,
    inv: &[f32; 64],
    ac_rounding: f32,
    out: &mut [i32],
    mask: &mut [u64],
) {
    mask.fill(0);
    let abs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
    let zero = _mm256_setzero_ps();
    let (fc, oc) = (f.as_chunks::<8>().0, out.as_chunks_mut::<8>().0);
    for (s, &inv) in inv.iter().enumerate() {
        let iv = _mm256_set1_ps(inv);
        let r = _mm256_set1_ps(if s == 0 { 0.5 } else { ac_rounding });
        for c in 0..n / 8 {
            let i = n / 8 * s + c;
            let x = _mm256_mul_ps(ld8(&fc[i]), iv);
            let m = _mm256_cvttps_epi32(_mm256_floor_ps(_mm256_add_ps(_mm256_and_ps(x, abs), r)));
            let neg = _mm256_castps_si256(_mm256_cmp_ps::<_CMP_LT_OQ>(x, zero));
            let q = _mm256_sub_epi32(_mm256_xor_si256(m, neg), neg);
            st8i(&mut oc[i], q);
            let zeros = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(
                q,
                _mm256_setzero_si256(),
            )));
            let bit = 8 * i;
            mask[bit / 64] |= ((!zeros & 0xff) as u64) << (bit % 64);
        }
    }
}

// ---- SSE4.1 ---------------------------------------------------------------

#[inline]
#[target_feature(enable = "sse4.1")]
fn samples4(v: __m128, out: Output) -> __m128i {
    let x = _mm_add_ps(_mm_mul_ps(v, _mm_set1_ps(out.scale)), _mm_set1_ps(out.bias));
    let x = _mm_min_ps(
        _mm_max_ps(_mm_floor_ps(x), _mm_setzero_ps()),
        _mm_set1_ps(out.max),
    );
    _mm_cvttps_epi32(x)
}

#[target_feature(enable = "sse4.1")]
pub(super) fn idct_put_sse41(coef: &[i32; 64], scale: &[f32; 64], out: Output) -> [u16; 64] {
    let (t, bt) = (rows(&BASIS), rows(&BASIS_T));
    let mut f = [0f32; 64];
    let (c, s) = (rows(coef), rows(scale));
    for (r, row) in rows_mut(&mut f).iter_mut().enumerate() {
        let ((c0, c1), (s0, s1)) = (halves(&c[r]), halves(&s[r]));
        let (f0, f1) = halves_mut(row);
        st4(f0, _mm_mul_ps(_mm_cvtepi32_ps(ld4i(c0)), ld4(s0)));
        st4(f1, _mm_mul_ps(_mm_cvtepi32_ps(ld4i(c1)), ld4(s1)));
    }
    let btv: [(__m128, __m128); 8] = std::array::from_fn(|u| {
        let (a, b) = halves(&bt[u]);
        (ld4(a), ld4(b))
    });
    let mut tmp = [(_mm_setzero_ps(), _mm_setzero_ps()); 8];
    for (v, row) in rows(&f).iter().enumerate() {
        let (mut a0, mut a1) = (_mm_setzero_ps(), _mm_setzero_ps());
        for u in 0..8 {
            let w = _mm_set1_ps(row[u]);
            a0 = _mm_add_ps(a0, _mm_mul_ps(w, btv[u].0));
            a1 = _mm_add_ps(a1, _mm_mul_ps(w, btv[u].1));
        }
        tmp[v] = (a0, a1);
    }
    let mut res = [0u16; 64];
    for (y, row) in rows_mut(&mut res).iter_mut().enumerate() {
        let (mut a0, mut a1) = (_mm_setzero_ps(), _mm_setzero_ps());
        for v in 0..8 {
            let w = _mm_set1_ps(t[y][v]);
            a0 = _mm_add_ps(a0, _mm_mul_ps(w, tmp[v].0));
            a1 = _mm_add_ps(a1, _mm_mul_ps(w, tmp[v].1));
        }
        st8w(row, _mm_packus_epi32(samples4(a0, out), samples4(a1, out)));
    }
    res
}

#[target_feature(enable = "sse4.1")]
pub(super) fn fdct_load_sse41(pix: &[u16; 64], scale: f32) -> [f32; 64] {
    let (t, pr) = (rows(&BASIS), rows(pix));
    let mut p = [0f32; 64];
    let (sc, off) = (_mm_set1_ps(scale), _mm_set1_ps(256.0));
    for (y, row) in rows_mut(&mut p).iter_mut().enumerate() {
        let w = ld8w_sse(&pr[y]);
        let lo = _mm_cvtepi32_ps(_mm_cvtepu16_epi32(w));
        let hi = _mm_cvtepi32_ps(_mm_cvtepu16_epi32(_mm_srli_si128::<8>(w)));
        let (r0, r1) = halves_mut(row);
        st4(r0, _mm_sub_ps(_mm_mul_ps(lo, sc), off));
        st4(r1, _mm_sub_ps(_mm_mul_ps(hi, sc), off));
    }
    let tv: [(__m128, __m128); 8] = std::array::from_fn(|x| {
        let (a, b) = halves(&t[x]);
        (ld4(a), ld4(b))
    });
    let mut tmp = [(_mm_setzero_ps(), _mm_setzero_ps()); 8];
    for (y, row) in rows(&p).iter().enumerate() {
        let (mut a0, mut a1) = (_mm_setzero_ps(), _mm_setzero_ps());
        for x in 0..8 {
            let w = _mm_set1_ps(row[x]);
            a0 = _mm_add_ps(a0, _mm_mul_ps(w, tv[x].0));
            a1 = _mm_add_ps(a1, _mm_mul_ps(w, tv[x].1));
        }
        tmp[y] = (a0, a1);
    }
    let mut res = [0f32; 64];
    for (v, row) in rows_mut(&mut res).iter_mut().enumerate() {
        let (mut a0, mut a1) = (_mm_setzero_ps(), _mm_setzero_ps());
        for y in 0..8 {
            let w = _mm_set1_ps(t[y][v]);
            a0 = _mm_add_ps(a0, _mm_mul_ps(w, tmp[y].0));
            a1 = _mm_add_ps(a1, _mm_mul_ps(w, tmp[y].1));
        }
        let (r0, r1) = halves_mut(row);
        st4(r0, a0);
        st4(r1, a1);
    }
    res
}

#[target_feature(enable = "sse4.1")]
pub(super) fn quantise_sse41(
    f: &[f32],
    n: usize,
    inv: &[f32; 64],
    ac_rounding: f32,
    out: &mut [i32],
    mask: &mut [u64],
) {
    mask.fill(0);
    let abs = _mm_castsi128_ps(_mm_set1_epi32(0x7fff_ffff));
    let zero = _mm_setzero_ps();
    let (fc, oc) = (f.as_chunks::<4>().0, out.as_chunks_mut::<4>().0);
    for (s, &inv) in inv.iter().enumerate() {
        let iv = _mm_set1_ps(inv);
        let r = _mm_set1_ps(if s == 0 { 0.5 } else { ac_rounding });
        for c in 0..n / 4 {
            let i = n / 4 * s + c;
            let x = _mm_mul_ps(ld4(&fc[i]), iv);
            let m = _mm_cvttps_epi32(_mm_floor_ps(_mm_add_ps(_mm_and_ps(x, abs), r)));
            let neg = _mm_castps_si128(_mm_cmplt_ps(x, zero));
            let q = _mm_sub_epi32(_mm_xor_si128(m, neg), neg);
            st4i(&mut oc[i], q);
            let zeros = _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(q, _mm_setzero_si128())));
            let bit = 4 * i;
            mask[bit / 64] |= ((!zeros & 0xf) as u64) << (bit % 64);
        }
    }
}
