//! The reference kernels: the transforms of `dct.rs` and the formulas of
//! §7.3 and §7.5.1, one value at a time. The SIMD versions must match these
//! bit for bit.

use super::Output;
use crate::dct::{fdct, idct};

pub(crate) fn idct_put(coef: &[i32; 64], scale: &[f32; 64], out: Output) -> [u16; 64] {
    let f: [f32; 64] = std::array::from_fn(|k| coef[k] as f32 * scale[k]);
    let v = idct(&f);
    std::array::from_fn(|k| (v[k] * out.scale + out.bias).floor().clamp(0.0, out.max) as u16)
}

pub(crate) fn fdct_load(pix: &[u16; 64], scale: f32) -> [f32; 64] {
    let p: [f32; 64] = std::array::from_fn(|k| pix[k] as f32 * scale - 256.0);
    fdct(&p)
}

pub(crate) fn quantise(
    f: &[f32],
    n: usize,
    inv: &[f32; 64],
    ac_rounding: f32,
    out: &mut [i32],
    mask: &mut [u64],
) {
    mask.fill(0);
    for (s, &inv) in inv.iter().enumerate() {
        let r = if s == 0 { 0.5 } else { ac_rounding };
        for b in 0..n {
            let i = n * s + b;
            let x = f[i] * inv;
            let m = (x.abs() + r).floor() as i32;
            let q = if x < 0.0 { -m } else { m };
            out[i] = q;
            mask[i / 64] |= ((q != 0) as u64) << (i % 64);
        }
    }
}
