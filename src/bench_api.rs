//! Per-kernel timings for `examples/prores_kernels.rs`: not part of the API.
//!
//! Each function runs one kernel `iters` times on realistic data (blocks
//! of the integration tests' kind of picture, transformed and quantised at
//! a 422 HQ-like quantiser) and returns nanoseconds per call, best of five.

use crate::bits::BitWriter;
use crate::dsp::{self, Isa, Output};
use crate::encode::{Quantised, coefficient_bits, encode_coefficients_masked};
use crate::tables::PROGRESSIVE_SCAN;
use std::hint::black_box;
use std::time::Instant;

/// The SIMD levels this CPU runs, by name, scalar first.
pub fn levels() -> Vec<&'static str> {
    Isa::available().into_iter().map(name).collect()
}

fn name(isa: Isa) -> &'static str {
    match isa {
        Isa::Scalar => "scalar",
        #[cfg(target_arch = "x86_64")]
        Isa::Sse41 => "sse4.1",
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => "avx2",
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => "neon",
    }
}

fn isa(level: &str) -> Isa {
    Isa::available().into_iter().find(|&i| name(i) == level).expect("an available level")
}

fn best_ns(iters: usize, mut f: impl FnMut()) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        best = best.min(t.elapsed().as_nanos() as f64 / iters as f64);
    }
    best
}

/// 8×8 blocks of 10-bit samples from a picture with edges, gradients,
/// stripes and grain.
fn sample_blocks(count: usize) -> Vec<[u16; 64]> {
    let mut seed = 0x2545_f491u32;
    (0..count)
        .map(|b| {
            std::array::from_fn(|k| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                let (x, y) = ((b * 8 + k % 8) as f64, (k / 8 + 8 * (b / 240)) as f64);
                let mut v = 200.0 + 0.3 * x + 120.0 * (x / 37.0).sin() * (y / 23.0).cos();
                if b % 7 == 3 {
                    v += 150.0 * (x * 1.3).sin();
                }
                v += (seed % 13) as f64 - 6.0;
                v.clamp(64.0, 940.0) as u16
            })
        })
        .collect()
}

/// A slice component of `n` blocks: scanned-order DCT coefficients, its
/// quantised form at quantiser 6 (weights 4: about the bytes per component
/// of 422 HQ at 1080p) and its coded bytes.
fn component(n: usize) -> (Vec<f32>, [f32; 64], Quantised, Vec<u8>) {
    let blocks = sample_blocks(n);
    let mut f = vec![0f32; 64 * n];
    for (b, pix) in blocks.iter().enumerate() {
        let c = dsp::scalar::fdct_load(pix, 0.5);
        for k in 0..64 {
            f[n * PROGRESSIVE_SCAN[k] as usize + b] = c[k];
        }
    }
    let inv = [8.0 / 24.0; 64];
    let mut q = Quantised::default();
    q.quantise(Isa::Scalar, &f, &inv);
    let mut bytes = Vec::new();
    let mut w = BitWriter::new(&mut bytes);
    encode_coefficients_masked(&mut w, &q, n);
    w.finish();
    (f, inv, q, bytes)
}

/// Inverse quantisation, IDCT and sample conversion of one block.
pub fn idct_put(level: &str, iters: usize) -> f64 {
    let isa = isa(level);
    let (_, _, q, _) = component(32);
    let blocks: Vec<[i32; 64]> =
        (0..32).map(|b| std::array::from_fn(|k| q.values[32 * PROGRESSIVE_SCAN[k] as usize + b])).collect();
    let scale = [2.0f32; 64];
    let out = Output::new(10);
    let mut i = 0;
    best_ns(iters, || {
        black_box(dsp::idct_put(isa, black_box(&blocks[i % 32]), &scale, out));
        i += 1;
    })
}

/// Sample conversion and FDCT of one block.
pub fn fdct_load(level: &str, iters: usize) -> f64 {
    let isa = isa(level);
    let blocks = sample_blocks(32);
    let mut i = 0;
    best_ns(iters, || {
        black_box(dsp::fdct_load(isa, black_box(&blocks[i % 32]), 0.5));
        i += 1;
    })
}

/// Quantisation of a 32-block slice component (a luma slice of 8
/// macroblocks), with its non-zero bitmap.
pub fn quantise(level: &str, iters: usize) -> f64 {
    let isa = isa(level);
    let (f, inv, _, _) = component(32);
    let mut q = Quantised::default();
    best_ns(iters, || {
        q.quantise(isa, black_box(&f), &inv);
        black_box(&q);
    })
}

/// Counting the coded bits of a quantised 32-block component.
pub fn count_bits(iters: usize) -> f64 {
    let (_, _, q, _) = component(32);
    best_ns(iters, || {
        black_box(coefficient_bits(black_box(&q), 32));
    })
}

/// Writing a quantised 32-block component.
pub fn write_coefficients(iters: usize) -> f64 {
    let (_, _, q, _) = component(32);
    let mut out = Vec::with_capacity(1 << 16);
    best_ns(iters, || {
        out.clear();
        let mut w = BitWriter::new(&mut out);
        encode_coefficients_masked(&mut w, black_box(&q), 32);
        w.finish();
        black_box(&out);
    })
}

/// Entropy decoding of a 32-block component into raster-order blocks.
pub fn decode_coefficients(iters: usize) -> f64 {
    let (_, _, _, bytes) = component(32);
    let unscan = crate::decode::inverse_scan(&PROGRESSIVE_SCAN);
    let mut out = vec![0i32; 64 * 32];
    best_ns(iters, || {
        out.fill(0);
        crate::decode::decode_coefficients_raster(black_box(&bytes), 32, &unscan, &mut out).unwrap();
        black_box(&out);
    })
}

/// Coded bytes per 32-block component of the data the entropy kernels use.
pub fn component_bytes() -> usize {
    component(32).3.len()
}

/// `Frame::to_le_bytes` of a 1920×1080 4:2:2 frame.
pub fn to_le_bytes(iters: usize) -> f64 {
    let f = crate::Frame::new(1920, 1080, crate::ChromaFormat::Yuv422, 10).unwrap();
    best_ns(iters, || {
        black_box(black_box(&f).to_le_bytes());
    })
}
