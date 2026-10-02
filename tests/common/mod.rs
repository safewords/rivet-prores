//! Test pictures and measurements shared by the integration tests.

#![allow(dead_code)]

use prores::{ChromaFormat, Frame};

/// A small deterministic PRNG (xorshift32).
pub struct Rng(pub u32);

impl Rng {
    pub fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
    /// Uniform in `-a..=a`.
    pub fn noise(&mut self, a: i32) -> i32 {
        (self.next() % (2 * a as u32 + 1)) as i32 - a
    }
}

/// A picture with what makes real pictures hard and easy: smooth gradients,
/// hard-edged shapes, fine periodic detail, and grain. Values stay inside
/// the video range (64–940 at 10 bits, scaled for other depths).
pub fn test_frame(width: u32, height: u32, chroma: ChromaFormat, bit_depth: u32, grain: i32, seed: u32) -> Frame {
    let mut f = Frame::new(width, height, chroma, bit_depth).unwrap();
    let mut rng = Rng(seed | 1);
    let shift = bit_depth as i32 - 10;
    let scale = |v: f64| -> u16 {
        let v = v.clamp(64.0, 940.0);
        let s = if shift >= 0 { v * (1 << shift) as f64 } else { v / (1 << -shift) as f64 };
        s.round() as u16
    };
    for plane in 0..3 {
        let p = f.planes[plane];
        let (sw, _) = chroma.subsampling();
        let xs = if plane == 0 { 1.0 } else { sw as f64 };
        let data = f.plane_mut(plane);
        for y in 0..p.height as usize {
            for x in 0..p.width as usize {
                let (fx, fy) = (x as f64 * xs, y as f64);
                let w = width as f64;
                let h = height as f64;
                let mut v = match plane {
                    0 => 200.0 + 500.0 * fx / w + 120.0 * (fx / 37.0).sin() * (fy / 23.0).cos(),
                    1 => 512.0 + 150.0 * (fx / w - 0.5) + 60.0 * (fy / 50.0).sin(),
                    _ => 512.0 - 120.0 * (fy / h - 0.5) + 50.0 * (fx / 70.0).cos(),
                };
                // A bright and a dark rectangle with hard edges.
                if fx > w * 0.15 && fx < w * 0.35 && fy > h * 0.2 && fy < h * 0.5 {
                    v = if plane == 0 { 900.0 } else { v + 150.0 };
                }
                if fx > w * 0.6 && fx < w * 0.8 && fy > h * 0.55 && fy < h * 0.85 {
                    v = if plane == 0 { 90.0 } else { v - 150.0 };
                }
                // Fine stripes in one corner.
                if fx < w * 0.25 && fy > h * 0.7 {
                    v += 150.0 * (fx * 1.3).sin();
                }
                if grain > 0 {
                    v += rng.noise(grain) as f64;
                }
                data[y * p.width as usize + x] = scale(v);
            }
        }
    }
    f
}

/// Peak signal-to-noise ratio of plane `plane` of `b` against `a`, in dB
/// (infinite when identical).
pub fn psnr(a: &Frame, b: &Frame, plane: usize) -> f64 {
    psnr_samples(a.plane(plane), b.plane(plane), a.bit_depth)
}

pub fn psnr_samples(a: &[u16], b: &[u16], bit_depth: u32) -> f64 {
    assert_eq!(a.len(), b.len());
    let mse: f64 = a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).powi(2)).sum::<f64>() / a.len() as f64;
    let peak = ((1u32 << bit_depth) - 1) as f64;
    if mse == 0.0 { f64::INFINITY } else { 10.0 * (peak * peak / mse).log10() }
}

/// Combined PSNR over all three planes, weighted by sample count.
pub fn psnr_all(a: &Frame, b: &Frame) -> f64 {
    psnr_samples(&a.data, &b.data, a.bit_depth)
}
