//! End-to-end throughput: encode and decode synthetic 720p / 1080p frames
//! and print frames per second (best of N runs) and a hash of the output,
//! so runs at different thread counts, SIMD levels (`PRORES_FORCE_SCALAR=1`)
//! or commits can be checked for identical bytes.
//!
//! ```sh
//! cargo run --release --example prores_bench -- [720p|1080p] [hq|4444a] [threads] [runs]
//! cargo run --release --example prores_bench -- file <frames.mov> [threads]
//! ```
//!
//! `file` decodes every ProRes frame in a file (found by `'icpf'`, without
//! parsing the container) and prints a hash of the pictures.
//!
//! `threads` 0 means one per core. The pictures are gradients, hard edges,
//! fine stripes and grain (the integration tests' generator), different in
//! each of the four frames.

use prores::{AlphaType, ChromaFormat, Config, Decoder, Encoder, Frame, Profile};
use std::time::Instant;

fn picture(
    width: u32,
    height: u32,
    chroma: ChromaFormat,
    depth: u32,
    seed: u32,
    alpha: bool,
) -> Frame {
    let mut f = Frame::new(width, height, chroma, depth).unwrap();
    let mut rng = seed | 1;
    let mut noise = |a: i32| {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        (rng % (2 * a as u32 + 1)) as i32 - a
    };
    let shift = depth as i32 - 10;
    let phase = seed as f64 * 0.37;
    for plane in 0..3 {
        let p = f.planes[plane];
        let xs = if plane == 0 {
            1.0
        } else {
            chroma.subsampling().0 as f64
        };
        let (w, h) = (width as f64, height as f64);
        let data = f.plane_mut(plane);
        for y in 0..p.height as usize {
            for x in 0..p.width as usize {
                let (fx, fy) = (x as f64 * xs, y as f64);
                let mut v = match plane {
                    0 => {
                        200.0
                            + 500.0 * fx / w
                            + 120.0 * (fx / 37.0 + phase).sin() * (fy / 23.0).cos()
                    }
                    1 => 512.0 + 150.0 * (fx / w - 0.5) + 60.0 * (fy / 50.0 + phase).sin(),
                    _ => 512.0 - 120.0 * (fy / h - 0.5) + 50.0 * (fx / 70.0).cos(),
                };
                if fx > w * 0.15 && fx < w * 0.35 && fy > h * 0.2 && fy < h * 0.5 {
                    v = if plane == 0 { 900.0 } else { v + 150.0 };
                }
                if fx < w * 0.25 && fy > h * 0.7 {
                    v += 150.0 * (fx * 1.3).sin();
                }
                v += noise(6) as f64;
                let v = v.clamp(64.0, 940.0);
                let s = if shift >= 0 {
                    v * (1 << shift) as f64
                } else {
                    v / (1 << -shift) as f64
                };
                data[y * p.width as usize + x] = s.round() as u16;
            }
        }
    }
    if alpha {
        let (w, h) = (width as usize, height as usize);
        let max = (1u32 << depth) - 1;
        // A soft-edged ellipse: runs of opaque and transparent with a ramp.
        let a = (0..h)
            .flat_map(|y| {
                (0..w).map(move |x| {
                    let dx = (x as f64 - w as f64 / 2.0) / (w as f64 * 0.4);
                    let dy = (y as f64 - h as f64 / 2.0) / (h as f64 * 0.4);
                    let d = (dx * dx + dy * dy).sqrt();
                    ((1.0 - (d - 0.9) * 10.0).clamp(0.0, 1.0) * max as f64).round() as u16
                })
            })
            .collect();
        f.alpha = Some(a);
    }
    f
}

fn fnv(h: &mut u64, bytes: impl IntoIterator<Item = u8>) {
    for b in bytes {
        *h ^= b as u64;
        *h = h.wrapping_mul(0x100_0000_01b3);
    }
}

/// Hashes the decoded pictures of every frame in a file.
fn decode_file(path: &str, threads: usize) {
    let data = std::fs::read(path).unwrap();
    let decoder = Decoder::new().with_threads(threads); // THREADS-API
    let (mut h, mut count, mut i) = (0xcbf2_9ce4_8422_2325u64, 0, 4);
    while i + 4 <= data.len() {
        if &data[i..i + 4] == b"icpf" {
            let size = u32::from_be_bytes(data[i - 4..i].try_into().unwrap()) as usize;
            if size >= 28 && i - 4 + size <= data.len() {
                let f = decoder.decode(&data[i - 4..i - 4 + size]).unwrap();
                fnv(&mut h, f.data.iter().flat_map(|s| s.to_le_bytes()));
                count += 1;
                i += size - 4;
                continue;
            }
        }
        i += 1;
    }
    println!("{path}: {count} frames, pictures {h:016x}");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("file") {
        return decode_file(&args[1], args.get(2).map_or(1, |s| s.parse().unwrap()));
    }
    let size = args.first().map(String::as_str).unwrap_or("1080p");
    let kind = args.get(1).map(String::as_str).unwrap_or("hq");
    let threads: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1);
    let runs: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(5);
    let (w, h) = match size {
        "720p" => (1280, 720),
        "1080p" => (1920, 1080),
        "2160p" => (3840, 2160),
        _ => panic!("size: 720p, 1080p or 2160p"),
    };
    let (profile, chroma, depth, alpha) = match kind {
        "hq" => (Profile::Hq, ChromaFormat::Yuv422, 10, false),
        "4444a" => (Profile::P4444, ChromaFormat::Yuv444, 12, true),
        _ => panic!("kind: hq or 4444a"),
    };
    let frames: Vec<Frame> = (0..4)
        .map(|s| picture(w, h, chroma, depth, 1 + s, alpha))
        .collect();
    let mut config = Config::new(profile);
    config.alpha = AlphaType::Bits16;
    config.threads = threads; // THREADS-API
    let encoder = Encoder::new(config);
    let decoder = Decoder::new().with_threads(threads); // THREADS-API

    // Each run codes the four frames `reps` times; the best run counts.
    let reps = 4;
    let mut packets = Vec::new();
    let mut best_enc = f64::MAX;
    for _ in 0..runs {
        let t = Instant::now();
        for _ in 0..reps {
            packets = frames.iter().map(|f| encoder.encode(f).unwrap()).collect();
        }
        best_enc = best_enc.min(t.elapsed().as_secs_f64());
    }
    let mut best_dec = f64::MAX;
    let mut decoded = Vec::new();
    for _ in 0..runs {
        let t = Instant::now();
        for _ in 0..reps {
            decoded = packets.iter().map(|p| decoder.decode(p).unwrap()).collect();
        }
        best_dec = best_dec.min(t.elapsed().as_secs_f64());
    }
    let n = (frames.len() * reps) as f64;
    let mut he = 0xcbf2_9ce4_8422_2325u64;
    for p in &packets {
        fnv(&mut he, p.iter().copied());
    }
    let mut hd = 0xcbf2_9ce4_8422_2325u64;
    for f in &decoded {
        fnv(&mut hd, f.data.iter().flat_map(|s| s.to_le_bytes()));
        if let Some(a) = &f.alpha {
            fnv(&mut hd, a.iter().flat_map(|s| s.to_le_bytes()));
        }
    }
    let bytes: usize = packets.iter().map(Vec::len).sum();
    println!(
        "{size} {kind} threads={threads}: encode {:.1} fps, decode {:.1} fps, {} bytes/frame, packets {he:016x}, pictures {hd:016x}",
        n / best_enc,
        n / best_dec,
        bytes / packets.len()
    );
}
