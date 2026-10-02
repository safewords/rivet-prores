//! Apple-encoded frames: the decoder on a bitstream this crate did not
//! write.
//!
//! The sample is `AppleProRes422.mov` from Probe.dev's sample media library
//! (<https://docs.probe.dev/reference/sample-media-library>, "freely
//! accessible for testing purposes"): ProRes 422 HQ (`apch`), 720×486,
//! bottom field first, both quantisation matrices loaded, encoder
//! identifier `apl0`. It is 100 MB and carries no licence beyond that
//! sentence, so it is not committed; the first 4 MB holds six whole frames:
//!
//! ```sh
//! curl -r 0-4194303 -o head.mov \
//!   https://probelibrary.s3.amazonaws.com/samples/V-codecs/HCPA/AppleProRes422.mov
//! PRORES_SAMPLE=head.mov cargo test --release --test sample
//! ```
//!
//! Without `PRORES_SAMPLE` the test says so and passes; CI sets it.

mod common;

use common::psnr_all;
use prores::{ChromaFormat, Config, Decoder, Encoder, FrameHeader, Interlace, Profile};

/// Every whole frame in a MOV (or a prefix of one), found by its `'icpf'`
/// identifier and `frame_size`, without parsing the container.
fn frames(data: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut i = 4;
    while i + 4 <= data.len() {
        if &data[i..i + 4] == b"icpf" {
            let size = u32::from_be_bytes(data[i - 4..i].try_into().unwrap()) as usize;
            if size >= 28 && i - 4 + size <= data.len() {
                out.push(&data[i - 4..i - 4 + size]);
                i += size - 4;
                continue;
            }
        }
        i += 1;
    }
    out
}

#[test]
fn apple_encoded_422_hq_interlaced() {
    let Ok(path) = std::env::var("PRORES_SAMPLE") else {
        eprintln!("PRORES_SAMPLE is not set; skipping the Apple-encoded sample (see tests/sample.rs)");
        return;
    };
    let data = std::fs::read(&path).unwrap();
    let frames = frames(&data);
    assert!(frames.len() >= 6, "{} frames in {path}", frames.len());
    for (n, frame) in frames.iter().enumerate() {
        let hdr = FrameHeader::parse(frame).unwrap();
        assert_eq!((hdr.width, hdr.height), (720, 486));
        assert_eq!(hdr.chroma, ChromaFormat::Yuv422);
        assert_eq!(hdr.interlace, Interlace::BottomFieldFirst);
        assert_eq!(&hdr.encoder_identifier, b"apl0");
        assert!(hdr.luma_matrix.is_some() && hdr.chroma_matrix.is_some());

        let f = Decoder::new().decode(frame).unwrap();
        // A real picture: a ColorChecker held in front of a grey shirt.
        let mean = |p: &[u16]| p.iter().map(|&v| v as f64).sum::<f64>() / p.len() as f64;
        let (y, cb, cr) = (mean(f.plane(0)), mean(f.plane(1)), mean(f.plane(2)));
        assert!((400.0..520.0).contains(&y), "Y′ mean {y}");
        assert!((470.0..520.0).contains(&cb) && (510.0..560.0).contains(&cr), "chroma means {cb} {cr}");
        // Fields woven in the right order: neighbouring rows (the other
        // field) are closer than rows two apart only if each field sits on
        // its own rows and the two are in step.
        let (w, h) = (720usize, 486usize);
        let luma = f.plane(0);
        let diff = |gap: usize| {
            let mut s = 0u64;
            for r in 0..h - 2 {
                for x in 0..w {
                    s += (luma[r * w + x] as i64 - luma[(r + gap) * w + x] as i64).unsigned_abs();
                }
            }
            s as f64 / ((h - 2) * w) as f64
        };
        let (adjacent, two_apart) = (diff(1), diff(2));
        assert!(adjacent < two_apart, "rows: adjacent {adjacent}, two apart {two_apart}");

        // Through this crate's encoder at Apple's own frame size.
        let config = Config { target_frame_bytes: Some(frame.len()), ..Config::new(Profile::Hq) };
        let ours = Encoder::new(config).encode(&f).unwrap();
        let again = Decoder::new().decode(&ours).unwrap();
        let p = psnr_all(&f, &again);
        eprintln!(
            "frame {n}: {} bytes from Apple; re-encoded in {} bytes at {p:.2} dB against Apple's decode",
            frame.len(),
            ours.len()
        );
        assert!(ours.len() <= frame.len());
        assert!(p > 45.0, "{p}");
    }
}
