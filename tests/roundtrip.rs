//! Encode → decode round trips: every profile, both chroma formats, both
//! scan orders, odd sizes, alpha, custom matrices; PSNR and frame size
//! against the profile's target.

mod common;

use common::{Rng, psnr, psnr_all, psnr_samples, test_frame};
use prores::{
    AlphaType, ChromaFormat, Config, Decoder, Encoder, Error, Frame, FrameHeader, Interlace, Metadata, Profile,
};

fn round_trip(frame: &Frame, config: Config) -> (Vec<u8>, Frame) {
    let packet = Encoder::new(config).encode(frame).unwrap();
    let decoded = Decoder::with_bit_depth(frame.bit_depth).unwrap().decode(&packet).unwrap();
    assert_eq!(decoded.width, frame.width);
    assert_eq!(decoded.height, frame.height);
    assert_eq!(decoded.chroma, frame.chroma);
    assert_eq!(decoded.interlace, frame.interlace);
    assert_eq!(decoded.metadata, frame.metadata);
    assert_eq!(decoded.planes, frame.planes);
    assert_eq!(decoded.alpha.is_some(), frame.alpha.is_some());
    (packet, decoded)
}

fn depth_for(profile: Profile) -> u32 {
    if profile.chroma() == ChromaFormat::Yuv444 { 12 } else { 10 }
}

/// A target large enough that the finest quantiser is chosen.
fn unconstrained(profile: Profile) -> Config {
    Config { target_frame_bytes: Some(1 << 30), ..Config::new(profile) }
}

/// Each profile at 1920×1080 on a grainy picture: within its target size
/// (and not far under it), at a PSNR that rises with the profile.
#[test]
fn every_profile_meets_its_target_at_1080p() {
    let mut last = 0.0;
    for profile in Profile::ALL {
        let frame = test_frame(1920, 1080, profile.chroma(), depth_for(profile), 24, 7);
        let (packet, decoded) = round_trip(&frame, Config::new(profile));
        let target = profile.target_frame_bytes(1920, 1080);
        let p = psnr_all(&frame, &decoded);
        let ratio = packet.len() as f64 / target as f64;
        eprintln!(
            "{profile:?} 1080p grain: {} bytes, target {target} ({:.2}%), PSNR {p:.2} dB (Y {:.2}, Cb {:.2}, Cr {:.2})",
            packet.len(),
            100.0 * ratio,
            psnr(&frame, &decoded, 0),
            psnr(&frame, &decoded, 1),
            psnr(&frame, &decoded, 2),
        );
        assert!(packet.len() <= target, "{profile:?} overshoots its target");
        assert!(ratio > 0.98, "{profile:?} undershoots its target: {ratio}");
        assert!(p > 36.0, "{profile:?}: PSNR {p}");
        assert!(p > last - 0.25, "{profile:?} is worse than the profile below it");
        last = p;
    }
}

/// A clean picture (no grain) is far easier: the 422 profiles reach high
/// PSNR, and stay within target.
#[test]
fn clean_picture_quality_by_profile() {
    let floors = [
        (Profile::Proxy, 40.0),
        (Profile::Lt, 45.0),
        (Profile::Standard, 47.0),
        (Profile::Hq, 50.0),
        (Profile::P4444, 50.0),
        (Profile::P4444Xq, 55.0),
    ];
    for (profile, floor) in floors {
        let frame = test_frame(1280, 720, profile.chroma(), depth_for(profile), 0, 3);
        let (packet, decoded) = round_trip(&frame, Config::new(profile));
        let target = profile.target_frame_bytes(1280, 720);
        let p = psnr_all(&frame, &decoded);
        eprintln!("{profile:?} 720p clean: {} bytes, target {target}, PSNR {p:.2} dB", packet.len());
        assert!(packet.len() <= target);
        assert!(p > floor, "{profile:?}: PSNR {p} below {floor}");
    }
}

/// With the finest quantiser and the smallest weights, coding is all but
/// lossless.
#[test]
fn finest_quantiser_is_nearly_lossless() {
    for (chroma, depth, profile) in [(ChromaFormat::Yuv422, 10, Profile::Hq), (ChromaFormat::Yuv444, 12, Profile::P4444Xq)] {
        let frame = test_frame(256, 128, chroma, depth, 30, 11);
        let config = Config { luma_matrix: Some([2; 64]), chroma_matrix: Some([2; 64]), ..unconstrained(profile) };
        let (packet, decoded) = round_trip(&frame, config);
        let hdr = FrameHeader::parse(&packet).unwrap();
        assert_eq!(hdr.luma_matrix, Some([2; 64]));
        let max_err = frame.data.iter().zip(&decoded.data).map(|(&a, &b)| (a as i32 - b as i32).abs()).max().unwrap();
        let p = psnr_all(&frame, &decoded);
        eprintln!("{profile:?} finest: PSNR {p:.2} dB, max error {max_err}");
        // The finest step is 1/4 in the 9-bit domain of §7.5.1: half an
        // LSB at 10 bits, two LSBs at 12.
        assert!(max_err <= if depth == 10 { 1 } else { 4 }, "max error {max_err}");
        assert!(p > 60.0);
    }
}

/// Sizes that are not multiples of 16, down to one pixel, with every slice
/// size, progressive and both field orders, both chroma formats.
#[test]
fn odd_sizes_slice_sizes_and_field_orders() {
    let sizes = [(1, 1), (2, 2), (17, 9), (31, 33), (48, 16), (100, 37), (129, 75), (250, 19)];
    for (w, h) in sizes {
        for (chroma, profile) in [(ChromaFormat::Yuv422, Profile::Hq), (ChromaFormat::Yuv444, Profile::P4444)] {
            for interlace in [Interlace::Progressive, Interlace::TopFieldFirst, Interlace::BottomFieldFirst] {
                for log2 in 0..4 {
                    let mut frame = test_frame(w, h, chroma, depth_for(profile), 6, w * 131 + h);
                    frame.interlace = interlace;
                    let config = Config { log2_slice_mbs: log2, ..unconstrained(profile) };
                    let (packet, decoded) = round_trip(&frame, config);
                    let hdr = FrameHeader::parse(&packet).unwrap();
                    assert_eq!(hdr.picture_count(), if interlace == Interlace::Progressive { 1 } else { 2 });
                    let p = psnr_all(&frame, &decoded);
                    assert!(p > 50.0, "{w}×{h} {chroma:?} {interlace:?} log2 {log2}: PSNR {p}");
                }
            }
        }
    }
}

/// The two fields of an interlaced frame are coded apart: a frame whose
/// fields differ wildly (every other row black and white) codes well as
/// fields — the even rows land on the even rows.
#[test]
fn fields_land_on_their_own_rows() {
    for interlace in [Interlace::TopFieldFirst, Interlace::BottomFieldFirst] {
        let mut frame = Frame::new(64, 35, ChromaFormat::Yuv422, 10).unwrap();
        frame.interlace = interlace;
        for plane in 0..3 {
            let w = frame.planes[plane].width as usize;
            for (i, s) in frame.plane_mut(plane).iter_mut().enumerate() {
                let (x, y) = (i % w, i / w);
                *s = if y % 2 == 0 { 100 + x as u16 } else { 900 - x as u16 };
            }
        }
        let (packet, decoded) = round_trip(&frame, Config::new(Profile::Standard));
        let p = psnr_all(&frame, &decoded);
        eprintln!("{interlace:?} striped fields: {} bytes, PSNR {p:.2} dB", packet.len());
        assert!(p > 50.0, "{interlace:?}: PSNR {p}");
        // Coded progressively, the same stripes are the worst case for a DCT.
        frame.interlace = Interlace::Progressive;
        let (progressive, _) = round_trip(&frame, unconstrained(Profile::Standard));
        let (fields, _) = {
            frame.interlace = interlace;
            round_trip(&frame, unconstrained(Profile::Standard))
        };
        assert!(fields.len() * 3 < progressive.len());
    }
}

/// Alpha is lossless: 16-bit alpha carries any input depth exactly, and
/// 8-bit alpha carries 8-bit input exactly.
#[test]
fn alpha_is_lossless() {
    let cases = [(12, AlphaType::Bits16, 12), (16, AlphaType::Bits16, 16), (8, AlphaType::Bits8, 8), (10, AlphaType::Bits16, 10)];
    for (depth, alpha_type, out_depth) in cases {
        for interlace in [Interlace::Progressive, Interlace::TopFieldFirst] {
            let mut frame = test_frame(97, 41, ChromaFormat::Yuv444, depth, 4, 5);
            frame.interlace = interlace;
            let max = (1u32 << depth) - 1;
            let mut rng = Rng(99);
            let mut alpha = Vec::new();
            for y in 0..41u32 {
                for x in 0..97u32 {
                    alpha.push(match (x / 20, y / 10) {
                        (0, _) => 0,
                        (1, _) => max,
                        (2, _) => (x * max / 97) as u16 as u32,
                        (3, 0..=1) => rng.next() % (max + 1),
                        _ => max / 2,
                    } as u16);
                }
            }
            frame.alpha = Some(alpha.clone());
            let config = Config { alpha: alpha_type, ..Config::new(Profile::P4444) };
            let packet = Encoder::new(config).encode(&frame).unwrap();
            let hdr = FrameHeader::parse(&packet).unwrap();
            assert_eq!(hdr.alpha, alpha_type);
            assert_eq!(hdr.bitstream_version, 1);
            let decoded = Decoder::with_bit_depth(out_depth).unwrap().decode(&packet).unwrap();
            assert_eq!(decoded.alpha.as_ref().unwrap(), &alpha, "{depth}-bit alpha as {alpha_type:?} {interlace:?}");
            let p = psnr_all(&frame, &decoded);
            assert!(p > 40.0, "{depth}-bit {interlace:?}: PSNR {p}");
        }
    }
}

/// Tiny frames get the profile's coded data per macroblock, with their
/// headers on top: a 16×16 frame is one macroblock whose frame, picture and
/// slice headers (44 bytes) once took most of an area-scaled share (113
/// bytes for HQ), leaving the coarsest quantisers. The frame stays within
/// its target, and quality is well above what the area-scaled target gave.
#[test]
fn tiny_frames_get_their_share_per_macroblock() {
    let per_mb = |p: Profile| {
        (p.target_frame_bytes(1920, 1080) - p.target_frame_bytes(1920, 1080 - 16 * 67)) as f64 / (120.0 * 67.0)
    };
    for profile in [Profile::Proxy, Profile::Standard, Profile::Hq, Profile::P4444] {
        let mb = per_mb(profile);
        for (w, h) in [(16u32, 16u32), (32, 32), (17, 9), (33, 31), (8, 40), (1, 1)] {
            let frame = test_frame(w, h, profile.chroma(), depth_for(profile), 8, w * 31 + h);
            let (packet, decoded) = round_trip(&frame, Config::new(profile));
            let target = profile.target_frame_bytes(w, h);
            let mbs = (w.div_ceil(16) * h.div_ceil(16)) as f64;
            // The old rule: the 1080 frame's bytes scaled by area.
            let area = (profile.target_frame_bytes(1920, 1080) as f64 * (w * h) as f64 / (1920.0 * 1080.0)) as usize;
            let (_, old) = round_trip(&frame, Config { target_frame_bytes: Some(area), ..Config::new(profile) });
            let (p, p_old) = (psnr_all(&frame, &decoded), psnr_all(&frame, &old));
            eprintln!(
                "{profile:?} {w}×{h}: {} bytes (target {target}, area-scaled {area}), PSNR {p:.2} dB (area-scaled {p_old:.2} dB)",
                packet.len()
            );
            assert!(packet.len() <= target, "{profile:?} {w}×{h} overshoots");
            assert!((target as f64) > mb * mbs, "{profile:?} {w}×{h}: target {target} under {mb:.0} per macroblock");
            assert!(p >= p_old, "{profile:?} {w}×{h}: {p} < {p_old}");
        }
    }
}

/// 8-bit alpha at sizes off the macroblock grid: the bottom slices' alpha
/// covers whole macroblocks (16 rows), and a run of equal values that
/// carries on below the picture is read (a 640×360 4444 XQ frame with a
/// vertical alpha ramp failed with "an alpha run passes the end of the
/// slice" when the decoder expected only the picture's rows).
#[test]
fn alpha8_off_the_macroblock_grid() {
    for (w, h) in [(640u32, 360u32), (100, 40), (33, 17), (16, 8), (24, 24)] {
        for interlace in [Interlace::Progressive, Interlace::BottomFieldFirst] {
            let mut frame = test_frame(w, h, ChromaFormat::Yuv444, 8, 2, w ^ h);
            frame.interlace = interlace;
            // A vertical ramp: every row one value, so runs span rows.
            let alpha: Vec<u16> = (0..h).flat_map(|y| (0..w).map(move |_| (y * 255 / (h - 1)) as u16)).collect();
            frame.alpha = Some(alpha.clone());
            for profile in [Profile::P4444, Profile::P4444Xq] {
                let config = Config { alpha: AlphaType::Bits8, ..Config::new(profile) };
                let packet = Encoder::new(config).encode(&frame).unwrap();
                let decoded = Decoder::with_bit_depth(8).unwrap().decode(&packet).unwrap();
                assert_eq!(decoded.alpha.as_ref().unwrap(), &alpha, "{w}×{h} {interlace:?} {profile:?}");
            }
        }
    }
}

/// 16-bit alpha decoded at 12 bits follows §7.5.2's rounding.
#[test]
fn alpha_converts_to_the_output_depth() {
    let mut frame = test_frame(32, 16, ChromaFormat::Yuv444, 16, 0, 1);
    let alpha: Vec<u16> = (0..512u32).map(|i| (i * 128 + i % 7) as u16).collect();
    frame.alpha = Some(alpha.clone());
    let packet = Encoder::new(Config::new(Profile::P4444)).encode(&frame).unwrap();
    let decoded = Decoder::new().decode(&packet).unwrap();
    assert_eq!(decoded.bit_depth, 12);
    for (a, d) in alpha.iter().zip(decoded.alpha.unwrap()) {
        let want = (4095.0 * *a as f64 / 65535.0).round() as u16;
        assert_eq!(d, want);
    }
}

/// Custom matrices are written, parsed and used; a chroma matrix absent
/// means the luma one.
#[test]
fn custom_quantisation_matrices() {
    let mut luma = [0u8; 64];
    let mut chroma = [0u8; 64];
    for v in 0..8 {
        for u in 0..8 {
            luma[8 * v + u] = (4 + 2 * (u + v)) as u8;
            chroma[8 * v + u] = (6 + 3 * (u + v)) as u8;
        }
    }
    let frame = test_frame(320, 180, ChromaFormat::Yuv422, 10, 8, 21);
    for (l, c) in [(Some(luma), Some(chroma)), (Some(luma), None), (None, Some(chroma))] {
        let config = Config { luma_matrix: l, chroma_matrix: c, ..Config::new(Profile::Hq) };
        let (packet, decoded) = round_trip(&frame, config);
        let hdr = FrameHeader::parse(&packet).unwrap();
        assert_eq!((hdr.luma_matrix, hdr.chroma_matrix), (l, c));
        assert_eq!(hdr.header_size as usize, 20 + 64 * (l.is_some() as usize + c.is_some() as usize));
        let p = psnr_all(&frame, &decoded);
        assert!(p > 40.0, "{p}");
        assert!(packet.len() <= Profile::Hq.target_frame_bytes(320, 180));
    }
}

#[test]
fn metadata_and_header_fields_survive() {
    let mut frame = test_frame(720, 486, ChromaFormat::Yuv422, 10, 10, 2);
    frame.interlace = Interlace::BottomFieldFirst;
    frame.metadata = Metadata {
        aspect_ratio: 2,
        frame_rate_code: 4,
        color_primaries: 6,
        transfer_characteristic: 1,
        matrix_coefficients: 6,
    };
    let (packet, decoded) = round_trip(&frame, Config::new(Profile::Standard));
    assert_eq!(decoded.metadata.frame_rate(), Some((30000, 1001)));
    let hdr = FrameHeader::parse(&packet).unwrap();
    assert_eq!(hdr.frame_size as usize, packet.len());
    assert_eq!(hdr.bitstream_version, 0);
    assert_eq!(&hdr.encoder_identifier, b"rivt");
    assert_eq!((hdr.width, hdr.height), (720, 486));
    assert_eq!(hdr.picture_height(0), 243);
    let target = Profile::Standard.target_frame_bytes(720, 486);
    eprintln!(
        "Standard 720×486 BFF: {} bytes, target {target}, PSNR {:.2} dB",
        packet.len(),
        psnr_all(&frame, &decoded)
    );
    assert!(packet.len() <= target && packet.len() as f64 > 0.98 * target as f64);
}

/// Any input depth from 8 to 16 is taken; output at another depth is the
/// same picture rescaled.
#[test]
fn bit_depths_in_and_out() {
    for depth in [8, 9, 10, 12, 14, 16] {
        let frame = test_frame(64, 64, ChromaFormat::Yuv422, depth, 0, 4);
        let (_, decoded) = round_trip(&frame, unconstrained(Profile::Hq));
        let p = psnr_all(&frame, &decoded);
        assert!(p > 50.0, "{depth}-bit: {p}");
    }
    let frame = test_frame(64, 64, ChromaFormat::Yuv422, 10, 0, 4);
    let packet = Encoder::new(unconstrained(Profile::Hq)).encode(&frame).unwrap();
    let at16 = Decoder::with_bit_depth(16).unwrap().decode(&packet).unwrap();
    let at10 = Decoder::new().decode(&packet).unwrap();
    for (a, b) in at10.data.iter().zip(&at16.data) {
        assert!((*a as i32 - ((*b as i32 + 32) >> 6)).abs() <= 1);
    }
    assert!(Decoder::with_bit_depth(7).is_err());
    assert!(Decoder::with_bit_depth(17).is_err());
}

#[test]
fn encoder_refuses_what_it_cannot_code() {
    let f422 = Frame::new(32, 32, ChromaFormat::Yuv422, 10).unwrap();
    let f444 = Frame::new(32, 32, ChromaFormat::Yuv444, 12).unwrap();
    let is_config = |r: Result<Vec<u8>, Error>| matches!(r, Err(Error::Config(_)));
    assert!(is_config(Encoder::new(Config::new(Profile::Hq)).encode(&f444)));
    assert!(is_config(Encoder::new(Config::new(Profile::P4444)).encode(&f422)));
    let mut with_alpha = f422.clone();
    with_alpha.alpha = Some(vec![0; 32 * 32]);
    assert!(is_config(Encoder::new(Config::new(Profile::Hq)).encode(&with_alpha)));
    let mut short_alpha = f444.clone();
    short_alpha.alpha = Some(vec![0; 10]);
    assert!(is_config(Encoder::new(Config::new(Profile::P4444)).encode(&short_alpha)));
    let bad_matrix = Config { luma_matrix: Some([1; 64]), ..Config::new(Profile::Hq) };
    assert!(is_config(Encoder::new(bad_matrix).encode(&f422)));
    let bad_slices = Config { log2_slice_mbs: 4, ..Config::new(Profile::Hq) };
    assert!(is_config(Encoder::new(bad_slices).encode(&f422)));
    let mut truncated = f422.clone();
    truncated.data.truncate(100);
    assert!(is_config(Encoder::new(Config::new(Profile::Hq)).encode(&truncated)));
    assert!(Frame::new(0, 1, ChromaFormat::Yuv422, 10).is_err());
    assert!(Frame::new(65536, 1, ChromaFormat::Yuv422, 10).is_err());
    assert!(Frame::new(1, 1, ChromaFormat::Yuv422, 7).is_err());
    // A 4444 frame with alpha and the alpha switched off codes no alpha.
    let mut a = f444.clone();
    a.alpha = Some(vec![7; 32 * 32]);
    let packet = Encoder::new(Config { alpha: AlphaType::None, ..Config::new(Profile::P4444) }).encode(&a).unwrap();
    assert!(Decoder::new().decode(&packet).unwrap().alpha.is_none());
}

/// A flat picture costs almost nothing, at any profile, and decodes flat.
#[test]
fn flat_pictures() {
    for profile in Profile::ALL {
        let frame = Frame::new(1920, 1080, profile.chroma(), depth_for(profile)).unwrap();
        let (packet, decoded) = round_trip(&frame, Config::new(profile));
        assert_eq!(decoded.data, frame.data);
        assert!(packet.len() < 60_000, "{profile:?}: {}", packet.len());
    }
}

/// Black and white extremes clip correctly rather than wrapping.
#[test]
fn extremes_clip_instead_of_wrapping() {
    let mut frame = Frame::new(64, 32, ChromaFormat::Yuv422, 10).unwrap();
    for plane in 0..3 {
        let w = frame.planes[plane].width as usize;
        for (i, s) in frame.plane_mut(plane).iter_mut().enumerate() {
            *s = if ((i % w) / 3).is_multiple_of(2) { 0 } else { 1023 };
        }
    }
    let (_, decoded) = round_trip(&frame, Config::new(Profile::Proxy));
    let p = psnr_samples(&frame.data, &decoded.data, 10);
    assert!(p > 20.0, "{p}");
}
