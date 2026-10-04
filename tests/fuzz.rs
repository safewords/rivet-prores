//! Property tests: malformed input is an error, never a panic, a hang or a
//! huge allocation; and random frames survive the round trip.

mod common;

use common::{psnr_all, test_frame};
use proptest::prelude::*;
use prores::{AlphaType, ChromaFormat, Config, Decoder, Encoder, FrameHeader, Interlace, Profile};
use std::sync::LazyLock;

/// Valid frames to mutate: 4:2:2 progressive, 4:4:4 interlaced with
/// 16-bit alpha and custom matrices, and 8-bit alpha.
static SEEDS: LazyLock<Vec<Vec<u8>>> = LazyLock::new(|| {
    let mut out = Vec::new();
    let f = test_frame(40, 24, ChromaFormat::Yuv422, 10, 20, 1);
    out.push(
        Encoder::new(Config::new(Profile::Standard))
            .encode(&f)
            .unwrap(),
    );
    let mut f = test_frame(33, 21, ChromaFormat::Yuv444, 12, 20, 2);
    f.interlace = Interlace::TopFieldFirst;
    f.alpha = Some((0..33 * 21).map(|i| (i * 7 % 4096) as u16).collect());
    let config = Config {
        luma_matrix: Some([5; 64]),
        chroma_matrix: Some([9; 64]),
        ..Config::new(Profile::P4444Xq)
    };
    out.push(Encoder::new(config).encode(&f).unwrap());
    f.interlace = Interlace::Progressive;
    let config = Config {
        alpha: AlphaType::Bits8,
        log2_slice_mbs: 0,
        ..Config::new(Profile::P4444)
    };
    out.push(Encoder::new(config).encode(&f).unwrap());
    out
});

/// Whatever happens, it is an `Ok` or an `Err`.
fn decode_anything(data: &[u8]) {
    let _ = FrameHeader::parse(data);
    let _ = Decoder::new().decode(data);
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 2000, ..ProptestConfig::default() })]

    #[test]
    fn arbitrary_bytes(data in proptest::collection::vec(any::<u8>(), 0..400)) {
        decode_anything(&data);
    }

    /// Arbitrary bytes behind a plausible start (`frame_size` = the length,
    /// `'icpf'`), so the header and slice parsers are reached.
    #[test]
    fn arbitrary_bytes_after_a_frame_identifier(mut data in proptest::collection::vec(any::<u8>(), 28..2000)) {
        let n = data.len() as u32;
        data[0..4].copy_from_slice(&n.to_be_bytes());
        data[4..8].copy_from_slice(b"icpf");
        decode_anything(&data);
    }

    #[test]
    fn valid_frames_with_bytes_changed(
        seed in 0usize..3,
        edits in proptest::collection::vec((any::<prop::sample::Index>(), any::<u8>()), 1..12),
    ) {
        let mut data = SEEDS[seed].clone();
        for (at, byte) in edits {
            let i = at.index(data.len());
            data[i] = byte;
        }
        decode_anything(&data);
    }

    #[test]
    fn valid_frames_with_bits_flipped(seed in 0usize..3, flips in proptest::collection::vec(any::<prop::sample::Index>(), 1..8)) {
        let mut data = SEEDS[seed].clone();
        for at in flips {
            let bit = at.index(data.len() * 8);
            data[bit / 8] ^= 0x80 >> (bit % 8);
        }
        decode_anything(&data);
    }

    #[test]
    fn valid_frames_cut_and_spliced(seed in 0usize..3, cut in any::<prop::sample::Index>(), junk in proptest::collection::vec(any::<u8>(), 0..64)) {
        let data = &SEEDS[seed];
        let at = cut.index(data.len());
        decode_anything(&data[..at]);
        let mut spliced = data[..at].to_vec();
        spliced.extend_from_slice(&junk);
        spliced.extend_from_slice(&data[at..]);
        decode_anything(&spliced);
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 48, ..ProptestConfig::default() })]

    /// Any size, format, field order and slice size round-trips, within the
    /// target size and at a PSNR that only a broken path would miss.
    #[test]
    fn random_frames_round_trip(
        w in 1u32..90,
        h in 1u32..70,
        is_444 in any::<bool>(),
        interlace in 0u8..3,
        log2 in 0u32..4,
        grain in 0i32..40,
        seed in any::<u32>(),
    ) {
        let (chroma, profile, depth) =
            if is_444 { (ChromaFormat::Yuv444, Profile::P4444, 12) } else { (ChromaFormat::Yuv422, Profile::Hq, 10) };
        let mut f = test_frame(w, h, chroma, depth, grain, seed);
        f.interlace = Interlace::from_code(interlace).unwrap();
        let config = Config { log2_slice_mbs: log2, target_frame_bytes: Some(1 << 30), ..Config::new(profile) };
        let packet = Encoder::new(config).encode(&f).unwrap();
        let d = Decoder::new().decode(&packet).unwrap();
        prop_assert_eq!(&d.planes, &f.planes);
        let p = psnr_all(&f, &d);
        prop_assert!(p > 45.0, "PSNR {}", p);
    }
}
