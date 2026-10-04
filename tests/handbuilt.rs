//! Frames assembled byte by byte and bit by bit from RDD 36's syntax tables
//! and codebooks — not by this crate's encoder — and the samples §7 says
//! they decode to, worked out by hand.

use prores::{AlphaType, ChromaFormat, Decoder, Error, FrameHeader, Interlace};

/// Bytes from a string of '0'/'1' (spaces ignored), padded with zero bits
/// to a whole byte (the `zero_bit`s of §5.3.2).
fn bits(s: &str) -> Vec<u8> {
    let b: Vec<u8> = s.bytes().filter(|c| *c != b' ').map(|c| c - b'0').collect();
    b.chunks(8)
        .map(|c| {
            c.iter()
                .enumerate()
                .fold(0u8, |acc, (i, &bit)| acc | (bit << (7 - i)))
        })
        .collect()
}

/// A frame header (§5.1.1) of `extra` more bytes than the 20 the syntax
/// needs (a version variant's informative data, which a decoder must skip
/// by `frame_header_size`).
fn frame_header(
    w: u16,
    h: u16,
    chroma: u8,
    interlace: u8,
    alpha: u8,
    version: u8,
    extra: usize,
) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(&((20 + extra) as u16).to_be_bytes());
    v.push(0); // reserved
    v.push(version);
    v.extend_from_slice(b"hand");
    v.extend_from_slice(&w.to_be_bytes());
    v.extend_from_slice(&h.to_be_bytes());
    v.push((chroma << 6) | (interlace << 2));
    v.push((1 << 4) | 3); // square pixels, 25 fps
    v.extend_from_slice(&[1, 1, 1]); // BT.709 primaries, transfer, matrix
    v.push(alpha);
    v.extend_from_slice(&[0, 0]); // reserved, no matrices
    v.extend(std::iter::repeat_n(0xAA, extra));
    v
}

/// A slice (§5.3.1) whose header has `extra` informative bytes.
fn slice(qindex: u8, y: &[u8], cb: &[u8], cr: &[u8], extra: usize) -> Vec<u8> {
    let mut v = vec![((6 + extra) as u8) << 3, qindex];
    v.extend_from_slice(&(y.len() as u16).to_be_bytes());
    v.extend_from_slice(&(cb.len() as u16).to_be_bytes());
    v.extend(std::iter::repeat_n(0x55, extra));
    v.extend_from_slice(y);
    v.extend_from_slice(cb);
    v.extend_from_slice(cr);
    v
}

/// A picture (§5.2) of the given slices, one slice size for all.
fn picture(log2_slice_mbs: u8, slices: &[Vec<u8>]) -> Vec<u8> {
    let size = 8 + 2 * slices.len() + slices.iter().map(Vec::len).sum::<usize>();
    let mut v = vec![8 << 3];
    v.extend_from_slice(&(size as u32).to_be_bytes());
    v.extend_from_slice(&(slices.len() as u16).to_be_bytes());
    v.push(log2_slice_mbs << 4);
    for s in slices {
        v.extend_from_slice(&(s.len() as u16).to_be_bytes());
    }
    for s in slices {
        v.extend_from_slice(s);
    }
    v
}

/// `frame_size`, `'icpf'`, the header, pictures, `stuffing` zero bytes;
/// then `trailing` bytes that are not part of the frame.
fn frame(header: &[u8], pictures: &[Vec<u8>], stuffing: usize, trailing: usize) -> Vec<u8> {
    let size = 8 + header.len() + pictures.iter().map(Vec::len).sum::<usize>() + stuffing;
    let mut v = (size as u32).to_be_bytes().to_vec();
    v.extend_from_slice(b"icpf");
    v.extend_from_slice(header);
    for p in pictures {
        v.extend_from_slice(p);
    }
    v.extend(std::iter::repeat_n(0, stuffing));
    v.extend(std::iter::repeat_n(0xFF, trailing));
    v
}

/// One 16×16 4:2:2 macroblock, coded by hand.
///
/// Y′: DC coefficients 0, 64, −64, 128:
/// - first_dc_coeff 0: S = 0, EG5 → `100000`
/// - difference +64 (previousDCDiff 3 → EG3): S = 128 → `0000 10001000`
/// - difference −128 (|prev| 64 → EG3; prev ≥ 0, coded as is): S = 255 →
///   `00000 100000111`
/// - difference +192 (|prev| 128 → EG3; prev < 0, so −192 is coded):
///   S = 383 → `00000 110000111`
///
/// then one AC coefficient: +100 at scanned frequency 1 of block 1 —
/// scanned index 4·1 + 1 = 5, one zero (index 4) after the DCs:
/// - run 1 (previousRun 4 → EG0) → `010`
/// - abs_level_minus_1 99 (previousLevelSymbol 1 → combo(1, 0, 1): two 0s,
///   then EG1(97)) → `00 00000 1100011`
/// - sign + → `0`
///
/// Cb: DCs 0, 0 → `100000` `1000`. Cr: DCs −32, −32 → EG5(63) `0 1011111`,
/// then `1000`; and one `zero_byte` after the zero bits.
fn one_macroblock() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let y = bits("100000 000010001000 00000100000111 00000110000111 010 0000000 1100011 0");
    let cb = bits("100000 1000");
    let mut cr = bits("01011111 1000");
    cr.push(0);
    (y, cb, cr)
}

/// §7.4 and §7.5.1 by hand: 10-bit s = round(2v + 512).
fn sample(v: f64) -> u16 {
    (2.0 * v + 512.0).round() as u16
}

#[test]
fn a_hand_coded_macroblock_decodes_to_the_hand_computed_samples() {
    let (y, cb, cr) = one_macroblock();
    let s = slice(1, &y, &cb, &cr, 1);
    let data = frame(
        &frame_header(16, 16, 2, 0, 0, 0, 4),
        &[picture(0, &[s])],
        5,
        7,
    );

    let hdr = FrameHeader::parse(&data).unwrap();
    assert_eq!(hdr.header_size, 24);
    assert_eq!(hdr.frame_size as usize, data.len() - 7);
    assert_eq!(&hdr.encoder_identifier, b"hand");
    assert_eq!(
        (hdr.width, hdr.height, hdr.chroma, hdr.interlace),
        (16, 16, ChromaFormat::Yuv422, Interlace::Progressive)
    );
    assert_eq!(hdr.metadata.aspect_ratio, 1);
    assert_eq!(hdr.metadata.frame_rate(), Some((25, 1)));
    assert_eq!(hdr.alpha, AlphaType::None);

    let f = Decoder::new().decode(&data).unwrap();
    assert_eq!(f.bit_depth, 10);
    let y = f.plane(0);
    // qScale 1, weights 4: F = QF · 4 · 1 ÷ 8 = QF / 2; a DC-only block is
    // the constant F00 / 8.
    for row in 0..16 {
        for col in 0..16 {
            let want = match (col / 8, row / 8) {
                (0, 0) => sample(0.0),
                (1, 0) => {
                    // F00 = 32, and F[0][1] = 50 at (v, u) = (0, 1): the
                    // progressive scan's index 1.
                    let x = (col - 8) as f64;
                    let ac = 50.0 / (4.0 * 2f64.sqrt())
                        * ((2.0 * x + 1.0) * std::f64::consts::PI / 16.0).cos();
                    sample(4.0 + ac)
                }
                (0, 1) => sample(-4.0),
                _ => sample(8.0),
            };
            assert_eq!(y[row * 16 + col], want, "Y′ row {row} col {col}");
        }
    }
    assert!(f.plane(1).iter().all(|&s| s == 512));
    // −32 · 4 ÷ 8 = −16; ÷ 8 = −2.
    assert!(f.plane(2).iter().all(|&s| s == sample(-2.0)));
    assert_eq!((f.planes[1].width, f.planes[1].height), (8, 16));
}

/// Interlaced: two one-row field pictures of a 16×2 frame. Each field is a
/// flat macroblock whose Y′ DCs are all `dc`; the first picture is the top
/// field for `interlace_mode` 1 and the bottom one for 2.
#[test]
fn field_pictures_are_woven_by_interlace_mode() {
    // DCs 64, 64, 64, 64: S = 128, EG5: v = 128 + 32 = 160 = 10100000
    // (8 bits), q = 8 − 5 − 1 = 2: `00 10100000`; then three
    // zero differences, the first with EG3 (`1000`), then EG0 (`1`).
    let y_hi = bits("0010100000 1000 1 1");
    // DCs −64 …: S = 127, v = 159 = 10011111: `00 10011111`, then zeros.
    let y_lo = bits("0010011111 1000 1 1");
    let flat_chroma = bits("100000 1000");
    for (mode, first_row) in [(1u8, 0usize), (2, 1)] {
        let first = picture(0, &[slice(1, &y_hi, &flat_chroma, &flat_chroma, 0)]);
        let second = picture(0, &[slice(1, &y_lo, &flat_chroma, &flat_chroma, 0)]);
        let data = frame(
            &frame_header(16, 2, 2, mode, 0, 0, 0),
            &[first, second],
            0,
            0,
        );
        let f = Decoder::new().decode(&data).unwrap();
        assert_eq!(
            f.interlace,
            if mode == 1 {
                Interlace::TopFieldFirst
            } else {
                Interlace::BottomFieldFirst
            }
        );
        let y = f.plane(0);
        assert!(
            y[first_row * 16..first_row * 16 + 16]
                .iter()
                .all(|&s| s == sample(4.0)),
            "mode {mode}"
        );
        let other = 1 - first_row;
        assert!(
            y[other * 16..other * 16 + 16]
                .iter()
                .all(|&s| s == sample(-4.0)),
            "mode {mode}"
        );
    }
}

/// A slice of alpha by hand (§5.3.3, Tables 12 and 13): a 16×16 4:4:4
/// frame, 8-bit alpha. Row 0 is 255 ×16, the rest 0 except the very last
/// sample, which is 10.
#[test]
fn hand_coded_alpha() {
    // Colour: 4:4:4, four blocks per component, all DCs 0.
    let flat4 = bits("100000 1000 1 1");
    // Alpha: from −1, +256 ≡ 0 mod 256 to reach 255 — an escape: `1` then
    // 0x00; run 16 → `01111`. Then −255 → escape `1` 0x01; run 239 →
    // escape `00000` + 238 in 11 bits. Then +10 → escape (|d| > 8) `1`
    // 0x0A; run 1 → `1`.
    let alpha = bits("1 00000000 01111  1 00000001 00000 00011101110  1 00001010 1");
    let mut s = vec![8 << 3, 1];
    s.extend_from_slice(&(flat4.len() as u16).to_be_bytes());
    s.extend_from_slice(&(flat4.len() as u16).to_be_bytes());
    s.extend_from_slice(&(flat4.len() as u16).to_be_bytes());
    for _ in 0..3 {
        s.extend_from_slice(&flat4);
    }
    s.extend_from_slice(&alpha);
    let data = frame(
        &frame_header(16, 16, 3, 0, 1, 1, 0),
        &[picture(3, &[s])],
        0,
        0,
    );
    let f = Decoder::with_bit_depth(8).unwrap().decode(&data).unwrap();
    let a = f.alpha.unwrap();
    assert!(a[..16].iter().all(|&v| v == 255));
    assert!(a[16..255].iter().all(|&v| v == 0));
    assert_eq!(a[255], 10);
    assert!(f.data.iter().all(|&s| s == 128));
}

#[test]
fn malformed_headers_are_errors() {
    let (y, cb, cr) = one_macroblock();
    let good = frame(
        &frame_header(16, 16, 2, 0, 0, 0, 0),
        &[picture(0, &[slice(1, &y, &cb, &cr, 0)])],
        0,
        0,
    );
    assert!(Decoder::new().decode(&good).is_ok());
    let invalid = |d: &[u8]| matches!(Decoder::new().decode(d), Err(Error::Invalid(_)));

    assert!(invalid(&good[..20]));
    assert!(invalid(&good[..good.len() - 1])); // frame_size past the data
    let mut b = good.clone();
    b[4] = b'x'; // not 'icpf'
    assert!(invalid(&b));
    let mut b = good.clone();
    b[8 + 12] = 1 << 6; // chroma_format 1: reserved
    assert!(invalid(&b));
    let mut b = good.clone();
    b[8 + 12] = (2 << 6) | (3 << 2); // interlace_mode 3: reserved
    assert!(invalid(&b));
    let mut b = good.clone();
    b[8 + 17] = 3; // alpha_channel_type 3: reserved
    assert!(invalid(&b));
    let mut b = good.clone();
    b[8 + 3] = 2; // bitstream_version 2
    assert!(matches!(
        Decoder::new().decode(&b),
        Err(Error::Unsupported(_))
    ));
    let mut b = good.clone();
    b[8 + 20 + 8 + 2 + 1] = 0; // quantization_index 0: reserved
    assert!(invalid(&b));
    let mut b = good.clone();
    b[8 + 20 + 8] = 0xFF; // slice size past picture_size
    assert!(invalid(&b));
    let mut b = good.clone();
    b[8 + 20 + 4] = 0xFF; // picture_size past the frame
    assert!(invalid(&b));

    // A run past the end of the coefficient array: DCs 0, 0, 0, 0, then a
    // run of 300 (EG0: 300 + 1 = 100101101, eight 0s first).
    let y = bits("100000 1000 1 1 00000000 100101101 1 0");
    let b = frame(
        &frame_header(16, 16, 2, 0, 0, 0, 0),
        &[picture(0, &[slice(1, &y, &cb, &cr, 0)])],
        0,
        0,
    );
    assert!(invalid(&b));
    // Coefficient data cut short mid-codeword.
    let y = bits("0000000000");
    let b = frame(
        &frame_header(16, 16, 2, 0, 0, 0, 0),
        &[picture(0, &[slice(1, &y, &cb, &cr, 0)])],
        0,
        0,
    );
    assert!(invalid(&b));
}
