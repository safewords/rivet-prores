//! A ProRes encoder: the inverse of each step of RDD 36 §7, and a rate
//! control that picks a quantiser per slice to land each frame at the
//! profile's target size.
//!
//! Each picture's blocks are transformed once. A binary search then finds
//! the finest quantiser that, applied to every slice, fits the picture's
//! byte budget; the slack it leaves is handed out slice by slice (each
//! slice's share in proportion to its macroblocks), and each slice takes the
//! finest quantiser that fits its share. The frame therefore never exceeds
//! its target unless even the coarsest quantiser (224) cannot reach it, and
//! the quality across a picture stays nearly uniform. Alpha, which is
//! lossless, is coded on top of the target.

use crate::bits::{BitCounter, BitSink, BitWriter};
use crate::decode::block_position;
use crate::dct::fdct;
use crate::error::{Result, config};
use crate::frame::{ChromaFormat, Frame, Interlace};
use crate::header::{AlphaType, FrameHeader, PICTURE_HEADER_SIZE, PictureHeader, slice_sizes};
use crate::tables::{INTERLACED_SCAN, PROGRESSIVE_SCAN, qscale};
use crate::vlc::{self, FIRST_DC_CODEBOOK, signed_to_symbol};

/// The ProRes family. The bitstream does not name its profile — they share
/// one syntax and differ in chroma format, alpha and bit rate — so the
/// profile lives in the container, as the MOV sample entry's four-character
/// code ([`Profile::fourcc`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Profile {
    /// ProRes 422 Proxy (`apco`).
    Proxy,
    /// ProRes 422 LT (`apcs`).
    Lt,
    /// ProRes 422 (`apcn`).
    Standard,
    /// ProRes 422 HQ (`apch`).
    Hq,
    /// ProRes 4444 (`ap4h`).
    P4444,
    /// ProRes 4444 XQ (`ap4x`).
    P4444Xq,
}

impl Profile {
    /// Every profile, from the lowest bit rate to the highest.
    pub const ALL: [Profile; 6] =
        [Profile::Proxy, Profile::Lt, Profile::Standard, Profile::Hq, Profile::P4444, Profile::P4444Xq];

    /// The MOV sample entry code.
    pub fn fourcc(self) -> [u8; 4] {
        *match self {
            Profile::Proxy => b"apco",
            Profile::Lt => b"apcs",
            Profile::Standard => b"apcn",
            Profile::Hq => b"apch",
            Profile::P4444 => b"ap4h",
            Profile::P4444Xq => b"ap4x",
        }
    }

    /// The profile a MOV sample entry code names.
    pub fn from_fourcc(code: [u8; 4]) -> Option<Profile> {
        Profile::ALL.into_iter().find(|p| p.fourcc() == code)
    }

    /// The chroma format the profile codes.
    pub fn chroma(self) -> ChromaFormat {
        match self {
            Profile::P4444 | Profile::P4444Xq => ChromaFormat::Yuv444,
            _ => ChromaFormat::Yuv422,
        }
    }

    /// Target bit rate in bits per second for 1920×1080 at 29.97 frames per
    /// second — the figures Apple publishes for each profile (ProRes white
    /// paper); 4444 and 4444 XQ without alpha.
    pub fn reference_bitrate(self) -> u64 {
        match self {
            Profile::Proxy => 45_000_000,
            Profile::Lt => 102_000_000,
            Profile::Standard => 147_000_000,
            Profile::Hq => 220_000_000,
            Profile::P4444 => 330_000_000,
            Profile::P4444Xq => 500_000_000,
        }
    }

    /// Target size of one frame in bytes: the reference rate's bytes per
    /// frame, scaled by the frame's area. Apple's rates scale with frame
    /// rate, so the size per frame does not depend on it. Alpha is coded
    /// losslessly on top of this.
    pub fn target_frame_bytes(self, width: u32, height: u32) -> usize {
        let per_frame = self.reference_bitrate() as f64 * 1001.0 / 30000.0 / 8.0;
        (per_frame * (width as f64 * height as f64) / (1920.0 * 1080.0)) as usize
    }
}

/// What to encode and how.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// The profile: chroma format and target bit rate.
    pub profile: Profile,
    /// log2 of the macroblocks per slice, 0–3 (default 3: 8 macroblocks,
    /// what every encoder writes for HD).
    pub log2_slice_mbs: u32,
    /// How a frame's alpha plane is coded, 4444 profiles only: 16 bits per
    /// sample (default, exact for any input depth up to 16) or 8. `None`
    /// drops the alpha plane.
    pub alpha: AlphaType,
    /// A luma quantisation matrix to load (raster order, weights 2–63);
    /// `None` uses the default (every weight 4).
    pub luma_matrix: Option<[u8; 64]>,
    /// A chroma quantisation matrix to load; `None` uses the luma weights.
    pub chroma_matrix: Option<[u8; 64]>,
    /// Overrides the profile's target frame size, in bytes.
    pub target_frame_bytes: Option<usize>,
    /// The `encoder_identifier` written in the frame header.
    pub encoder_identifier: [u8; 4],
}

impl Config {
    /// The defaults for a profile.
    pub fn new(profile: Profile) -> Config {
        Config {
            profile,
            log2_slice_mbs: 3,
            alpha: AlphaType::Bits16,
            luma_matrix: None,
            chroma_matrix: None,
            target_frame_bytes: None,
            encoder_identifier: *b"rivt",
        }
    }

    fn validate(&self) -> Result<()> {
        if self.log2_slice_mbs > 3 {
            return Err(config(format!("log2_slice_mbs {} is outside 0–3", self.log2_slice_mbs)));
        }
        for m in [&self.luma_matrix, &self.chroma_matrix].into_iter().flatten() {
            if m.iter().any(|w| !(2..=63).contains(w)) {
                return Err(config("quantisation weights must be 2–63"));
            }
        }
        Ok(())
    }
}

/// A ProRes encoder. Stateless between frames, like the decoder.
#[derive(Debug, Clone)]
pub struct Encoder {
    config: Config,
}

/// Rounding offsets for quantisation: nearest for DC, a small dead zone for
/// AC (it costs little distortion and saves bits at every quantiser).
const DC_ROUNDING: f32 = 0.5;
const AC_ROUNDING: f32 = 0.42;

impl Encoder {
    /// An encoder for `config`. The configuration is checked when a frame
    /// is encoded.
    pub fn new(config: Config) -> Encoder {
        Encoder { config }
    }

    /// The configuration.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Encodes one frame into one ProRes frame: `frame_size`, `'icpf'`, the
    /// header and the pictures — the bytes of one MOV sample. The frame's
    /// chroma format must be the profile's; any depth from 8 to 16 bits is
    /// taken. An interlaced frame (by [`Frame::interlace`]) is coded as two
    /// field pictures.
    pub fn encode(&self, frame: &Frame) -> Result<Vec<u8>> {
        let cfg = &self.config;
        cfg.validate()?;
        frame.validate()?;
        if frame.chroma != cfg.profile.chroma() {
            return Err(config(format!(
                "{:?} codes {:?}, and the frame is {:?}",
                cfg.profile,
                cfg.profile.chroma(),
                frame.chroma
            )));
        }
        let alpha = match (&frame.alpha, cfg.profile.chroma()) {
            (None, _) => AlphaType::None,
            (Some(_), ChromaFormat::Yuv422) => {
                return Err(config(format!("{:?} carries no alpha channel", cfg.profile)));
            }
            (Some(_), ChromaFormat::Yuv444) => cfg.alpha,
        };
        let hdr = FrameHeader {
            frame_size: 0,
            header_size: 0,
            bitstream_version: if frame.chroma == ChromaFormat::Yuv422 && alpha == AlphaType::None { 0 } else { 1 },
            encoder_identifier: cfg.encoder_identifier,
            width: frame.width as u16,
            height: frame.height as u16,
            chroma: frame.chroma,
            interlace: frame.interlace,
            metadata: frame.metadata,
            alpha,
            luma_matrix: cfg.luma_matrix,
            chroma_matrix: cfg.chroma_matrix,
        };
        let mut out = Vec::new();
        hdr.write(&mut out);
        let target = cfg.target_frame_bytes.unwrap_or_else(|| cfg.profile.target_frame_bytes(frame.width, frame.height));
        let pictures = hdr.picture_count();
        let per_picture = target.saturating_sub(out.len()) / pictures;
        for index in 0..pictures {
            encode_picture(&hdr, index, frame, cfg.log2_slice_mbs, per_picture, &mut out);
        }
        let size = out.len() as u32;
        out[0..4].copy_from_slice(&size.to_be_bytes());
        Ok(out)
    }
}

/// One slice's transformed blocks, per component, in slice block order;
/// and its alpha, already coded (alpha is lossless, so its size is fixed).
struct SliceBlocks {
    mbs: u32,
    comps: [Vec<[f32; 64]>; 3],
    alpha: Vec<u8>,
}

/// The coded bytes of one slice for one component at one quantiser.
fn component_bytes(blocks: &[[f32; 64]], weights: &[u8; 64], q: u32, scan: &[u8; 64], scratch: &mut Vec<i32>) -> usize {
    quantise(blocks, weights, q, scan, scratch);
    let mut c = BitCounter::default();
    encode_coefficients(&mut c, scratch, blocks.len());
    c.0.div_ceil(8)
}

fn quantise(blocks: &[[f32; 64]], weights: &[u8; 64], q: u32, scan: &[u8; 64], out: &mut Vec<i32>) {
    let n = blocks.len();
    out.clear();
    out.resize(n * 64, 0);
    let mut inv = [0f32; 64];
    for k in 0..64 {
        // QF = F · 8 ÷ (W · qScale), the inverse of §7.3.
        inv[k] = 8.0 / (weights[k] as u32 * q) as f32;
    }
    for (blk, f) in blocks.iter().enumerate() {
        for k in 0..64 {
            let x = f[k] * inv[k];
            let r = if k == 0 { DC_ROUNDING } else { AC_ROUNDING };
            let m = (x.abs() + r).floor() as i32;
            out[n * scan[k] as usize + blk] = if x < 0.0 { -m } else { m };
        }
    }
}

/// `scanned_coefficients()` (§5.3.2) from a scanned array, byte-aligned.
pub(crate) fn encode_coefficients<S: BitSink>(s: &mut S, scanned: &[i32], n_blocks: usize) {
    let first = scanned[0];
    FIRST_DC_CODEBOOK.put(s, signed_to_symbol(first));
    let mut prev_dc = first;
    let mut prev_diff = 3i32;
    for &dc in &scanned[1..n_blocks] {
        let diff = dc - prev_dc;
        let n = if prev_diff < 0 { -diff } else { diff };
        vlc::dc_codebook(prev_diff).put(s, signed_to_symbol(n));
        prev_diff = diff;
        prev_dc = dc;
    }
    let mut prev_run = 4u32;
    let mut prev_level = 1u32;
    let mut run = 0u32;
    for &c in &scanned[n_blocks..] {
        if c == 0 {
            run += 1;
            continue;
        }
        vlc::run_codebook(prev_run).put(s, run);
        prev_run = run;
        run = 0;
        let level_symbol = c.unsigned_abs() - 1;
        vlc::level_codebook(prev_level).put(s, level_symbol);
        prev_level = level_symbol;
        s.put((c < 0) as u64, 1);
    }
    s.align();
}

/// `scanned_alpha()` (§5.3.3) for raster-scanned values of `bits` bits.
pub(crate) fn encode_alpha<S: BitSink>(s: &mut S, values: &[u16], bits: u32) {
    let mut prev: i32 = -1;
    let mut i = 0;
    while i < values.len() {
        let a = values[i] as i32;
        let mut run = 1;
        while run < 2048 && i + run < values.len() && values[i + run] as i32 == a {
            run += 1;
        }
        vlc::put_alpha_difference(s, a - prev, bits);
        vlc::put_alpha_run(s, run as u32);
        prev = a;
        i += run;
    }
    s.align();
}

/// The picture's samples of one component, padded to whole macroblocks by
/// repeating the last column and row, as reconstructed values
/// `v = 512 · s ÷ 2^b − 256` (the inverse of §7.5.1).
fn padded_component(
    frame: &Frame,
    plane: usize,
    pic_w: usize,
    pic_h: usize,
    rows: (usize, usize, usize),
) -> Vec<f32> {
    let (first_row, step, height) = rows;
    let p = frame.planes[plane];
    let src = frame.plane(plane);
    let scale = 512.0 / (1u32 << frame.bit_depth) as f32;
    let (w, stride) = (p.width as usize, p.width as usize);
    let mut out = vec![0f32; pic_w * pic_h];
    for y in 0..pic_h {
        let row = first_row + step * y.min(height - 1);
        let line = &src[row * stride..row * stride + w];
        for x in 0..pic_w {
            out[y * pic_w + x] = line[x.min(w - 1)] as f32 * scale - 256.0;
        }
    }
    out
}

fn encode_picture(hdr: &FrameHeader, index: usize, frame: &Frame, log2_slice_mbs: u32, budget: usize, out: &mut Vec<u8>) {
    let width_in_mb = frame.width.div_ceil(16);
    let height = hdr.picture_height(index) as usize;
    let height_in_mb = height.div_ceil(16);
    let (first_row, step) = hdr.picture_rows(index);
    let sizes = slice_sizes(width_in_mb, log2_slice_mbs);
    let scan = if hdr.interlace == Interlace::Progressive { &PROGRESSIVE_SCAN } else { &INTERLACED_SCAN };
    let (chroma_w, chroma_per_mb) = match frame.chroma {
        ChromaFormat::Yuv422 => (8, 2),
        ChromaFormat::Yuv444 => (16, 4),
    };
    let pic_h = 16 * height_in_mb;
    let rows = (first_row, step, height);
    let comps = [
        padded_component(frame, 0, 16 * width_in_mb as usize, pic_h, rows),
        padded_component(frame, 1, chroma_w * width_in_mb as usize, pic_h, rows),
        padded_component(frame, 2, chroma_w * width_in_mb as usize, pic_h, rows),
    ];
    let comp_w = [16 * width_in_mb as usize, chroma_w * width_in_mb as usize, chroma_w * width_in_mb as usize];

    // Transform every block once.
    let mut slices = Vec::with_capacity(sizes.len() * height_in_mb);
    for mb_y in 0..height_in_mb {
        let mut mb_x = 0u32;
        for &mbs in &sizes {
            let mut blocks: [Vec<[f32; 64]>; 3] = Default::default();
            for plane in 0..3 {
                let per_mb = if plane == 0 { 4 } else { chroma_per_mb };
                for blk in 0..per_mb * mbs as usize {
                    let (bx, by) = block_position(plane, frame.chroma, mb_x, mb_y, blk / per_mb, blk % per_mb);
                    let mut pix = [0f32; 64];
                    for y in 0..8 {
                        let src = &comps[plane][(by + y) * comp_w[plane] + bx..][..8];
                        pix[8 * y..8 * y + 8].copy_from_slice(src);
                    }
                    blocks[plane].push(fdct(&pix));
                }
            }
            let alpha = match (&frame.alpha, hdr.alpha) {
                (Some(a), t) if t != AlphaType::None => {
                    let rows = if mb_y + 1 < height_in_mb { 16 } else { height - 16 * (height_in_mb - 1) };
                    slice_alpha(frame, a, t.bits(), mb_x, mbs, mb_y, rows, first_row, step)
                }
                _ => Vec::new(),
            };
            slices.push(SliceBlocks { mbs, comps: blocks, alpha });
            mb_x += mbs;
        }
    }

    let header_bytes = if hdr.alpha != AlphaType::None { 8 } else { 6 };
    let luma_w = hdr.luma_weights();
    let chroma_w = hdr.chroma_weights();
    let mut scratch = Vec::new();
    let slice_size = |s: &SliceBlocks, qi: u8, scratch: &mut Vec<i32>| -> (usize, bool) {
        let q = qscale(qi);
        let y = component_bytes(&s.comps[0], &luma_w, q, scan, scratch);
        let cb = component_bytes(&s.comps[1], &chroma_w, q, scan, scratch);
        let cr = component_bytes(&s.comps[2], &chroma_w, q, scan, scratch);
        let total = header_bytes + y + cb + cr + s.alpha.len();
        // Every size field is 16 bits.
        (total, total <= 65535 && y <= 65535 && cb <= 65535 && cr <= 65535)
    };

    // The finest uniform quantiser that fits.
    let overhead = PICTURE_HEADER_SIZE + 2 * slices.len();
    // Alpha is lossless and comes on top of the profile's rate.
    let alpha_bytes: usize = slices.iter().map(|s| s.alpha.len()).sum();
    let budget = (budget + alpha_bytes).saturating_sub(overhead);
    let totals = |qi: u8, scratch: &mut Vec<i32>| -> (Vec<usize>, bool) {
        let mut ok = true;
        let v = slices
            .iter()
            .map(|s| {
                let (n, fits) = slice_size(s, qi, scratch);
                ok &= fits;
                n
            })
            .collect::<Vec<_>>();
        let sum: usize = v.iter().sum();
        (v, ok && sum <= budget)
    };
    let (mut lo, mut hi) = (1u8, 224u8);
    let mut best = totals(224, &mut scratch);
    if best.1 {
        // Invariant: `hi` fits, with `best` its sizes; find the smallest that does.
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let t = totals(mid, &mut scratch);
            if t.1 {
                hi = mid;
                best = t;
            } else {
                lo = mid + 1;
            }
        }
    }
    let uniform = hi;
    let base_sizes = best.0;

    // Per-slice refinement: hand out the slack.
    let mut qindices = vec![uniform; slices.len()];
    if uniform > 1 && base_sizes.iter().sum::<usize>() <= budget {
        let mut remaining_budget = budget;
        let mut remaining_base: usize = base_sizes.iter().sum();
        let mut remaining_mbs: u64 = slices.iter().map(|s| s.mbs as u64).sum();
        for (i, s) in slices.iter().enumerate() {
            let slack = (remaining_budget - remaining_base) as u64;
            let allowed = base_sizes[i] + (slack * s.mbs as u64 / remaining_mbs) as usize;
            let (mut lo, mut hi) = (1u8, uniform);
            let mut used = base_sizes[i];
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                let (n, fits) = slice_size(s, mid, &mut scratch);
                if fits && n <= allowed {
                    hi = mid;
                    used = n;
                } else {
                    lo = mid + 1;
                }
            }
            qindices[i] = hi;
            remaining_budget -= used;
            remaining_base -= base_sizes[i];
            remaining_mbs -= s.mbs as u64;
        }
    }

    // Write the picture.
    let pic_start = out.len();
    PictureHeader::write(out, 0, slices.len(), log2_slice_mbs);
    let table = out.len();
    out.resize(table + 2 * slices.len(), 0);
    for (k, (s, &qi)) in slices.iter().zip(&qindices).enumerate() {
        let start = out.len();
        out.push((header_bytes as u8) << 3);
        out.push(qi);
        out.resize(start + header_bytes, 0);
        let q = qscale(qi);
        let mut comp_sizes = [0usize; 3];
        for (plane, size) in comp_sizes.iter_mut().enumerate() {
            let w = if plane == 0 { &luma_w } else { &chroma_w };
            quantise(&s.comps[plane], w, q, scan, &mut scratch);
            let mut bw = BitWriter::new(out);
            encode_coefficients(&mut bw, &scratch, s.comps[plane].len());
            *size = bw.finish();
        }
        out[start + 2..start + 4].copy_from_slice(&(comp_sizes[0] as u16).to_be_bytes());
        out[start + 4..start + 6].copy_from_slice(&(comp_sizes[1] as u16).to_be_bytes());
        if header_bytes == 8 {
            out[start + 6..start + 8].copy_from_slice(&(comp_sizes[2] as u16).to_be_bytes());
        }
        out.extend_from_slice(&s.alpha);
        let size = (out.len() - start) as u16;
        out[table + 2 * k..table + 2 * k + 2].copy_from_slice(&size.to_be_bytes());
    }
    let picture_size = (out.len() - pic_start) as u32;
    out[pic_start + 1..pic_start + 5].copy_from_slice(&picture_size.to_be_bytes());
}

/// One slice's alpha, converted to `bits` bits and coded. Columns past the
/// frame's right edge repeat the last one (§7.5.3: they are coded and
/// discarded); the rows are the slice's rows within the picture.
#[allow(clippy::too_many_arguments)]
fn slice_alpha(
    frame: &Frame,
    alpha: &[u16],
    bits: u32,
    mb_x: u32,
    mbs: u32,
    mb_y: usize,
    rows: usize,
    first_row: usize,
    step: usize,
) -> Vec<u8> {
    let width = frame.width as usize;
    let max_in = (1u64 << frame.bit_depth) - 1;
    let max_out = (1u64 << bits) - 1;
    let cols = 16 * mbs as usize;
    let x0 = 16 * mb_x as usize;
    let mut values = Vec::with_capacity(cols * rows);
    for r in 0..rows {
        let row = first_row + step * (16 * mb_y + r);
        for n in 0..cols {
            let a = (alpha[row * width + (x0 + n).min(width - 1)] as u64).min(max_in);
            // round(max_out · a ÷ max_in)
            values.push(((2 * max_out * a + max_in) / (2 * max_in)) as u16);
        }
    }
    let mut out = Vec::new();
    let mut w = BitWriter::new(&mut out);
    encode_alpha(&mut w, &values, bits);
    w.finish();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{decode_alpha, decode_coefficients};

    #[test]
    fn coefficients_round_trip_through_the_entropy_coder() {
        let mut seed = 0x1234_5678u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for n_blocks in [1usize, 2, 4, 8, 16, 32] {
            for density in [0u32, 1, 8, 50, 100] {
                let mut c = vec![0i32; n_blocks * 64];
                for (i, v) in c.iter_mut().enumerate() {
                    if i < n_blocks || rnd() % 100 < density {
                        let mag = (rnd() % 3000) as i32 >> (rnd() % 10);
                        *v = if rnd() & 1 == 1 { -mag } else { mag };
                        if i >= n_blocks && *v == 0 {
                            *v = 1;
                        }
                    }
                }
                let mut bytes = Vec::new();
                let mut w = BitWriter::new(&mut bytes);
                encode_coefficients(&mut w, &c, n_blocks);
                w.finish();
                let mut got = vec![0i32; n_blocks * 64];
                decode_coefficients(&bytes, n_blocks, &mut got).unwrap();
                assert_eq!(got, c);
            }
        }
    }

    #[test]
    fn alpha_round_trips_through_its_coder() {
        for bits in [8u32, 16] {
            let max = (1u32 << bits) - 1;
            let mut v: Vec<u16> = Vec::new();
            v.extend(std::iter::repeat_n(max as u16, 3000)); // a run past 2048
            v.extend([0, 1, 2, 200, (max / 2) as u16, 3, 3, 3, max as u16, 0]);
            v.extend((0..500).map(|i| ((i * 37) % (max as usize + 1)) as u16));
            let mut bytes = Vec::new();
            let mut w = BitWriter::new(&mut bytes);
            encode_alpha(&mut w, &v, bits);
            w.finish();
            assert_eq!(decode_alpha(&bytes, bits, v.len()).unwrap(), v);
        }
    }

    #[test]
    fn profiles_and_fourccs() {
        for p in Profile::ALL {
            assert_eq!(Profile::from_fourcc(p.fourcc()), Some(p));
        }
        // 220 Mb/s at 29.97 fps is 917 583 bytes per 1080 frame.
        assert_eq!(Profile::Hq.target_frame_bytes(1920, 1080), 917_583);
    }
}
