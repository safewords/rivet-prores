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

use crate::bits::{BitSink, BitWriter};
use crate::decode::{Rescale, block_position};
use crate::dsp::{self, Isa};
use crate::error::{Result, config};
use crate::frame::{ChromaFormat, Frame, Interlace};
use crate::header::{AlphaType, FrameHeader, PICTURE_HEADER_SIZE, PictureHeader, slice_sizes};
use crate::pool;
use crate::tables::{INTERLACED_SCAN, PROGRESSIVE_SCAN, qscale};
use crate::vlc::{self, Codebook, FIRST_DC_CODEBOOK, signed_to_symbol};
use std::cell::RefCell;
use std::sync::Mutex;

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

    /// Target size of one frame in bytes, for the default configuration
    /// (progressive, 8-macroblock slices, no loaded matrices): the
    /// reference rate's bytes per macroblock times the frame's macroblocks,
    /// plus the frame's headers — the frame header, the picture header, and
    /// each slice's table entry and header.
    ///
    /// The reference rate's bytes per 1920×1080 frame less that frame's
    /// headers, over its 8160 macroblocks (1920×1088 coded), is the
    /// profile's coded data per macroblock; a frame of any size gets as
    /// much per macroblock (partial macroblocks at the right and bottom
    /// edges are coded whole), with its own headers on top. 1920×1080
    /// comes out at the reference rate. Scaling by area instead would let
    /// the headers, fixed per frame and per slice, eat a small frame's
    /// budget: a single 16×16 macroblock would keep a third of its share.
    /// Apple's rates scale with frame rate, so the size per frame does not
    /// depend on it. Alpha is coded losslessly on top of this.
    pub fn target_frame_bytes(self, width: u32, height: u32) -> usize {
        let per_frame = self.reference_bitrate() as f64 * 1001.0 / 30000.0 / 8.0;
        let per_mb = (per_frame - default_headers(1920, 1080) as f64) / macroblocks(1920, 1080) as f64;
        (per_mb * macroblocks(width, height) as f64) as usize + default_headers(width, height)
    }
}

/// Macroblocks of a progressive `width` × `height` frame.
fn macroblocks(width: u32, height: u32) -> u64 {
    width.div_ceil(16) as u64 * height.div_ceil(16) as u64
}

/// Header bytes of a progressive frame in the default configuration: the
/// frame header without matrices (8 + 20), one picture header, and per
/// slice its 2-byte table entry and 6-byte slice header.
fn default_headers(width: u32, height: u32) -> usize {
    let slices = slice_sizes(width.div_ceil(16), 3).len() * height.div_ceil(16) as usize;
    8 + crate::header::FRAME_HEADER_FIXED + PICTURE_HEADER_SIZE + (2 + 6) * slices
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
    /// Threads to encode on: the calling one and workers shared by every
    /// encoder and decoder in the process. 0 (the default) means one per
    /// CPU; 1 keeps all the work on the calling thread. The output is the
    /// same at any thread count.
    pub threads: usize,
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
            threads: 0,
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

/// Rounding offset for quantising AC coefficients: a small dead zone (it
/// costs little distortion and saves bits at every quantiser). The DC is
/// rounded to nearest (0.5, in [`dsp::quantise`]).
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
            encode_picture(&hdr, index, frame, cfg.log2_slice_mbs, per_picture, cfg.threads, &mut out);
        }
        let size = out.len() as u32;
        out[0..4].copy_from_slice(&size.to_be_bytes());
        Ok(out)
    }
}

/// One slice's transformed blocks, per component, in scanned order
/// (`comps[c][n · s + b]` is position `s` of the scan of block `b`, `n`
/// blocks — the order the entropy coder walks); and its alpha, already
/// coded (alpha is lossless, so its size is fixed).
struct SliceBlocks<'a> {
    mbs: u32,
    comps: [&'a [f32]; 3],
    alpha: Vec<u8>,
}

/// `8 ÷ (W · qScale)` at every scanned position, for luma and chroma, for
/// every quantization_index (index 0 unused): QF = F · 8 ÷ (W · qScale),
/// the inverse of §7.3.
struct Quantisers(Vec<[[f32; 64]; 2]>);

impl Quantisers {
    fn new(luma: &[u8; 64], chroma: &[u8; 64], scan: &[u8; 64]) -> Quantisers {
        let table = |w: &[u8; 64], q: u32| {
            let mut inv = [0f32; 64];
            for k in 0..64 {
                inv[scan[k] as usize] = 8.0 / (w[k] as u32 * q) as f32;
            }
            inv
        };
        Quantisers((0..=224u8).map(|qi| [table(luma, qscale(qi.max(1))), table(chroma, qscale(qi.max(1)))]).collect())
    }

    fn get(&self, qi: u8, plane: usize) -> &[f32; 64] {
        &self.0[qi as usize][(plane > 0) as usize]
    }
}

/// A slice component quantised: the scanned array and its non-zero bitmap.
#[derive(Default)]
pub(crate) struct Quantised {
    pub(crate) values: Vec<i32>,
    pub(crate) mask: Vec<u64>,
}

impl Quantised {
    pub(crate) fn quantise(&mut self, isa: Isa, f: &[f32], inv: &[f32; 64]) {
        let n = f.len() / 64;
        self.values.resize(64 * n, 0);
        self.mask.resize(n, 0);
        dsp::quantise(isa, f, n, inv, AC_ROUNDING, &mut self.values, &mut self.mask);
    }
}

thread_local! {
    /// Each thread's quantisation buffer, reused from slice to slice.
    static SCRATCH: RefCell<Quantised> = RefCell::default();
    /// The transformed picture, kept by the encoding thread from frame to
    /// frame: a picture's worth of fresh memory each frame costs more in
    /// page faults than transforming it does.
    static COEFFICIENTS: RefCell<Vec<f32>> = RefCell::default();
}

/// Calls `f(i)` for every set bit `i ≥ from` of `mask`, in order.
#[inline(always)]
fn for_each_set_bit(mask: &[u64], from: usize, mut f: impl FnMut(usize)) {
    for (w, &word) in mask.iter().enumerate().skip(from / 64) {
        let mut bits = if w == from / 64 { word & (!0u64 << (from % 64)) } else { word };
        while bits != 0 {
            f(64 * w + bits.trailing_zeros() as usize);
            bits &= bits - 1;
        }
    }
}

/// The DC part of `scanned_coefficients()` (§7.1.1.3), as (codebook,
/// symbol) pairs.
#[inline(always)]
fn dc_symbols(scanned: &[i32], n_blocks: usize, mut f: impl FnMut(Codebook, u32)) {
    let first = scanned[0];
    f(FIRST_DC_CODEBOOK, signed_to_symbol(first));
    let mut prev_dc = first;
    let mut prev_diff = 3i32;
    for &dc in &scanned[1..n_blocks] {
        let diff = dc - prev_dc;
        let n = if prev_diff < 0 { -diff } else { diff };
        f(vlc::dc_codebook(prev_diff), signed_to_symbol(n));
        prev_diff = diff;
        prev_dc = dc;
    }
}

/// Bits [`encode_coefficients_masked`] writes, before byte alignment.
pub(crate) fn coefficient_bits(q: &Quantised, n_blocks: usize) -> usize {
    let mut bits = 0u32;
    dc_symbols(&q.values, n_blocks, |cb, n| bits += cb.len(n));
    let (mut prev_run, mut prev_level, mut pos) = (4u32, 1u32, n_blocks);
    for_each_set_bit(&q.mask, n_blocks, |i| {
        let run = (i - pos) as u32;
        let level = q.values[i].unsigned_abs() - 1;
        bits += vlc::run_len(prev_run, run) + vlc::level_len(prev_level, level) + 1;
        (prev_run, prev_level, pos) = (run, level, i + 1);
    });
    bits as usize
}

/// `scanned_coefficients()` (§5.3.2) from a quantised component,
/// byte-aligned.
pub(crate) fn encode_coefficients_masked<S: BitSink>(s: &mut S, q: &Quantised, n_blocks: usize) {
    dc_symbols(&q.values, n_blocks, |cb, n| cb.put(s, n));
    let (mut prev_run, mut prev_level, mut pos) = (4u32, 1u32, n_blocks);
    for_each_set_bit(&q.mask, n_blocks, |i| {
        let run = (i - pos) as u32;
        let c = q.values[i];
        let level = c.unsigned_abs() - 1;
        // Run, level and sign in one write when they fit.
        match (vlc::run_code(prev_run, run), vlc::level_code(prev_level, level)) {
            (Some((r, rl)), Some((l, ll))) if rl + ll < 57 => {
                s.put((((r << ll) | l) << 1) | (c < 0) as u64, rl + ll + 1);
            }
            _ => {
                vlc::run_codebook(prev_run).put(s, run);
                vlc::level_codebook(prev_level).put(s, level);
                s.put((c < 0) as u64, 1);
            }
        }
        (prev_run, prev_level, pos) = (run, level, i + 1);
    });
    s.align();
}

/// `scanned_coefficients()` (§5.3.2) from a scanned array, byte-aligned.
#[cfg(test)]
pub(crate) fn encode_coefficients<S: BitSink>(s: &mut S, scanned: &[i32], n_blocks: usize) {
    let mut mask = vec![0u64; scanned.len().div_ceil(64)];
    for (i, &c) in scanned.iter().enumerate() {
        mask[i / 64] |= ((c != 0) as u64) << (i % 64);
    }
    encode_coefficients_masked(s, &Quantised { values: scanned.to_vec(), mask }, n_blocks);
}

/// `scanned_alpha()` (§5.3.3) for raster-scanned values of `bits` bits.
pub(crate) fn encode_alpha<S: BitSink>(s: &mut S, values: &[u16], bits: u32) {
    let mut prev: i32 = -1;
    let mut i = 0;
    while i < values.len() {
        let a = values[i];
        let end = (i + 2048).min(values.len());
        let run = 1 + values[i + 1..end].iter().take_while(|&&v| v == a).count();
        vlc::put_alpha_difference(s, a as i32 - prev, bits);
        vlc::put_alpha_run(s, run as u32);
        prev = a as i32;
        i += run;
    }
    s.align();
}

/// Where a picture's slices are and what they share.
struct PictureLayout<'a> {
    frame: &'a Frame,
    hdr: &'a FrameHeader,
    /// Picture rows, and the frame row of picture row 0 and the step.
    height: usize,
    first_row: usize,
    step: usize,
    scan: &'static [u8; 64],
    isa: Isa,
}

impl PictureLayout<'_> {
    /// The 8×8 samples of a block at (`bx`, `by`) in picture coordinates,
    /// the picture padded to whole macroblocks by repeating its last column
    /// and row.
    fn block(&self, plane: usize, bx: usize, by: usize) -> [u16; 64] {
        let p = self.frame.planes[plane];
        let src = self.frame.plane(plane);
        let w = p.width as usize;
        let mut pix = [0u16; 64];
        for (y, out) in pix.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            let row = self.first_row + self.step * (by + y).min(self.height - 1);
            let line = &src[row * w..row * w + w];
            if bx + 8 <= w {
                out.copy_from_slice(&line[bx..bx + 8]);
            } else {
                for (x, o) in out.iter_mut().enumerate() {
                    *o = line[(bx + x).min(w - 1)];
                }
            }
        }
        pix
    }

    /// Coefficients of a slice of `mbs` macroblocks, all components.
    fn slice_len(&self, mbs: u32) -> usize {
        let chroma_per_mb = match self.frame.chroma {
            ChromaFormat::Yuv422 => 2,
            ChromaFormat::Yuv444 => 4,
        };
        64 * (4 + 2 * chroma_per_mb) * mbs as usize
    }

    /// Transforms one slice into `out` (component after component, each in
    /// scanned order) and codes its alpha.
    fn transform(&self, mb_x: u32, mbs: u32, mb_y: usize, out: &mut [f32]) -> Vec<u8> {
        let frame = self.frame;
        let chroma_per_mb = match frame.chroma {
            ChromaFormat::Yuv422 => 2,
            ChromaFormat::Yuv444 => 4,
        };
        // v = 512 · s ÷ 2^b − 256, the inverse of §7.5.1.
        let scale = 512.0 / (1u32 << frame.bit_depth) as f32;
        let mut rest = out;
        for plane in 0..3 {
            let per_mb = if plane == 0 { 4 } else { chroma_per_mb };
            let n = per_mb * mbs as usize;
            let (out, r) = rest.split_at_mut(64 * n);
            rest = r;
            for blk in 0..n {
                let (bx, by) = block_position(plane, frame.chroma, mb_x, mb_y, blk / per_mb, blk % per_mb);
                let f = dsp::fdct_load(self.isa, &self.block(plane, bx, by), scale);
                for (k, &v) in f.iter().enumerate() {
                    out[n * self.scan[k] as usize + blk] = v;
                }
            }
        }
        match (&frame.alpha, self.hdr.alpha) {
            (Some(a), t) if t != AlphaType::None => {
                slice_alpha(frame, a, t.bits(), mb_x, mbs, mb_y, self.height, self.first_row, self.step)
            }
            _ => Vec::new(),
        }
    }

    /// A transformed slice's components, as [`Self::transform`] laid them out.
    fn components<'a>(&self, mbs: u32, coefficients: &'a [f32]) -> [&'a [f32]; 3] {
        let luma = 64 * 4 * mbs as usize;
        let chroma = (coefficients.len() - luma) / 2;
        let (y, c) = coefficients.split_at(luma);
        let (cb, cr) = c.split_at(chroma);
        [y, cb, cr]
    }
}

fn encode_picture(
    hdr: &FrameHeader,
    index: usize,
    frame: &Frame,
    log2_slice_mbs: u32,
    budget: usize,
    threads: usize,
    out: &mut Vec<u8>,
) {
    let width_in_mb = frame.width.div_ceil(16);
    let height = hdr.picture_height(index) as usize;
    let height_in_mb = height.div_ceil(16);
    let (first_row, step) = hdr.picture_rows(index);
    let sizes = slice_sizes(width_in_mb, log2_slice_mbs);
    let scan = if hdr.interlace == Interlace::Progressive { &PROGRESSIVE_SCAN } else { &INTERLACED_SCAN };
    let isa = Isa::get();
    let layout = PictureLayout { frame, hdr, height, first_row, step, scan, isa };

    // Transform every block once, a slice per task.
    let mut places = Vec::with_capacity(sizes.len() * height_in_mb);
    for mb_y in 0..height_in_mb {
        let mut mb_x = 0u32;
        for &mbs in &sizes {
            places.push((mb_x, mbs, mb_y));
            mb_x += mbs;
        }
    }
    let lens: Vec<usize> = places.iter().map(|&(_, mbs, _)| layout.slice_len(mbs)).collect();
    let mut coefficients = COEFFICIENTS.with_borrow_mut(std::mem::take);
    coefficients.resize(lens.iter().sum(), 0.0);
    let alphas = {
        let mut chunks = Vec::with_capacity(places.len());
        let mut rest = &mut coefficients[..];
        for &len in &lens {
            let (chunk, r) = rest.split_at_mut(len);
            chunks.push(Mutex::new(Some(chunk)));
            rest = r;
        }
        pool::map(threads, places.len(), |i| {
            let (mb_x, mbs, mb_y) = places[i];
            let chunk = chunks[i].lock().unwrap_or_else(|e| e.into_inner()).take().expect("each slice once");
            layout.transform(mb_x, mbs, mb_y, chunk)
        })
    };
    let mut slices = Vec::with_capacity(places.len());
    let mut rest = &coefficients[..];
    for ((&(_, mbs, _), &len), alpha) in places.iter().zip(&lens).zip(alphas) {
        let (chunk, r) = rest.split_at(len);
        rest = r;
        slices.push(SliceBlocks { mbs, comps: layout.components(mbs, chunk), alpha });
    }
    code_picture(hdr, &slices, scan, isa, log2_slice_mbs, budget, threads, out);
    drop(slices);
    COEFFICIENTS.with_borrow_mut(|c| *c = coefficients);
}

/// Rate control and coding of a transformed picture.
#[allow(clippy::too_many_arguments)]
fn code_picture(
    hdr: &FrameHeader,
    slices: &[SliceBlocks],
    scan: &[u8; 64],
    isa: Isa,
    log2_slice_mbs: u32,
    budget: usize,
    threads: usize,
    out: &mut Vec<u8>,
) {
    let header_bytes = if hdr.alpha != AlphaType::None { 8 } else { 6 };
    let quantisers = Quantisers::new(&hdr.luma_weights(), &hdr.chroma_weights(), scan);
    // A slice's coded size at a quantiser, and whether every size field
    // (16 bits each) can hold it.
    let slice_size = |s: &SliceBlocks, qi: u8| -> (usize, bool) {
        SCRATCH.with_borrow_mut(|q| {
            let mut comp = [0usize; 3];
            for (plane, bytes) in comp.iter_mut().enumerate() {
                q.quantise(isa, s.comps[plane], quantisers.get(qi, plane));
                *bytes = coefficient_bits(q, s.comps[plane].len() / 64).div_ceil(8);
            }
            let total = header_bytes + comp.iter().sum::<usize>() + s.alpha.len();
            (total, total <= 65535 && comp.iter().all(|&c| c <= 65535))
        })
    };
    // Every slice's size at `qi`, in parallel.
    let sizes_at = |qi: u8| pool::map(threads, slices.len(), |i| slice_size(&slices[i], qi));

    // The finest uniform quantiser that fits.
    let overhead = PICTURE_HEADER_SIZE + 2 * slices.len();
    // Alpha is lossless and comes on top of the profile's rate.
    let alpha_bytes: usize = slices.iter().map(|s| s.alpha.len()).sum();
    let budget = (budget + alpha_bytes).saturating_sub(overhead);
    let totals = |qi: u8| -> (Vec<usize>, bool) {
        let v = sizes_at(qi);
        let ok = v.iter().all(|&(_, fits)| fits) && v.iter().map(|&(n, _)| n).sum::<usize>() <= budget;
        (v.into_iter().map(|(n, _)| n).collect(), ok)
    };
    let (mut lo, mut hi) = (1u8, 224u8);
    let mut best = totals(224);
    if best.1 {
        // Invariant: `hi` fits, with `best` its sizes; find the smallest that does.
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let t = totals(mid);
            if t.1 {
                hi = mid;
                best = t;
            } else {
                lo = mid + 1;
            }
        }
    }
    let uniform = hi;
    let mut used = best.0;

    // Per-slice refinement: hand out the slack, slice by slice, each
    // slice's share in proportion to its macroblocks (what one slice leaves
    // goes to those after it), and each slice steps to finer quantisers
    // while it fits its share. The uniform quantiser's slack is under one
    // step for the picture, so few slices move more than a step or two:
    // their sizes at the next `WINDOW` quantisers are found in parallel
    // first, and the hand-out, which is sequential, rarely codes anything.
    const WINDOW: usize = 3;
    let mut qindices = vec![uniform; slices.len()];
    if uniform > 1 && best.1 {
        let window = (uniform as usize - 1).min(WINDOW);
        let finer: Vec<[(usize, bool); WINDOW]> = pool::map(threads, slices.len(), |i| {
            std::array::from_fn(|j| if j < window { slice_size(&slices[i], uniform - 1 - j as u8) } else { (0, false) })
        });
        let mut remaining_budget = budget;
        let mut remaining_base: usize = used.iter().sum();
        let mut remaining_mbs: u64 = slices.iter().map(|s| s.mbs as u64).sum();
        for (i, s) in slices.iter().enumerate() {
            let base = used[i];
            let slack = (remaining_budget - remaining_base) as u64;
            let allowed = base + (slack * s.mbs as u64 / remaining_mbs) as usize;
            let mut qi = uniform;
            while qi > 1 {
                let step = (uniform - qi) as usize;
                let (n, fits) = if step < window { finer[i][step] } else { slice_size(s, qi - 1) };
                if !(fits && n <= allowed) {
                    break;
                }
                qi -= 1;
                used[i] = n;
            }
            qindices[i] = qi;
            remaining_budget -= used[i];
            remaining_base -= base;
            remaining_mbs -= s.mbs as u64;
        }
    }

    // Code every slice, in parallel, then write the picture.
    let coded = pool::map(threads, slices.len(), |i| {
        let (s, qi) = (&slices[i], qindices[i]);
        let mut bytes = Vec::with_capacity(used[i]);
        bytes.push((header_bytes as u8) << 3);
        bytes.push(qi);
        bytes.resize(header_bytes, 0);
        let mut comp_sizes = [0usize; 3];
        SCRATCH.with_borrow_mut(|q| {
            for (plane, size) in comp_sizes.iter_mut().enumerate() {
                q.quantise(isa, s.comps[plane], quantisers.get(qi, plane));
                let mut bw = BitWriter::new(&mut bytes);
                encode_coefficients_masked(&mut bw, q, s.comps[plane].len() / 64);
                *size = bw.finish();
            }
        });
        bytes[2..4].copy_from_slice(&(comp_sizes[0] as u16).to_be_bytes());
        bytes[4..6].copy_from_slice(&(comp_sizes[1] as u16).to_be_bytes());
        if header_bytes == 8 {
            bytes[6..8].copy_from_slice(&(comp_sizes[2] as u16).to_be_bytes());
        }
        bytes.extend_from_slice(&s.alpha);
        bytes
    });
    let pic_start = out.len();
    PictureHeader::write(out, 0, slices.len(), log2_slice_mbs);
    for s in &coded {
        out.extend_from_slice(&(s.len() as u16).to_be_bytes());
    }
    for s in &coded {
        out.extend_from_slice(s);
    }
    let picture_size = (out.len() - pic_start) as u32;
    out[pic_start + 1..pic_start + 5].copy_from_slice(&picture_size.to_be_bytes());
}

/// One slice's alpha, converted to `bits` bits and coded: its whole
/// macroblocks, 16 rows of `16 · mbs` values (§5.3.3). Columns past the
/// frame's right edge repeat the last one and rows past the picture's
/// bottom (`height` rows) the last row (§7.5.3: they are coded and
/// discarded).
#[allow(clippy::too_many_arguments)]
fn slice_alpha(
    frame: &Frame,
    alpha: &[u16],
    bits: u32,
    mb_x: u32,
    mbs: u32,
    mb_y: usize,
    height: usize,
    first_row: usize,
    step: usize,
) -> Vec<u8> {
    let width = frame.width as usize;
    let max_in = u16::MAX >> (16 - frame.bit_depth);
    // round(max_out · a ÷ max_in)
    let convert = Rescale::new(frame.bit_depth, bits);
    let cols = 16 * mbs as usize;
    let x0 = 16 * mb_x as usize;
    let mut values = Vec::with_capacity(cols * 16);
    for r in 0..16 {
        let row = first_row + step * (16 * mb_y + r).min(height - 1);
        for n in 0..cols {
            let a = alpha[row * width + (x0 + n).min(width - 1)].min(max_in);
            values.push(convert.apply(a));
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
                // The rate control's count is what the writer writes.
                let mut mask = vec![0u64; n_blocks];
                for (i, &v) in c.iter().enumerate() {
                    mask[i / 64] |= ((v != 0) as u64) << (i % 64);
                }
                let q = Quantised { values: c.clone(), mask };
                assert_eq!(coefficient_bits(&q, n_blocks).div_ceil(8), bytes.len());
                // And the decoder's block layout is the scan, undone.
                for scan in [&PROGRESSIVE_SCAN, &INTERLACED_SCAN] {
                    let unscan = crate::decode::inverse_scan(scan);
                    let mut blocks = vec![0i32; n_blocks * 64];
                    crate::decode::decode_coefficients_raster(&bytes, n_blocks, &unscan, &mut blocks).unwrap();
                    for b in 0..n_blocks {
                        for k in 0..64 {
                            assert_eq!(blocks[64 * b + k], c[n_blocks * scan[k] as usize + b]);
                        }
                    }
                }
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
            assert_eq!(decode_alpha(&bytes, bits, v.len(), v.len()).unwrap(), v);
        }
    }

    /// A bottom slice's alpha covers its whole macroblocks: 16 rows, those
    /// below the picture repeating its last row (§5.3.3, §7.5.3), so a
    /// decoder that reads 256 values per macroblock finds them all.
    #[test]
    fn slice_alpha_codes_whole_macroblocks() {
        // 40×20: two macroblock rows, the second with 4 picture rows.
        let mut frame = Frame::new(40, 20, ChromaFormat::Yuv444, 8).unwrap();
        let alpha: Vec<u16> = (0..20u16).flat_map(|y| (0..40u16).map(move |x| 10 * y + x % 3)).collect();
        frame.alpha = Some(alpha.clone());
        let (mbs, cols) = (3u32, 48usize);
        let bytes = slice_alpha(&frame, &alpha, 8, 0, mbs, 1, 20, 0, 1);
        let got = decode_alpha(&bytes, 8, 16 * cols, 16 * cols).unwrap();
        assert_eq!(got.len(), 16 * cols);
        for r in 0..16 {
            for n in 0..cols {
                let (y, x) = ((16 + r).min(19), n.min(39));
                assert_eq!(got[r * cols + n], alpha[y * 40 + x], "row {r} column {n}");
            }
        }
    }

    /// Alpha coded only down to the picture's last row (as this encoder
    /// wrote it before 2026-10) is still read: the decoder stops where the
    /// data does once it has the values it uses.
    #[test]
    fn alpha_that_stops_at_the_picture_is_read() {
        let v: Vec<u16> = (0..4 * 32).map(|i| (i / 32) as u16 * 50).collect();
        let mut bytes = Vec::new();
        let mut w = BitWriter::new(&mut bytes);
        encode_alpha(&mut w, &v, 8);
        w.finish();
        let got = decode_alpha(&bytes, 8, 16 * 32, v.len()).unwrap();
        assert_eq!(&got[..v.len()], &v[..]);
        assert!(decode_alpha(&bytes, 8, 16 * 32, v.len() + 1).is_err());
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
