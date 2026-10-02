//! The decoding process of RDD 36 §7: entropy decoding (§7.1), inverse
//! scanning (§7.2), inverse quantisation (§7.3), the IDCT (§7.4) and sample
//! generation and placement (§7.5).

use crate::bits::BitReader;
use crate::dct::idct;
use crate::error::{Result, config, invalid};
use crate::frame::{ChromaFormat, Frame, Interlace};
use crate::header::{AlphaType, FrameHeader, PictureHeader, slice_sizes};
use crate::tables::{INTERLACED_SCAN, PROGRESSIVE_SCAN, qscale};
use crate::vlc::{self, FIRST_DC_CODEBOOK, symbol_to_signed};

/// A ProRes decoder. Every frame stands alone (ProRes is intra-only), so
/// the decoder holds nothing but the output bit depth and can decode frames
/// in any order, from any number of threads.
#[derive(Debug, Clone, Default)]
pub struct Decoder {
    bit_depth: Option<u32>,
}

impl Decoder {
    /// A decoder that outputs each frame at its natural depth: 10 bits for
    /// 4:2:2 (the 422 profiles), 12 bits for 4:4:4 (the 4444 profiles).
    pub fn new() -> Decoder {
        Decoder { bit_depth: None }
    }

    /// A decoder that outputs every frame at `bit_depth` bits (8–16),
    /// colour and alpha alike (§7.5 defines the conversion for any depth).
    pub fn with_bit_depth(bit_depth: u32) -> Result<Decoder> {
        if !(8..=16).contains(&bit_depth) {
            return Err(config(format!("output bit depth {bit_depth} is outside 8–16")));
        }
        Ok(Decoder { bit_depth: Some(bit_depth) })
    }

    /// Decodes one frame: the bytes of a MOV sample, starting at
    /// `frame_size`. Bytes after `frame_size` are ignored.
    pub fn decode(&self, data: &[u8]) -> Result<Frame> {
        let hdr = FrameHeader::parse(data)?;
        let bit_depth = self.bit_depth.unwrap_or(match hdr.chroma {
            ChromaFormat::Yuv422 => 10,
            ChromaFormat::Yuv444 => 12,
        });
        let mut frame = Frame::new(hdr.width as u32, hdr.height as u32, hdr.chroma, bit_depth)?;
        frame.interlace = hdr.interlace;
        frame.metadata = hdr.metadata;
        if hdr.alpha != AlphaType::None {
            frame.alpha = Some(vec![0; frame.width as usize * frame.height as usize]);
        }
        let data = &data[..hdr.frame_size as usize];
        let mut pos = 8 + hdr.header_size as usize;
        let ctx = Context::new(&hdr, bit_depth);
        for index in 0..hdr.picture_count() {
            let rest = data.get(pos..).ok_or_else(|| invalid("a picture starts past the end of the frame"))?;
            let ph = PictureHeader::parse(rest)?;
            decode_picture(&ctx, index, &ph, &rest[..ph.picture_size], &mut frame)?;
            pos += ph.picture_size;
        }
        Ok(frame)
    }
}

/// What every slice of a frame shares.
struct Context<'a> {
    hdr: &'a FrameHeader,
    luma_weights: [u8; 64],
    chroma_weights: [u8; 64],
    scan: &'static [u8; 64],
    bit_depth: u32,
}

impl<'a> Context<'a> {
    fn new(hdr: &'a FrameHeader, bit_depth: u32) -> Self {
        Context {
            hdr,
            luma_weights: hdr.luma_weights(),
            chroma_weights: hdr.chroma_weights(),
            scan: if hdr.interlace == Interlace::Progressive { &PROGRESSIVE_SCAN } else { &INTERLACED_SCAN },
            bit_depth,
        }
    }
}

/// Where one picture's rows land in the frame (§7.5.3).
#[derive(Clone, Copy)]
struct Placement {
    /// Luma rows in this picture (`picture_vertical_size`).
    height: usize,
    /// Frame row of picture row 0, and the step between picture rows.
    first_row: usize,
    row_step: usize,
}

fn decode_picture(ctx: &Context, index: usize, ph: &PictureHeader, pic: &[u8], frame: &mut Frame) -> Result<()> {
    let hdr = ctx.hdr;
    let width_in_mb = (hdr.width as u32).div_ceil(16);
    let height = hdr.picture_height(index);
    let height_in_mb = height.div_ceil(16);
    let sizes = slice_sizes(width_in_mb, ph.log2_slice_mbs);
    let n_slices = sizes.len() * height_in_mb as usize;
    let table = pic
        .get(ph.header_size..ph.header_size + 2 * n_slices)
        .ok_or_else(|| invalid("the slice table runs past picture_size"))?;
    let (first_row, row_step) = hdr.picture_rows(index);
    let place = Placement { height: height as usize, first_row, row_step };
    let mut pos = ph.header_size + 2 * n_slices;
    let mut scratch = Vec::new();
    for i in 0..height_in_mb as usize {
        let mut mb_x = 0u32;
        for (j, &mbs) in sizes.iter().enumerate() {
            let k = i * sizes.len() + j;
            let size = u16::from_be_bytes([table[2 * k], table[2 * k + 1]]) as usize;
            let slice = pic.get(pos..pos + size).ok_or_else(|| invalid("a slice runs past picture_size"))?;
            decode_slice(ctx, slice, i, mb_x, mbs, height_in_mb as usize, place, frame, &mut scratch)?;
            pos += size;
            mb_x += mbs;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_slice(
    ctx: &Context,
    s: &[u8],
    mb_y: usize,
    mb_x: u32,
    mbs: u32,
    height_in_mb: usize,
    place: Placement,
    frame: &mut Frame,
    coeffs: &mut Vec<i32>,
) -> Result<()> {
    let hdr = ctx.hdr;
    let has_alpha = hdr.alpha != AlphaType::None;
    let min_header = if has_alpha { 8 } else { 6 };
    if s.len() < min_header {
        return Err(invalid("a slice is shorter than its header"));
    }
    let header_size = (s[0] >> 3) as usize;
    if header_size < min_header || header_size > s.len() {
        return Err(invalid(format!("slice_header_size {header_size} is out of range")));
    }
    let qindex = s[1];
    if !(1..=224).contains(&qindex) {
        return Err(invalid(format!("quantization_index {qindex} is reserved")));
    }
    let q = qscale(qindex);
    let y_size = u16::from_be_bytes([s[2], s[3]]) as usize;
    let cb_size = u16::from_be_bytes([s[4], s[5]]) as usize;
    let body = s.len() - header_size;
    let cr_size = if has_alpha {
        u16::from_be_bytes([s[6], s[7]]) as usize
    } else {
        body.checked_sub(y_size + cb_size).ok_or_else(|| invalid("slice component sizes exceed the slice"))?
    };
    if y_size + cb_size + cr_size > body {
        return Err(invalid("slice component sizes exceed the slice"));
    }
    let mut pos = header_size;
    let mut take = |n: usize| {
        let d = &s[pos..pos + n];
        pos += n;
        d
    };
    let (y_data, cb_data, cr_data) = (take(y_size), take(cb_size), take(cr_size));
    let alpha_data = &s[pos..];

    let chroma_blocks_per_mb = match hdr.chroma {
        ChromaFormat::Yuv422 => 2,
        ChromaFormat::Yuv444 => 4,
    };
    let components: [(&[u8], usize, &[u8; 64]); 3] = [
        (y_data, 4, &ctx.luma_weights),
        (cb_data, chroma_blocks_per_mb, &ctx.chroma_weights),
        (cr_data, chroma_blocks_per_mb, &ctx.chroma_weights),
    ];
    for (plane, (data, per_mb, weights)) in components.into_iter().enumerate() {
        let n_blocks = per_mb * mbs as usize;
        coeffs.clear();
        coeffs.resize(n_blocks * 64, 0);
        decode_coefficients(data, n_blocks, coeffs)?;
        // F = QF · W · qScale / 8 (§7.3), folded into one factor per position.
        let mut scale = [0f32; 64];
        for k in 0..64 {
            scale[k] = (weights[k] as u32 * q) as f32 / 8.0;
        }
        for blk in 0..n_blocks {
            let mut f = [0f32; 64];
            for k in 0..64 {
                let qf = coeffs[n_blocks * ctx.scan[k] as usize + blk];
                if qf != 0 {
                    f[k] = qf as f32 * scale[k];
                }
            }
            let v = idct(&f);
            let (bx, by) = block_position(plane, hdr.chroma, mb_x, mb_y, blk / per_mb, blk % per_mb);
            put_block(frame, plane, bx, by, &v, place, ctx.bit_depth);
        }
    }

    if has_alpha {
        let rows = if mb_y + 1 < height_in_mb { 16 } else { place.height - 16 * (height_in_mb - 1) };
        let cols = 16 * mbs as usize;
        let values = decode_alpha(alpha_data, hdr.alpha.bits(), cols * rows)?;
        let max_in = (1u64 << hdr.alpha.bits()) - 1;
        let max_out = (1u64 << ctx.bit_depth) - 1;
        let width = frame.width as usize;
        let x0 = 16 * mb_x as usize;
        let alpha = frame.alpha.as_mut().expect("allocated for an alpha frame");
        for r in 0..rows {
            let row = place.first_row + place.row_step * (16 * mb_y + r);
            for n in 0..cols.min(width.saturating_sub(x0)) {
                let a = values[cols * r + n] as u64;
                // §7.5.2: round((2^b − 1) · alpha ÷ (2^bits − 1)).
                let out = if max_in == max_out { a } else { (2 * max_out * a + max_in) / (2 * max_in) };
                alpha[row * width + x0 + n] = out as u16;
            }
        }
    }
    Ok(())
}

/// Top-left sample of block `b` of macroblock `m` of a slice, in the
/// component's picture coordinates (§7.5.3, Figures 6–8).
pub(crate) fn block_position(plane: usize, chroma: ChromaFormat, mb_x: u32, mb_y: usize, m: usize, b: usize) -> (usize, usize) {
    let mb_x = mb_x as usize + m;
    let y0 = 16 * mb_y;
    match (plane, chroma) {
        // Luma: 0 1 / 2 3.
        (0, _) => (16 * mb_x + 8 * (b & 1), y0 + 8 * (b >> 1)),
        // 4:2:2 chroma: 0 above 1, the macroblock 8 samples wide.
        (_, ChromaFormat::Yuv422) => (8 * mb_x, y0 + 8 * b),
        // 4:4:4 chroma: 0 2 / 1 3.
        (_, ChromaFormat::Yuv444) => (16 * mb_x + 8 * (b >> 1), y0 + 8 * (b & 1)),
    }
}

/// Converts reconstructed values to samples (§7.5.1,
/// `s = clamp(round(2^b · (v + 256) ÷ 512))`, clamped to 0..2^b − 1) and
/// writes those inside the frame.
fn put_block(frame: &mut Frame, plane: usize, bx: usize, by: usize, v: &[f32; 64], place: Placement, bit_depth: u32) {
    let p = frame.planes[plane];
    let (pw, stride) = (p.width as usize, p.width as usize);
    if bx >= pw {
        return;
    }
    let scale = (1u32 << bit_depth) as f32 / 512.0;
    let bias = (1u32 << (bit_depth - 1)) as f32 + 0.5;
    let max = ((1u32 << bit_depth) - 1) as f32;
    let cols = (pw - bx).min(8);
    let data = &mut frame.data[p.offset..p.offset + p.len()];
    for y in 0..8 {
        let pic_row = by + y;
        if pic_row >= place.height {
            break;
        }
        let row = place.first_row + place.row_step * pic_row;
        let out = &mut data[row * stride + bx..row * stride + bx + cols];
        for (x, o) in out.iter_mut().enumerate() {
            *o = (v[8 * y + x] * scale + bias).floor().clamp(0.0, max) as u16;
        }
    }
}

/// `scanned_coefficients()` (§5.3.2, §7.1.1): fills `coeffs` (zeroed,
/// `n_blocks * 64` long) in scanned order.
pub(crate) fn decode_coefficients(data: &[u8], n_blocks: usize, coeffs: &mut [i32]) -> Result<()> {
    let mut r = BitReader::new(data);
    // DC coefficients (§7.1.1.3).
    let first = symbol_to_signed(FIRST_DC_CODEBOOK.read(&mut r)?);
    coeffs[0] = first;
    let mut prev_dc = first;
    let mut prev_diff: i32 = 3;
    for c in coeffs.iter_mut().take(n_blocks).skip(1) {
        let n = symbol_to_signed(vlc::dc_codebook(prev_diff).read(&mut r)?);
        let diff = if prev_diff < 0 { -n } else { n };
        prev_dc = prev_dc.checked_add(diff).ok_or_else(|| invalid("a DC coefficient overflows"))?;
        *c = prev_dc;
        prev_diff = diff;
    }
    // AC coefficients (§7.1.1.4): runs of zeros, each ended by a level.
    let total = n_blocks * 64;
    let mut n = n_blocks;
    let mut prev_run = 4u32;
    let mut prev_level = 1u32;
    while !r.end_of_data() {
        let run = vlc::run_codebook(prev_run).read(&mut r)?;
        prev_run = run;
        n += run as usize;
        if n >= total {
            return Err(invalid("a run passes the end of the coefficient array"));
        }
        let level_symbol = vlc::level_codebook(prev_level).read(&mut r)?;
        prev_level = level_symbol;
        let level = level_symbol as i32 + 1;
        coeffs[n] = if r.read_bit()? == 1 { -level } else { level };
        n += 1;
    }
    Ok(())
}

/// `scanned_alpha()` (§5.3.3, §7.1.2): `count` raster-scanned values.
pub(crate) fn decode_alpha(data: &[u8], bits: u32, count: usize) -> Result<Vec<u16>> {
    let mut r = BitReader::new(data);
    let mask = (1i32 << bits) - 1;
    let mut out = Vec::with_capacity(count);
    let mut prev: i32 = -1;
    while out.len() < count {
        let (diff, _modulo) = vlc::read_alpha_difference(&mut r, bits)?;
        // Masking an exact difference changes nothing (§7.1.2's note).
        let alpha = prev.wrapping_add(diff) & mask;
        prev = alpha;
        let run = vlc::read_alpha_run(&mut r)? as usize;
        if out.len() + run > count {
            return Err(invalid("an alpha run passes the end of the slice"));
        }
        out.resize(out.len() + run, alpha as u16);
    }
    Ok(out)
}
