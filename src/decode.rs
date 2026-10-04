//! The decoding process of RDD 36 §7: entropy decoding (§7.1), inverse
//! scanning (§7.2), inverse quantisation (§7.3), the IDCT (§7.4) and sample
//! generation and placement (§7.5).

use crate::bits::BitReader;
use crate::dsp::{self, Isa, Output};
use crate::error::{Result, config, invalid};
use crate::frame::{ChromaFormat, Frame, Interlace};
use crate::header::{AlphaType, FrameHeader, PictureHeader, slice_sizes};
use crate::pool;
use crate::tables::{INTERLACED_SCAN, PROGRESSIVE_SCAN, qscale};
use crate::vlc::{self, FIRST_DC_CODEBOOK, symbol_to_signed};
use std::sync::Mutex;

/// A ProRes decoder. Every frame stands alone (ProRes is intra-only), so
/// the decoder holds nothing but its settings and can decode frames in any
/// order, from any number of threads.
#[derive(Debug, Clone, Default)]
pub struct Decoder {
    bit_depth: Option<u32>,
    threads: usize,
}

impl Decoder {
    /// A decoder that outputs each frame at its natural depth: 10 bits for
    /// 4:2:2 (the 422 profiles), 12 bits for 4:4:4 (the 4444 profiles).
    /// It decodes on one thread per CPU ([`Decoder::with_threads`]).
    pub fn new() -> Decoder {
        Decoder { bit_depth: None, threads: 0 }
    }

    /// A decoder that outputs every frame at `bit_depth` bits (8–16),
    /// colour and alpha alike (§7.5 defines the conversion for any depth).
    pub fn with_bit_depth(bit_depth: u32) -> Result<Decoder> {
        if !(8..=16).contains(&bit_depth) {
            return Err(config(format!("output bit depth {bit_depth} is outside 8–16")));
        }
        Ok(Decoder { bit_depth: Some(bit_depth), threads: 0 })
    }

    /// Decodes on up to `threads` threads: the calling one and workers
    /// shared by every encoder and decoder in the process. 0, the default,
    /// means one per CPU; 1 keeps all the work on the calling thread. A
    /// picture's rows of slices are independent, so the output is the same
    /// at any thread count.
    pub fn with_threads(mut self, threads: usize) -> Decoder {
        self.threads = threads;
        self
    }

    /// Decodes one frame: the bytes of a MOV sample, starting at
    /// `frame_size`. Bytes after `frame_size` are ignored.
    pub fn decode(&self, data: &[u8]) -> Result<Frame> {
        let hdr = FrameHeader::parse(data)?;
        check_plausible_size(&hdr)?;
        let bit_depth = self.bit_depth.unwrap_or(match hdr.chroma {
            ChromaFormat::Yuv422 => 10,
            ChromaFormat::Yuv444 => 12,
        });
        // Every sample is written: zeroes need not be.
        let mut frame = Frame::zeroed(hdr.width as u32, hdr.height as u32, hdr.chroma, bit_depth)?;
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
            decode_picture(&ctx, index, &ph, &rest[..ph.picture_size], &mut frame, self.threads)?;
            pos += ph.picture_size;
        }
        Ok(frame)
    }
}

/// Refuses, before the frame buffer is allocated, a frame too small to
/// hold the slices its dimensions imply: each slice needs at least 11
/// bytes (its slice table entry, a 6-byte header and a byte of data per
/// colour component), and a picture has at least one slice per 8
/// macroblocks. Without this a few bytes claiming 65535×65535 would cost
/// gigabytes before the first error.
fn check_plausible_size(hdr: &FrameHeader) -> Result<()> {
    let width_in_mb = (hdr.width as u64).div_ceil(16);
    let mut need = 8 + hdr.header_size as u64;
    for index in 0..hdr.picture_count() {
        let slices = slice_sizes(width_in_mb as u32, 3).len() as u64 * (hdr.picture_height(index) as u64).div_ceil(16);
        need += crate::header::PICTURE_HEADER_SIZE as u64 + 11 * slices;
    }
    if (hdr.frame_size as u64) < need {
        return Err(invalid(format!(
            "frame_size {} is too small for a {}×{} frame (at least {need} bytes)",
            hdr.frame_size, hdr.width, hdr.height
        )));
    }
    Ok(())
}

/// What every slice of a frame shares.
struct Context<'a> {
    hdr: &'a FrameHeader,
    luma_weights: [u8; 64],
    chroma_weights: [u8; 64],
    /// Raster position of each scanned index: the inverse of the scan.
    unscan: [u8; 64],
    bit_depth: u32,
    output: Output,
    isa: Isa,
}

impl<'a> Context<'a> {
    fn new(hdr: &'a FrameHeader, bit_depth: u32) -> Self {
        Context {
            hdr,
            luma_weights: hdr.luma_weights(),
            chroma_weights: hdr.chroma_weights(),
            unscan: inverse_scan(if hdr.interlace == Interlace::Progressive {
                &PROGRESSIVE_SCAN
            } else {
                &INTERLACED_SCAN
            }),
            bit_depth,
            output: Output::new(bit_depth),
            isa: Isa::get(),
        }
    }
}

/// Where one picture's rows land in the frame (§7.5.3).
#[derive(Clone, Copy)]
struct Placement {
    /// Luma rows in this picture (`picture_vertical_size`).
    height: usize,
    /// Frame rows between picture rows: 1 for a frame, 2 for a field.
    row_step: usize,
}

/// The frame rows one row of macroblocks of a picture covers — 16 picture
/// rows, `16 · row_step` frame rows from the first — in each plane and the
/// alpha. Disjoint from every other band of the same picture, so the rows
/// decode in parallel.
struct Band<'a> {
    planes: [&'a mut [u16]; 3],
    widths: [usize; 3],
    alpha: Option<&'a mut [u16]>,
}

/// A slice, located: its first macroblock, its macroblock count and its
/// bytes.
#[derive(Clone, Copy)]
struct SliceRef<'a> {
    mb_x: u32,
    mbs: u32,
    data: &'a [u8],
}

fn decode_picture(
    ctx: &Context,
    index: usize,
    ph: &PictureHeader,
    pic: &[u8],
    frame: &mut Frame,
    threads: usize,
) -> Result<()> {
    let hdr = ctx.hdr;
    let width_in_mb = (hdr.width as u32).div_ceil(16);
    let height = hdr.picture_height(index);
    let height_in_mb = height.div_ceil(16) as usize;
    let sizes = slice_sizes(width_in_mb, ph.log2_slice_mbs);
    let n_slices = sizes.len() * height_in_mb;
    let table = pic
        .get(ph.header_size..ph.header_size + 2 * n_slices)
        .ok_or_else(|| invalid("the slice table runs past picture_size"))?;
    let (first_row, row_step) = hdr.picture_rows(index);
    let place = Placement { height: height as usize, row_step };

    // Locate every slice first (the table gives sizes, not offsets).
    let mut pos = ph.header_size + 2 * n_slices;
    let mut slices = Vec::with_capacity(n_slices);
    for k in 0..n_slices {
        let size = u16::from_be_bytes([table[2 * k], table[2 * k + 1]]) as usize;
        let data = pic.get(pos..pos + size).ok_or_else(|| invalid("a slice runs past picture_size"))?;
        slices.push(data);
        pos += size;
    }

    // Split the frame into one band per row of slices.
    let band_rows = 16 * row_step;
    let widths = frame.planes.map(|p| p.width as usize);
    let mut bands: Vec<Mutex<Option<Band>>> = Vec::with_capacity(height_in_mb);
    {
        let (y, rest) = frame.data.split_at_mut(frame.planes[1].offset);
        let (cb, cr) = rest.split_at_mut(frame.planes[2].offset - frame.planes[1].offset);
        let cr = &mut cr[..frame.planes[2].len()];
        fn split(plane: &mut [u16], first_row: usize, band_rows: usize, w: usize) -> std::slice::ChunksMut<'_, u16> {
            let start = (first_row * w).min(plane.len());
            plane[start..].chunks_mut(band_rows * w)
        }
        let split = |plane, w| split(plane, first_row, band_rows, w);
        let (mut it_y, mut it_cb, mut it_cr) = (split(y, widths[0]), split(cb, widths[1]), split(cr, widths[2]));
        let mut it_a = frame.alpha.as_mut().map(|a| split(a, widths[0]));
        for _ in 0..height_in_mb {
            let (Some(py), Some(pcb), Some(pcr)) = (it_y.next(), it_cb.next(), it_cr.next()) else {
                return Err(invalid("the picture has more rows of macroblocks than the frame"));
            };
            let alpha = match &mut it_a {
                Some(it) => Some(it.next().ok_or_else(|| invalid("the picture has more rows than the alpha"))?),
                None => None,
            };
            bands.push(Mutex::new(Some(Band { planes: [py, pcb, pcr], widths, alpha })));
        }
    }

    let row_slices = sizes.len();
    let results = pool::map(threads, height_in_mb, |i| {
        let mut band = bands[i].lock().unwrap_or_else(|e| e.into_inner()).take().expect("each band once");
        let mut scratch = Vec::new();
        let mut mb_x = 0u32;
        for (j, &mbs) in sizes.iter().enumerate() {
            let slice = SliceRef { mb_x, mbs, data: slices[i * row_slices + j] };
            decode_slice(ctx, slice, i, height_in_mb, place, &mut band, &mut scratch)?;
            mb_x += mbs;
        }
        Ok(())
    });
    // The first error in picture order, whichever thread met it.
    results.into_iter().collect()
}

fn decode_slice(
    ctx: &Context,
    slice: SliceRef,
    mb_y: usize,
    height_in_mb: usize,
    place: Placement,
    band: &mut Band,
    coeffs: &mut Vec<i32>,
) -> Result<()> {
    let SliceRef { mb_x, mbs, data: s } = slice;
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
    let band_y = 16 * mb_y;
    for (plane, (data, per_mb, weights)) in components.into_iter().enumerate() {
        let n_blocks = per_mb * mbs as usize;
        coeffs.clear();
        coeffs.resize(n_blocks * 64, 0);
        decode_coefficients_raster(data, n_blocks, &ctx.unscan, coeffs)?;
        // F = QF · W · qScale / 8 (§7.3), folded into one factor per position.
        let scale: [f32; 64] = std::array::from_fn(|k| (weights[k] as u32 * q) as f32 / 8.0);
        for (blk, coef) in coeffs.as_chunks::<64>().0.iter().enumerate() {
            let v = dsp::idct_put(ctx.isa, coef, &scale, ctx.output);
            let (bx, by) = block_position(plane, hdr.chroma, mb_x, mb_y, blk / per_mb, blk % per_mb);
            put_block(band, plane, bx, by - band_y, band_y, &v, place);
        }
    }

    if has_alpha {
        // The slice's alpha covers its whole macroblocks, 16 rows of
        // 16 · mbs values (§5.3.3), like its colour; values outside the
        // picture are discarded (§7.5.3). A bottom slice that stops at the
        // picture's last row is read too: the rows below are not used.
        let rows = if mb_y + 1 < height_in_mb { 16 } else { place.height - 16 * (height_in_mb - 1) };
        let cols = 16 * mbs as usize;
        let values = decode_alpha(alpha_data, hdr.alpha.bits(), cols * 16, cols * rows)?;
        // §7.5.2: round((2^b − 1) · alpha ÷ (2^bits − 1)).
        let convert = Rescale::new(hdr.alpha.bits(), ctx.bit_depth);
        let width = band.widths[0];
        let x0 = 16 * mb_x as usize;
        let n = cols.min(width.saturating_sub(x0));
        let alpha = band.alpha.as_mut().expect("allocated for an alpha frame");
        for r in 0..rows {
            let out = &mut alpha[place.row_step * r * width + x0..][..n];
            for (o, &a) in out.iter_mut().zip(&values[cols * r..cols * r + n]) {
                *o = convert.apply(a);
            }
        }
    }
    Ok(())
}

/// Top-left sample of block `b` of macroblock `m` of a slice, in the
/// component's picture coordinates (§7.5.3, Figures 6–8).
pub(crate) fn block_position(
    plane: usize,
    chroma: ChromaFormat,
    mb_x: u32,
    mb_y: usize,
    m: usize,
    b: usize,
) -> (usize, usize) {
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

/// Writes the samples of a block (§7.5.1, already converted) that fall
/// inside the picture: `by` is the block's first row within the band,
/// `band_y` the band's first picture row.
fn put_block(band: &mut Band, plane: usize, bx: usize, by: usize, band_y: usize, v: &[u16; 64], place: Placement) {
    let pw = band.widths[plane];
    if bx >= pw {
        return;
    }
    let cols = (pw - bx).min(8);
    let rows = place.height.saturating_sub(band_y + by).min(8);
    let data = &mut *band.planes[plane];
    for (y, src) in v.as_chunks::<8>().0.iter().take(rows).enumerate() {
        let row = place.row_step * (by + y);
        data[row * pw + bx..][..cols].copy_from_slice(&src[..cols]);
    }
}

/// `round((2^out − 1) · a ÷ (2^in − 1))` (§7.5.2) for an `in`-bit `a`,
/// without a division: `(a · m) >> 40` with `m` the quotient rounded up,
/// plus a half. Ties cannot occur (`2^in − 1` is odd, so the exact quotient
/// is never an odd multiple of ½), and the tests check every value of
/// every pair of depths.
#[derive(Clone, Copy)]
pub(crate) struct Rescale {
    m: u64,
    identity: bool,
}

impl Rescale {
    pub(crate) fn new(bits_in: u32, bits_out: u32) -> Rescale {
        let (max_in, max_out) = ((1u64 << bits_in) - 1, (1u64 << bits_out) - 1);
        let m = (max_out << 40).div_ceil(max_in);
        Rescale { m, identity: bits_in == bits_out }
    }

    #[inline]
    pub(crate) fn apply(self, a: u16) -> u16 {
        if self.identity { a } else { ((a as u64 * self.m + (1 << 39)) >> 40) as u16 }
    }
}

/// The inverse of a scan: `out[scan[k]] = k`.
pub(crate) fn inverse_scan(scan: &[u8; 64]) -> [u8; 64] {
    let mut out = [0u8; 64];
    for (k, &s) in scan.iter().enumerate() {
        out[s as usize] = k as u8;
    }
    out
}

/// `scanned_coefficients()` (§5.3.2, §7.1.1), calling `put(n, value)` for
/// the coefficient at index `n` of the scanned array (`n_blocks * 64`
/// long; positions not reported are zero).
#[inline(always)]
fn decode_scanned(data: &[u8], n_blocks: usize, mut put: impl FnMut(usize, i32)) -> Result<()> {
    let mut r = BitReader::new(data);
    // DC coefficients (§7.1.1.3).
    let first = symbol_to_signed(FIRST_DC_CODEBOOK.read(&mut r)?);
    put(0, first);
    let mut prev_dc = first;
    let mut prev_diff: i32 = 3;
    for i in 1..n_blocks {
        let n = symbol_to_signed(vlc::dc_codebook(prev_diff).read(&mut r)?);
        let diff = if prev_diff < 0 { -n } else { n };
        prev_dc = prev_dc.checked_add(diff).ok_or_else(|| invalid("a DC coefficient overflows"))?;
        put(i, prev_dc);
        prev_diff = diff;
    }
    // AC coefficients (§7.1.1.4): runs of zeros, each ended by a level.
    let total = n_blocks * 64;
    let mut n = n_blocks;
    let mut prev_run = 4u32;
    let mut prev_level = 1u32;
    loop {
        // Fast path: run, level and sign from one 64-bit window.
        if let Some(w) = r.peek_fast()
            && let Some((run, l1)) = vlc::peek_run(prev_run, w)
            && let Some((level_symbol, l2)) = vlc::peek_level(prev_level, w << l1)
            && l1 + l2 < 57
        {
            n += run as usize;
            if n >= total {
                return Err(invalid("a run passes the end of the coefficient array"));
            }
            let level = level_symbol as i32 + 1;
            let negative = (w << (l1 + l2)) >> 63 == 1;
            r.skip(l1 + l2 + 1);
            put(n, if negative { -level } else { level });
            n += 1;
            (prev_run, prev_level) = (run, level_symbol);
            continue;
        }
        if r.end_of_data() {
            break;
        }
        let run = vlc::run_codebook(prev_run).read(&mut r)?;
        prev_run = run;
        n += run as usize;
        if n >= total {
            return Err(invalid("a run passes the end of the coefficient array"));
        }
        let level_symbol = vlc::level_codebook(prev_level).read(&mut r)?;
        prev_level = level_symbol;
        let level = level_symbol as i32 + 1;
        put(n, if r.read_bit()? == 1 { -level } else { level });
        n += 1;
    }
    Ok(())
}

/// `scanned_coefficients()` into the scanned array itself: `coeffs`
/// (zeroed, `n_blocks * 64` long) in scanned order.
#[cfg(test)]
pub(crate) fn decode_coefficients(data: &[u8], n_blocks: usize, coeffs: &mut [i32]) -> Result<()> {
    decode_scanned(data, n_blocks, |n, v| coeffs[n] = v)
}

/// `scanned_coefficients()` straight into blocks (§7.2): `out` (zeroed,
/// `n_blocks * 64` long) gets block `b`'s coefficients at `64 · b`, in
/// raster order. Scanned index `n` is position `n / n_blocks` of the scan
/// of block `n % n_blocks` (§7.2.1); `n_blocks` is a power of two (2 or 4
/// blocks per macroblock, 1, 2, 4 or 8 macroblocks per slice).
pub(crate) fn decode_coefficients_raster(
    data: &[u8],
    n_blocks: usize,
    unscan: &[u8; 64],
    out: &mut [i32],
) -> Result<()> {
    debug_assert!(n_blocks.is_power_of_two());
    let shift = n_blocks.trailing_zeros();
    let mask = n_blocks - 1;
    decode_scanned(data, n_blocks, |n, v| out[((n & mask) << 6) | unscan[n >> shift] as usize] = v)
}

/// `scanned_alpha()` (§5.3.3, §7.1.2): `count` raster-scanned values, or,
/// when the data ends first, at least the `needed` the caller uses.
pub(crate) fn decode_alpha(data: &[u8], bits: u32, count: usize, needed: usize) -> Result<Vec<u16>> {
    let mut r = BitReader::new(data);
    let mask = (1i32 << bits) - 1;
    let mut out = Vec::with_capacity(count);
    let mut prev: i32 = -1;
    while out.len() < count {
        if out.len() >= needed && r.end_of_data() {
            break;
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// §7.5.2's rounded rescaling, by division, for every value of every
    /// pair of depths the decoder and encoder use.
    #[test]
    fn rescale_is_exact() {
        for bits_in in 8..=16u32 {
            for bits_out in 8..=16u32 {
                let r = Rescale::new(bits_in, bits_out);
                let (max_in, max_out) = ((1u64 << bits_in) - 1, (1u64 << bits_out) - 1);
                for a in 0..=max_in {
                    let want = (2 * max_out * a + max_in) / (2 * max_in);
                    assert_eq!(r.apply(a as u16) as u64, want, "{bits_in} → {bits_out}: {a}");
                }
            }
        }
    }
}
