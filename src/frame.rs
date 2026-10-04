//! The picture type the decoder hands back and the encoder takes: planar
//! Y′CbCr (and alpha), `u16` samples, the shape of `h26x::Picture`.

use crate::error::{Result, config};

/// Chroma sampling of a ProRes frame (`chroma_format`, RDD 36 Table 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChromaFormat {
    /// 4:2:2 (`chroma_format` 2): chroma at half the luma width, full height.
    Yuv422,
    /// 4:4:4 (`chroma_format` 3).
    Yuv444,
}

impl ChromaFormat {
    /// `(SubWidthC, SubHeightC)` — how many luma samples one chroma sample
    /// covers in each direction.
    pub fn subsampling(self) -> (u32, u32) {
        match self {
            ChromaFormat::Yuv422 => (2, 1),
            ChromaFormat::Yuv444 => (1, 1),
        }
    }

    /// From the frame header's `chroma_format` code (0 and 1 are reserved).
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            2 => Some(ChromaFormat::Yuv422),
            3 => Some(ChromaFormat::Yuv444),
            _ => None,
        }
    }

    /// The frame header's `chroma_format` code.
    pub fn code(self) -> u8 {
        match self {
            ChromaFormat::Yuv422 => 2,
            ChromaFormat::Yuv444 => 3,
        }
    }
}

/// Progressive or interlaced, and the field order (`interlace_mode`, RDD 36
/// Table 2). An interlaced frame is still handed over woven: the top field
/// on the even rows, the bottom field on the odd ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Interlace {
    /// One full-height picture (`interlace_mode` 0).
    #[default]
    Progressive,
    /// Two field pictures, the top field first in time and in the bitstream
    /// (`interlace_mode` 1).
    TopFieldFirst,
    /// Two field pictures, the bottom field first (`interlace_mode` 2).
    BottomFieldFirst,
}

impl Interlace {
    /// From `interlace_mode` (3 is reserved).
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Interlace::Progressive),
            1 => Some(Interlace::TopFieldFirst),
            2 => Some(Interlace::BottomFieldFirst),
            _ => None,
        }
    }

    /// `interlace_mode`.
    pub fn code(self) -> u8 {
        match self {
            Interlace::Progressive => 0,
            Interlace::TopFieldFirst => 1,
            Interlace::BottomFieldFirst => 2,
        }
    }
}

/// The frame header's descriptive fields, carried through decoding and
/// encoding unchanged. The colour codes are those of ITU-T H.273 (RDD 36
/// Tables 5 and 6 and the `transfer_characteristic` semantics agree with
/// it): 0 and 2 mean unspecified; 1 is BT.709, 9 BT.2020, 16 PQ, 18 HLG.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Metadata {
    /// `aspect_ratio_information` (Table 3): 0 unknown, 1 square pixels,
    /// 2 a 4:3 picture, 3 a 16:9 picture.
    pub aspect_ratio: u8,
    /// `frame_rate_code` (Table 4); see [`Metadata::frame_rate`].
    pub frame_rate_code: u8,
    /// `color_primaries` (Table 5).
    pub color_primaries: u8,
    /// `transfer_characteristic`.
    pub transfer_characteristic: u8,
    /// `matrix_coefficients` (Table 6).
    pub matrix_coefficients: u8,
}

impl Default for Metadata {
    /// Everything unspecified (the codes a caller with no information
    /// writes).
    fn default() -> Self {
        Metadata {
            aspect_ratio: 0,
            frame_rate_code: 0,
            color_primaries: 2,
            transfer_characteristic: 2,
            matrix_coefficients: 2,
        }
    }
}

impl Metadata {
    /// The frame rate `frame_rate_code` names, as `(numerator,
    /// denominator)` frames per second, or `None` for unknown or reserved.
    pub fn frame_rate(&self) -> Option<(u32, u32)> {
        Some(match self.frame_rate_code {
            1 => (24000, 1001),
            2 => (24, 1),
            3 => (25, 1),
            4 => (30000, 1001),
            5 => (30, 1),
            6 => (50, 1),
            7 => (60000, 1001),
            8 => (60, 1),
            9 => (100, 1),
            10 => (120000, 1001),
            11 => (120, 1),
            _ => return None,
        })
    }

    /// The `frame_rate_code` for a rate, 0 (unknown) if Table 4 has none.
    pub fn frame_rate_code_for(numerator: u32, denominator: u32) -> u8 {
        (1..=11)
            .find(|&c| {
                let m = Metadata { frame_rate_code: c, ..Metadata::default() };
                m.frame_rate().is_some_and(|(n, d)| n as u64 * denominator as u64 == numerator as u64 * d as u64)
            })
            .unwrap_or(0)
    }
}

/// One plane of a [`Frame`]: where it sits in [`Frame::data`]. Samples are
/// tightly packed (stride == width).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plane {
    /// Offset of the plane's first sample in [`Frame::data`], in samples.
    pub offset: usize,
    /// Width in samples.
    pub width: u32,
    /// Height in samples.
    pub height: u32,
}

impl Plane {
    /// Samples in the plane.
    pub fn len(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// Whether the plane has no samples.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A picture: one buffer holding the planes one after the other (Y′, then
/// Cb, then Cr — the layout of a packed planar frame, `yuv422p10le` and
/// friends once written little-endian), plus an optional alpha plane.
///
/// Samples are `u16` with the value in the low [`Frame::bit_depth`] bits.
/// The decoder produces 10-bit 4:2:2 and 12-bit 4:4:4 unless asked for
/// another depth; the encoder takes any depth from 8 to 16.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    /// Luma width.
    pub width: u32,
    /// Luma height (of the whole frame, both fields).
    pub height: u32,
    /// Bits per sample, every plane alike (8–16).
    pub bit_depth: u32,
    /// Chroma sampling.
    pub chroma: ChromaFormat,
    /// Progressive, or interlaced and the field order.
    pub interlace: Interlace,
    /// The frame header's descriptive fields.
    pub metadata: Metadata,
    /// The samples of every colour plane, packed.
    pub data: Vec<u16>,
    /// Y′, then Cb, then Cr.
    pub planes: [Plane; 3],
    /// The alpha plane, `width × height` samples at [`Frame::bit_depth`]
    /// bits (0 transparent, the maximum opaque), when there is one. A
    /// decoded frame's alpha has been converted from the stream's 8 or 16
    /// bits as RDD 36 §7.5.2 says; [`crate::decode::Decoder::with_bit_depth`]
    /// at 16 keeps 16-bit alpha exact.
    pub alpha: Option<Vec<u16>>,
}

impl Frame {
    /// A frame of mid-grey, every sample `2^(bit_depth - 1)`, no alpha,
    /// progressive, metadata unspecified. Fill its planes with
    /// [`Frame::plane_mut`].
    pub fn new(width: u32, height: u32, chroma: ChromaFormat, bit_depth: u32) -> Result<Frame> {
        let mut f = Frame::zeroed(width, height, chroma, bit_depth)?;
        f.data.fill(1 << (bit_depth - 1));
        Ok(f)
    }

    /// [`Frame::new`] with every sample 0, for the decoder, which writes
    /// every sample: zeroed memory comes from the allocator untouched,
    /// which writing grey to it does not.
    pub(crate) fn zeroed(width: u32, height: u32, chroma: ChromaFormat, bit_depth: u32) -> Result<Frame> {
        if width == 0 || height == 0 || width > 65535 || height > 65535 {
            return Err(config(format!("a frame of {width}×{height} cannot be coded (1–65535 each way)")));
        }
        if !(8..=16).contains(&bit_depth) {
            return Err(config(format!("bit depth {bit_depth} is outside 8–16")));
        }
        let planes = Self::layout(width, height, chroma);
        let total = planes[2].offset + planes[2].len();
        Ok(Frame {
            width,
            height,
            bit_depth,
            chroma,
            interlace: Interlace::Progressive,
            metadata: Metadata::default(),
            data: vec![0; total],
            planes,
            alpha: None,
        })
    }

    pub(crate) fn layout(width: u32, height: u32, chroma: ChromaFormat) -> [Plane; 3] {
        let (sw, _) = chroma.subsampling();
        let cw = width.div_ceil(sw);
        let luma = Plane { offset: 0, width, height };
        let cb = Plane { offset: luma.len(), width: cw, height };
        let cr = Plane { offset: cb.offset + cb.len(), width: cw, height };
        [luma, cb, cr]
    }

    /// The samples of plane `i` (0 Y′, 1 Cb, 2 Cr).
    pub fn plane(&self, i: usize) -> &[u16] {
        let p = &self.planes[i];
        &self.data[p.offset..p.offset + p.len()]
    }

    /// The samples of plane `i`, writable.
    pub fn plane_mut(&mut self, i: usize) -> &mut [u16] {
        let p = self.planes[i];
        &mut self.data[p.offset..p.offset + p.len()]
    }

    /// The planes concatenated: Y′ then Cb then Cr.
    pub fn packed(&self) -> &[u16] {
        &self.data
    }

    /// The packed planes as little-endian bytes — the memory layout of
    /// `yuv422p10le`, `yuv444p12le` and the like.
    pub fn to_le_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.data.len() * 2);
        for s in &self.data {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out
    }

    /// Checks that the planes and alpha are the sizes the header fields
    /// say, so the encoder can index without bounds surprises.
    pub(crate) fn validate(&self) -> Result<()> {
        if self.width == 0 || self.height == 0 || self.width > 65535 || self.height > 65535 {
            return Err(config(format!("a frame of {}×{} cannot be coded", self.width, self.height)));
        }
        if !(8..=16).contains(&self.bit_depth) {
            return Err(config(format!("bit depth {} is outside 8–16", self.bit_depth)));
        }
        let want = Self::layout(self.width, self.height, self.chroma);
        for (i, (p, w)) in self.planes.iter().zip(want.iter()).enumerate() {
            if p.width != w.width || p.height != w.height {
                return Err(config(format!(
                    "plane {i} is {}×{}; a {:?} frame of {}×{} needs {}×{}",
                    p.width, p.height, self.chroma, self.width, self.height, w.width, w.height
                )));
            }
            if p.offset.checked_add(p.len()).is_none_or(|end| end > self.data.len()) {
                return Err(config(format!("plane {i} runs past the end of the frame's data")));
            }
        }
        if let Some(a) = &self.alpha
            && a.len() != self.width as usize * self.height as usize
        {
            return Err(config(format!("the alpha plane has {} samples, not {}×{}", a.len(), self.width, self.height)));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_rounds_chroma_width_up() {
        let f = Frame::new(15, 3, ChromaFormat::Yuv422, 10).unwrap();
        assert_eq!(f.planes[1].width, 8);
        assert_eq!(f.planes[2].offset, 45 + 24);
        assert_eq!(f.data.len(), 45 + 48);
        assert!(f.validate().is_ok());
    }

    #[test]
    fn frame_rate_codes_round_trip() {
        for c in 1..=11u8 {
            let m = Metadata { frame_rate_code: c, ..Metadata::default() };
            let (n, d) = m.frame_rate().unwrap();
            assert_eq!(Metadata::frame_rate_code_for(n, d), c);
        }
        assert_eq!(Metadata::frame_rate_code_for(48, 2), 2);
        assert_eq!(Metadata::frame_rate_code_for(48, 1), 0);
    }
}
