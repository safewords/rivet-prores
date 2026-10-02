//! The frame header (RDD 36 §5.1.1, §6.1.1), the picture header (§5.2.1,
//! §6.2.1) and the slice layout they imply (§6.2).

use crate::error::{Result, invalid, unsupported};
use crate::frame::{ChromaFormat, Interlace, Metadata};

/// The four-character code that follows `frame_size` (§6.1).
pub const FRAME_IDENTIFIER: [u8; 4] = *b"icpf";

/// `alpha_channel_type` (Table 7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AlphaType {
    /// No alpha in the bitstream (0).
    #[default]
    None,
    /// 8 bits per alpha sample (1).
    Bits8,
    /// 16 bits per alpha sample (2).
    Bits16,
}

impl AlphaType {
    /// The header code.
    pub fn code(self) -> u8 {
        match self {
            AlphaType::None => 0,
            AlphaType::Bits8 => 1,
            AlphaType::Bits16 => 2,
        }
    }

    /// Bits per coded alpha sample (0 for none).
    pub fn bits(self) -> u32 {
        match self {
            AlphaType::None => 0,
            AlphaType::Bits8 => 8,
            AlphaType::Bits16 => 16,
        }
    }
}

/// A parsed `frame()` header: everything up to the first picture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameHeader {
    /// `frame_size`: the whole frame in bytes, from `frame_size` itself to
    /// the end of any stuffing.
    pub frame_size: u32,
    /// `frame_header_size` in bytes; the first picture starts this far
    /// after the 8 bytes of `frame_size` and `frame_identifier`.
    pub header_size: u16,
    /// `bitstream_version` (0 or 1; this crate refuses higher).
    pub bitstream_version: u8,
    /// `encoder_identifier`, the writer's registered four-character code.
    pub encoder_identifier: [u8; 4],
    /// `horizontal_size`: luma width.
    pub width: u16,
    /// `vertical_size`: luma height of the frame.
    pub height: u16,
    /// `chroma_format`.
    pub chroma: ChromaFormat,
    /// `interlace_mode`.
    pub interlace: Interlace,
    /// The descriptive fields.
    pub metadata: Metadata,
    /// `alpha_channel_type`.
    pub alpha: AlphaType,
    /// `luma_quantization_matrix` in raster order (`8 * v + u`), when
    /// loaded.
    pub luma_matrix: Option<[u8; 64]>,
    /// `chroma_quantization_matrix`, when loaded.
    pub chroma_matrix: Option<[u8; 64]>,
}

/// Bytes of the frame header before the optional matrices.
pub(crate) const FRAME_HEADER_FIXED: usize = 20;

impl FrameHeader {
    /// Parses the start of a frame: `frame_size`, `frame_identifier` and
    /// `frame_header()`. Checks that `frame_size` fits in `data` (a MOV
    /// sample may carry trailing bytes; they are ignored) and that the
    /// header fits in the frame.
    pub fn parse(data: &[u8]) -> Result<FrameHeader> {
        if data.len() < 8 + FRAME_HEADER_FIXED {
            return Err(invalid(format!("{} bytes is too short for a ProRes frame header", data.len())));
        }
        let frame_size = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        if data[4..8] != FRAME_IDENTIFIER {
            return Err(invalid("no 'icpf' frame identifier"));
        }
        if (frame_size as usize) > data.len() {
            return Err(invalid(format!("frame_size {frame_size} exceeds the {} bytes given", data.len())));
        }
        let h = &data[8..frame_size as usize];
        if h.len() < FRAME_HEADER_FIXED {
            return Err(invalid("frame_size leaves no room for the frame header"));
        }
        let header_size = u16::from_be_bytes([h[0], h[1]]);
        if (header_size as usize) < FRAME_HEADER_FIXED || header_size as usize > h.len() {
            return Err(invalid(format!("frame_header_size {header_size} is out of range")));
        }
        let h = &h[..header_size as usize];
        let bitstream_version = h[3];
        if bitstream_version > 1 {
            return Err(unsupported(format!("bitstream_version {bitstream_version}")));
        }
        let encoder_identifier = [h[4], h[5], h[6], h[7]];
        let width = u16::from_be_bytes([h[8], h[9]]);
        let height = u16::from_be_bytes([h[10], h[11]]);
        if width == 0 || height == 0 {
            return Err(invalid(format!("a {width}×{height} frame")));
        }
        let chroma = ChromaFormat::from_code(h[12] >> 6)
            .ok_or_else(|| invalid(format!("chroma_format {} is reserved", h[12] >> 6)))?;
        let interlace = Interlace::from_code((h[12] >> 2) & 3).ok_or_else(|| invalid("interlace_mode 3 is reserved"))?;
        let metadata = Metadata {
            aspect_ratio: h[13] >> 4,
            frame_rate_code: h[13] & 15,
            color_primaries: h[14],
            transfer_characteristic: h[15],
            matrix_coefficients: h[16],
        };
        let alpha = match h[17] & 15 {
            0 => AlphaType::None,
            1 => AlphaType::Bits8,
            2 => AlphaType::Bits16,
            n => return Err(invalid(format!("alpha_channel_type {n} is reserved"))),
        };
        let load_luma = h[19] & 2 != 0;
        let load_chroma = h[19] & 1 != 0;
        let mut pos = FRAME_HEADER_FIXED;
        let mut matrix = |load: bool| -> Result<Option<[u8; 64]>> {
            if !load {
                return Ok(None);
            }
            let m: [u8; 64] = h
                .get(pos..pos + 64)
                .ok_or_else(|| invalid("a quantisation matrix runs past frame_header_size"))?
                .try_into()
                .unwrap();
            pos += 64;
            Ok(Some(m))
        };
        let luma_matrix = matrix(load_luma)?;
        let chroma_matrix = matrix(load_chroma)?;
        Ok(FrameHeader {
            frame_size,
            header_size,
            bitstream_version,
            encoder_identifier,
            width,
            height,
            chroma,
            interlace,
            metadata,
            alpha,
            luma_matrix,
            chroma_matrix,
        })
    }

    /// Appends `frame_size` (as 0, patched by the caller), `'icpf'` and the
    /// frame header. `header_size` is computed, not taken from `self`.
    pub(crate) fn write(&self, out: &mut Vec<u8>) {
        let size = FRAME_HEADER_FIXED
            + 64 * (self.luma_matrix.is_some() as usize + self.chroma_matrix.is_some() as usize);
        out.extend_from_slice(&0u32.to_be_bytes());
        out.extend_from_slice(&FRAME_IDENTIFIER);
        out.extend_from_slice(&(size as u16).to_be_bytes());
        out.push(0);
        out.push(self.bitstream_version);
        out.extend_from_slice(&self.encoder_identifier);
        out.extend_from_slice(&self.width.to_be_bytes());
        out.extend_from_slice(&self.height.to_be_bytes());
        out.push((self.chroma.code() << 6) | (self.interlace.code() << 2));
        out.push((self.metadata.aspect_ratio << 4) | (self.metadata.frame_rate_code & 15));
        out.push(self.metadata.color_primaries);
        out.push(self.metadata.transfer_characteristic);
        out.push(self.metadata.matrix_coefficients);
        out.push(self.alpha.code());
        out.push(0);
        out.push(((self.luma_matrix.is_some() as u8) << 1) | self.chroma_matrix.is_some() as u8);
        if let Some(m) = &self.luma_matrix {
            out.extend_from_slice(m);
        }
        if let Some(m) = &self.chroma_matrix {
            out.extend_from_slice(m);
        }
    }

    /// The luma weights in force (§7.3): the loaded matrix or all 4s.
    pub(crate) fn luma_weights(&self) -> [u8; 64] {
        self.luma_matrix.unwrap_or(crate::tables::DEFAULT_MATRIX)
    }

    /// The chroma weights in force: the loaded chroma matrix, else the luma
    /// weights.
    pub(crate) fn chroma_weights(&self) -> [u8; 64] {
        self.chroma_matrix.unwrap_or_else(|| self.luma_weights())
    }

    /// Pictures in the frame: 1 progressive, 2 interlaced.
    pub fn picture_count(&self) -> usize {
        if self.interlace == Interlace::Progressive { 1 } else { 2 }
    }

    /// `picture_vertical_size` of picture `index` (0 first, 1 second, §6.2).
    pub fn picture_height(&self, index: usize) -> u32 {
        let v = self.height as u32;
        match self.interlace {
            Interlace::Progressive => v,
            mode => {
                if self.is_top_field(index, mode) { v.div_ceil(2) } else { v / 2 }
            }
        }
    }

    fn is_top_field(&self, index: usize, mode: Interlace) -> bool {
        (mode == Interlace::TopFieldFirst) == (index == 0)
    }

    /// For picture `index`: the frame row of its row 0 and the step between
    /// its rows (§7.5.3) — `(0, 1)` progressive, `(0, 2)` the top field,
    /// `(1, 2)` the bottom one.
    pub(crate) fn picture_rows(&self, index: usize) -> (usize, usize) {
        match self.interlace {
            Interlace::Progressive => (0, 1),
            mode => {
                if self.is_top_field(index, mode) { (0, 2) } else { (1, 2) }
            }
        }
    }
}

/// `picture_header()` (§5.2.1), 8 bytes as this crate writes it.
pub(crate) const PICTURE_HEADER_SIZE: usize = 8;

/// The parsed fields of a `picture_header()` this crate uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PictureHeader {
    pub(crate) header_size: usize,
    pub(crate) picture_size: usize,
    pub(crate) log2_slice_mbs: u32,
}

impl PictureHeader {
    pub(crate) fn parse(data: &[u8]) -> Result<PictureHeader> {
        if data.len() < PICTURE_HEADER_SIZE {
            return Err(invalid("a picture header runs past the end of the frame"));
        }
        let header_size = (data[0] >> 3) as usize;
        let picture_size = u32::from_be_bytes([data[1], data[2], data[3], data[4]]) as usize;
        if header_size < PICTURE_HEADER_SIZE || header_size > data.len() {
            return Err(invalid(format!("picture_header_size {header_size} is out of range")));
        }
        if picture_size < header_size || picture_size > data.len() {
            return Err(invalid(format!("picture_size {picture_size} is out of range")));
        }
        Ok(PictureHeader { header_size, picture_size, log2_slice_mbs: ((data[7] >> 4) & 3) as u32 })
    }

    pub(crate) fn write(out: &mut Vec<u8>, picture_size: u32, slice_count: usize, log2_slice_mbs: u32) {
        out.push((PICTURE_HEADER_SIZE as u8) << 3);
        out.extend_from_slice(&picture_size.to_be_bytes());
        // deprecated_number_of_slices: the count when it fits, else 0.
        out.extend_from_slice(&u16::try_from(slice_count).unwrap_or(0).to_be_bytes());
        out.push((log2_slice_mbs as u8 & 3) << 4);
    }
}

/// `slice_size_in_mb[]` for a picture `width_in_mb` macroblocks wide
/// (§6.2): slices of the desired size from the left, then halving.
pub(crate) fn slice_sizes(width_in_mb: u32, log2_slice_mbs: u32) -> Vec<u32> {
    let mut sizes = Vec::new();
    let mut size = 1u32 << log2_slice_mbs;
    let mut remaining = width_in_mb;
    while remaining > 0 {
        while remaining >= size {
            sizes.push(size);
            remaining -= size;
        }
        size /= 2;
    }
    sizes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_layout_is_the_example_of_section_4() {
        // 720 wide, 8-macroblock slices: 8, 8, 8, 8, 8, 4, 1 (Figure 1).
        assert_eq!(slice_sizes(45, 3), vec![8, 8, 8, 8, 8, 4, 1]);
        assert_eq!(slice_sizes(1, 3), vec![1]);
        assert_eq!(slice_sizes(7, 3), vec![4, 2, 1]);
        assert_eq!(slice_sizes(120, 3).len(), 15);
        assert_eq!(slice_sizes(5, 0), vec![1; 5]);
    }

    #[test]
    fn field_heights_split_an_odd_frame_height() {
        let h = |interlace, height| FrameHeader {
            frame_size: 0,
            header_size: 20,
            bitstream_version: 0,
            encoder_identifier: *b"test",
            width: 720,
            height,
            chroma: ChromaFormat::Yuv422,
            interlace,
            metadata: Metadata::default(),
            alpha: AlphaType::None,
            luma_matrix: None,
            chroma_matrix: None,
        };
        let tff = h(Interlace::TopFieldFirst, 487);
        assert_eq!((tff.picture_height(0), tff.picture_height(1)), (244, 243));
        assert_eq!((tff.picture_rows(0), tff.picture_rows(1)), ((0, 2), (1, 2)));
        let bff = h(Interlace::BottomFieldFirst, 487);
        assert_eq!((bff.picture_height(0), bff.picture_height(1)), (243, 244));
        assert_eq!((bff.picture_rows(0), bff.picture_rows(1)), ((1, 2), (0, 2)));
        let p = h(Interlace::Progressive, 486);
        assert_eq!(p.picture_count(), 1);
        assert_eq!(p.picture_height(0), 486);
    }
}
