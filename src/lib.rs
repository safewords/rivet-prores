//! An Apple ProRes decoder and encoder.
//!
//! Rust, no C, no system libraries, no build script. Written from SMPTE
//! RDD 36:2022, *Apple ProRes Bitstream Syntax and Decoding Process* — the
//! syntax of §5, the semantics of §6 and the decoding process of §7 — and
//! not translated from any other implementation.
//!
//! - [`decode::Decoder`] decodes every ProRes frame RDD 36 describes:
//!   bitstream versions 0 and 1, 4:2:2 and 4:4:4, progressive and
//!   interlaced, default and custom quantisation matrices, 8- and 16-bit
//!   alpha. That covers every profile, Proxy to 4444 XQ — the profiles
//!   share one syntax and differ in format and bit rate.
//! - [`encode::Encoder`] writes the 4:2:2 profiles (Proxy, LT, 422, HQ) and
//!   4444 / 4444 XQ with or without alpha, progressive or interlaced, with a
//!   per-slice quantiser chosen to meet the profile's target frame size.
//!
//! A frame is the bytes of one MOV sample: `frame_size`, `'icpf'`, the
//! frame header, one picture (two for interlaced) and any stuffing. The
//! container's sample entry code names the profile ([`Profile::fourcc`]);
//! the bitstream does not.
//!
//! ```
//! use prores::{ChromaFormat, Config, Decoder, Encoder, Frame, Profile};
//!
//! let mut frame = Frame::new(64, 48, ChromaFormat::Yuv422, 10)?;
//! for (i, s) in frame.plane_mut(0).iter_mut().enumerate() {
//!     *s = 64 + (i % 64) as u16 * 14;
//! }
//! let packet = Encoder::new(Config::new(Profile::Hq)).encode(&frame)?;
//! let decoded = Decoder::new().decode(&packet)?;
//! assert_eq!((decoded.width, decoded.height, decoded.bit_depth), (64, 48, 10));
//! # Ok::<(), prores::Error>(())
//! ```

#![warn(missing_docs)]

#[cfg(feature = "bench")]
#[doc(hidden)]
pub mod bench_api;
pub(crate) mod bits;
pub(crate) mod dct;
pub mod decode;
pub(crate) mod dsp;
pub mod encode;
mod error;
pub mod frame;
pub mod header;
pub(crate) mod pool;
pub(crate) mod tables;
pub(crate) mod vlc;

pub use decode::Decoder;
pub use encode::{Config, Encoder, Profile};
pub use error::{Error, Result};
pub use frame::{ChromaFormat, Frame, Interlace, Metadata, Plane};
pub use header::{AlphaType, FrameHeader};
