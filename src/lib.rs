//! An Apple ProRes decoder and encoder, written from SMPTE RDD 36.

#![allow(dead_code)]

pub(crate) mod bits;
pub(crate) mod dct;
pub mod decode;
mod error;
pub mod frame;
pub mod header;
pub(crate) mod tables;
pub(crate) mod vlc;

pub use decode::Decoder;
pub use error::{Error, Result};
pub use frame::{ChromaFormat, Frame, Interlace, Metadata, Plane};
pub use header::{AlphaType, FrameHeader};
