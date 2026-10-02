//! An Apple ProRes decoder and encoder, written from SMPTE RDD 36.

#![allow(dead_code)]

pub(crate) mod bits;
pub(crate) mod dct;
mod error;
pub(crate) mod tables;
pub(crate) mod vlc;

pub use error::{Error, Result};
