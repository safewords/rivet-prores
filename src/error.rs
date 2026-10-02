//! The crate's one error type.

/// What went wrong. Every malformed input comes back as one of these; the
/// decoder never panics on bytes it is given.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The bitstream breaks the syntax or the semantics of RDD 36: a size
    /// that runs past the end of the frame, a reserved code, a codeword that
    /// overruns its component's data or the coefficient array.
    #[error("invalid ProRes data: {0}")]
    Invalid(String),
    /// Valid-looking ProRes this crate does not implement, named: a
    /// bitstream version above 1 (RDD 36 says a decoder shall refuse those).
    #[error("unsupported ProRes feature: {0}")]
    Unsupported(String),
    /// A configuration or input the caller handed the encoder that cannot be
    /// coded: a frame whose format does not match the profile, planes of the
    /// wrong size, a quantisation matrix entry out of range.
    #[error("invalid ProRes configuration: {0}")]
    Config(String),
}

#[cold]
#[inline(never)]
pub(crate) fn invalid(msg: impl Into<String>) -> Error {
    Error::Invalid(msg.into())
}

#[cold]
#[inline(never)]
pub(crate) fn unsupported(msg: impl Into<String>) -> Error {
    Error::Unsupported(msg.into())
}

#[cold]
#[inline(never)]
pub(crate) fn config(msg: impl Into<String>) -> Error {
    Error::Config(msg.into())
}

/// `Result` with this crate's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
