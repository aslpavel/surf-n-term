//! Error type
use rasterize::{ColorError, SvgParserError};
use std::{borrow::Cow, fmt};

#[derive(Debug)]
pub enum Error {
    Quit,
    IOError(std::io::Error),
    FmtError(std::fmt::Error),
    RustixError(rustix::io::Errno),
    NotATTY,
    ParseError(&'static str, String),
    FeatureNotSupported,
    Other(Cow<'static, str>),
    SvgParseError(SvgParserError),
    ColorError(ColorError),
    InvalidLayout,
    Json(serde_json::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        use Error::*;
        match self {
            Quit => None,
            IOError(error) => Some(error),
            FmtError(error) => Some(error),
            RustixError(error) => Some(error),
            SvgParseError(error) => Some(error),
            ColorError(error) => Some(error),
            NotATTY => None,
            ParseError(..) => None,
            FeatureNotSupported => None,
            Other(..) => None,
            InvalidLayout => None,
            Json(error) => Some(error),
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::IOError(error)
    }
}

impl From<std::fmt::Error> for Error {
    fn from(error: std::fmt::Error) -> Self {
        Self::FmtError(error)
    }
}

impl From<rustix::io::Errno> for Error {
    fn from(error: rustix::io::Errno) -> Self {
        Self::RustixError(error)
    }
}

impl From<SvgParserError> for Error {
    fn from(error: SvgParserError) -> Self {
        Self::SvgParseError(error)
    }
}

impl From<ColorError> for Error {
    fn from(error: ColorError) -> Self {
        Self::ColorError(error)
    }
}

impl From<serde_json::Error> for Error {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
