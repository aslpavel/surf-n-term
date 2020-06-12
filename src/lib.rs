#![allow(clippy::type_complexity)]

pub mod automata;
pub mod color;
mod common;
pub mod decoder;
pub mod encoder;
pub mod error;
pub mod image;
pub mod render;
pub mod style;
pub mod terminal;
mod unix;

pub use color::{Blend, Color, ColorLinear, RGBA};
pub use error::Error;
pub use image::{ImageHandle, ImageStorage, KittyImageStorage};
pub use render::{Cell, TerminalSurface, TerminalSurfaceExt};
pub use style::{Face, FaceAttrs};
pub use surface::{
    Shape, Surface, SurfaceIter, SurfaceMut, SurfaceMutIter, SurfaceMutView, SurfaceOwned,
    SurfaceOwnedView, SurfaceView,
};
pub use terminal::{
    DecMode, DecModeStatus, Key, KeyMod, KeyName, Position, Terminal, TerminalAction,
    TerminalCommand, TerminalEvent,
};

pub type SystemTerminal = unix::UnixTerminal;
