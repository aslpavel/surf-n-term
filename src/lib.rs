#![allow(clippy::type_complexity)]

pub mod automata;
mod common;
pub mod decoder;
pub mod encoder;
pub mod error;
pub mod render;
mod style;
mod surface;
pub mod terminal;
mod unix;

pub use error::Error;
pub use render::{Cell, TerminalSurface, TerminalSurfaceExt};
pub use style::{Color, Face, FaceAttrs};
pub use surface::{
    Shape, Surface, SurfaceIter, SurfaceMut, SurfaceMutIter, SurfaceMutView, SurfaceOwned,
    SurfaceOwnedView, SurfaceView,
};
pub use terminal::{
    DecMode, DecModeStatus, Key, KeyMod, KeyName, Position, Terminal, TerminalAction,
    TerminalCommand, TerminalEvent,
};

pub type SystemTerminal = unix::UnixTerminal;
