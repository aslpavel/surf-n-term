#![allow(clippy::type_complexity)]

pub mod automata;
mod cell;
mod common;
pub mod decoder;
pub mod encoder;
pub mod error;
mod surface;
pub mod terminal;
mod unix;

pub use cell::{Cell, Color, Face, FaceAttrs};
pub use surface::{Surface, View, ViewMut};
pub use terminal::{
    DecMode, DecModeStatus, Key, KeyMod, KeyName, Position, Renderer, Terminal, TerminalCommand,
    TerminalEvent,
};

pub type SystemTerminal = unix::UnixTerminal;
