#![allow(clippy::type_complexity)]

mod cell;
pub use cell::{Color, Face, FaceAttrs};

mod surface;
pub use surface::{Surface, View, ViewMut};

pub mod terminal;
pub use terminal::{
    DecMode, DecModeStatus, Key, KeyMod, KeyName, Renderer, Terminal, TerminalCommand,
    TerminalEvent,
};

pub mod automata;
pub mod decoder;

mod unix;
pub type SystemTerminal = unix::UnixTerminal;

mod common;
