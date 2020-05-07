#![allow(clippy::type_complexity)]

mod cell;
pub use crate::cell::{Color, Face, FaceAttrs};

mod surface;
pub use crate::surface::{Surface, View};

pub mod terminal;
pub use crate::terminal::{
    Key, KeyMod, KeyName, Renderer, SystemTerminal, Terminal, TerminalCommand, TerminalEvent,
};
