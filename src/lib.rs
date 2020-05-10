#![allow(clippy::type_complexity)]

mod cell;
pub use crate::cell::{Color, Face, FaceAttrs};

mod surface;
pub use crate::surface::{Surface, View, ViewMut};

pub mod terminal;
pub use crate::terminal::{
    DecMode, DecModeStatus, Decoder, Key, KeyMod, KeyName, Renderer, SystemTerminal, TTYDecoder,
    Terminal, TerminalCommand, TerminalEvent, DFA, NFA,
};
