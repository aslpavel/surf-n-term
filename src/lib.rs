//! This crate is used to interact with Posix terminal. It can be used to
//!  - Read events from the terminal
//!  - Send commands to the terminal
//!  - Render on a surface which will be reconciled with current content of the terminal
//!  - Issue direct commends to the terminal
//!  - Supports kitty/sixel image protocol
//!
//! ### Simple example
//! ```no_run
//! use surf_n_term::{Terminal, TerminalEvent, TerminalAction, SystemTerminal, Error};
//!
//! fn main() -> Result<(), Error> {
//!     let ctrl_c = TerminalEvent::Key("ctrl+c".parse()?);
//!     let mut term = SystemTerminal::new()?;
//!     term.run_render(|term, event, mut view| -> Result<_, Error> {
//!         // This function will be executed on each event from terminal
//!         // - term  - implements Terminal trait
//!         // - event - is a TerminalEvent
//!         // - view  - is a Surface that can be used to render on, see render module for details
//!         match event {
//!             Some(event) if &event == &ctrl_c => {
//!                 // exit if 'ctrl+c' is pressed
//!                 Ok(TerminalAction::Quit(()))
//!             }
//!             _ => {
//!                 // do some rendering by updating the view
//!                 Ok(TerminalAction::Wait)
//!             },
//!         }
//!     })?;
//!     Ok(())
//! }
//! ```
#![allow(clippy::type_complexity)]
#![allow(clippy::reversed_empty_ranges)]
#![allow(clippy::excessive_precision)]
#![deny(warnings)]

pub mod automata;
pub mod common;
pub mod decoder;
pub mod encoder;
pub mod error;
pub mod face;
pub mod glyph;
pub mod image;
pub mod keys;
pub mod render;
pub mod surface;
pub mod terminal;
mod unix;
pub mod view;

pub use error::Error;
pub use face::{Face, FaceAttrs};
pub use glyph::{BBox, FillRule, Glyph, Path};
pub use image::{ColorPalette, Image, ImageHandler, KittyImageHandler, SixelImageHandler};
pub use keys::{Key, KeyMap, KeyMod, KeyName};
pub use rasterize;
pub use rasterize::{Color, LinColor, RGBA};
pub use render::{Cell, TerminalSurface, TerminalSurfaceExt, TerminalWriter};
pub use surface::{
    Shape, Surface, SurfaceIter, SurfaceMut, SurfaceMutIter, SurfaceMutView, SurfaceOwned,
    SurfaceOwnedView, SurfaceView,
};
pub use terminal::{
    DecMode, DecModeStatus, Position, Size, Terminal, TerminalAction, TerminalCaps, TerminalColor,
    TerminalCommand, TerminalEvent, TerminalSize, TerminalWaker,
};

/// System specific terminal
pub type SystemTerminal = unix::UnixTerminal;
