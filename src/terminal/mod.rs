use crate::{Face, Surface};
use std::{fmt, io::Write};

mod common;
mod unix;

pub type SystemTerminal = unix::UnixTerminal;

/*
trait Terminal: Sync + Write {
    fn write(&mut self, command: TerminalCommand) -> Result<(), TerminalError>;
    fn flush(&mut self) -> Result<(), TerminalError>;

    fn recv<F, FR>(&mut self, f: F) -> FR
    where
        F: FnMut(&mut Vec<TerminalEvent>) -> FR;
}


 */
pub type BoxTerminal = std::boxed::Box<dyn Terminal>;

pub trait Terminal: Write {
    fn execute(&mut self, cmd: TerminalCommand) -> Result<(), TerminalError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TerminalCommand {
    /// Enable/disable wrapping of the text when it reaches end of the line
    AutoWrap(bool),
    /// Enable/Disable alternative screen
    AltScreen(bool),
    /// Report mouse events, if motion is ture it will also report mouse movements
    MouseSupport { enable: bool, motion: bool },
    /// Hide/Show cursor
    CursorVisible(bool),
    /// Request current cursor postion
    CursorReport,
    /// Move cursor to specified row and column
    CursorTo { row: usize, col: usize },
    /// Save current cursor position
    CursorSave,
    /// Restore previously saved cursor position
    CursorRestore,
    /// Erase line using current background color to the left of the cursor
    EraseLineLeft,
    /// Erase line using current background color to the right of the cursor
    EraseLineRight,
    /// Erase line using current background color
    EraseLine,
    /// Set current face (foreground/background colors and text attributes)
    Face(Face),
}

pub enum TerminalKeyMode {
    Alt,
    Ctrl,
    Shift,
}

pub enum TerminalEvent {}

pub trait Renderer {
    fn render(&mut self, surface: &Surface) -> Result<(), TerminalError>;
}

#[derive(Debug)]
pub enum TerminalError {
    IOError(std::io::Error),
    NixError(nix::Error),
    Closed,
    NotATTY,
}

impl fmt::Display for TerminalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for TerminalError {}

impl From<std::io::Error> for TerminalError {
    fn from(error: std::io::Error) -> Self {
        Self::IOError(error)
    }
}

impl From<nix::Error> for TerminalError {
    fn from(error: nix::Error) -> Self {
        Self::NixError(error)
    }
}
