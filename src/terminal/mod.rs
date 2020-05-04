use crate::Surface;
use std::fmt;

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

type BoxTerminal = std::boxed::Box<dyn Terminal>;
 */

pub enum TerminalCommand {
    AutoWrap(bool),
    AltScreen(bool),
    MouseEvent(bool),
    MouseMotionEvent(bool),
    CursorShow(bool),
    CursorReport,
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
