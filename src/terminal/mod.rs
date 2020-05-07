use crate::{Face, Surface};
use std::{fmt, io::Write};

pub mod automata;
pub mod common;
pub mod parser;
pub mod unix;

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

pub trait Renderer {
    fn render(&mut self, surface: &Surface) -> Result<(), TerminalError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TerminalCommand {
    /// Enable/disable wrapping of the text when it reaches end of the line
    AutoWrap(bool),
    /// Enable/Disable alternative screen
    AltScreen(bool),
    /// Report mouse events, if motion is ture it will also report mouse movements
    MouseReport { enable: bool, motion: bool },
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
    /// Full reset of the terminal
    Reset,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TerminalEvent {
    Key(Key),
    CursorPosition { row: usize, col: usize },
    Resize(TerminalSize),
    Raw(Vec<u8>),
}

impl fmt::Debug for TerminalEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use TerminalEvent::*;
        match self {
            Key(key) => write!(f, "{:?}", key)?,
            Resize(size) => write!(f, "{:?}", size)?,
            CursorPosition { row, col } => write!(f, "Cursor({}, {})", row, col)?,
            Raw(raw) => write!(f, "Raw({:?})", String::from_utf8_lossy(raw))?,
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TerminalSize {
    pub width: usize,
    pub height: usize,
    pub width_pixels: usize,
    pub height_pixels: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key {
    name: KeyName,
    mode: KeyMod,
}

impl Key {
    pub fn new(name: KeyName, mode: KeyMod) -> Self {
        Self { name, mode }
    }
}

impl From<KeyName> for Key {
    fn from(name: KeyName) -> Self {
        Self {
            name,
            mode: KeyMod::EMPTY,
        }
    }
}

impl From<(KeyName, KeyMod)> for Key {
    fn from(pair: (KeyName, KeyMod)) -> Self {
        Self {
            name: pair.0,
            mode: pair.1,
        }
    }
}

impl fmt::Debug for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.mode.is_empty() {
            write!(f, "{:?}", self.name)?;
        } else {
            write!(f, "{:?}-{:?}", self.name, self.mode)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KeyName {
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    Esc,
    PageUp,
    PageDown,
    Home,
    End,
    Up,
    Down,
    Right,
    Left,
    Char(char),
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KeyMod {
    bits: u8,
}

impl KeyMod {
    pub const EMPTY: Self = KeyMod { bits: 0 };
    pub const ALT: Self = KeyMod { bits: 1 };
    pub const CTRL: Self = KeyMod { bits: 2 };
    pub const SHIFT: Self = KeyMod { bits: 4 };

    pub fn is_empty(&self) -> bool {
        self == &Self::EMPTY
    }

    pub fn contains(&self, other: Self) -> bool {
        self.bits & other.bits == other.bits
    }
}

impl std::ops::BitOr for KeyMod {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self {
            bits: self.bits | rhs.bits,
        }
    }
}

impl fmt::Debug for KeyMod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            write!(f, "None")?;
        } else {
            let mut first = true;
            for (flag, name) in &[
                (Self::ALT, "Alt"),
                (Self::CTRL, "Ctrl"),
                (Self::SHIFT, "Shift"),
            ] {
                if self.contains(*flag) {
                    if first {
                        first = false;
                        write!(f, "{}", name)?;
                    } else {
                        write!(f, "-{}", name)?;
                    }
                }
            }
        }
        Ok(())
    }
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
