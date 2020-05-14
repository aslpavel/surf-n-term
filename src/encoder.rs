use crate::{error::Error, FaceAttrs, TerminalCommand};
use std::io::Write;

pub trait Encoder {
    type Item;
    type Error;

    fn encode<W: Write>(&mut self, out: W, item: Self::Item) -> Result<(), Self::Error>;
}

#[derive(Debug)]
pub struct TTYEncoder;

impl Default for TTYEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl TTYEncoder {
    pub fn new() -> Self {
        TTYEncoder
    }
}

impl Encoder for TTYEncoder {
    type Item = TerminalCommand;
    type Error = Error;

    fn encode<W: Write>(&mut self, mut out: W, cmd: Self::Item) -> Result<(), Self::Error> {
        use TerminalCommand::*;

        match cmd {
            DecModeSet { enable, mode } => {
                let flag = if enable { "h" } else { "l" };
                write!(out, "\x1b[?{}{}", mode as usize, flag)?;
            }
            DecModeReport(mode) => {
                write!(out, "\x1b[?{}$p", mode as usize)?;
            }
            CursorTo(pos) => write!(out, "\x1b[{};{}H", pos.row + 1, pos.col + 1)?,
            CursorReport => out.write_all(b"\x1b[6n")?,
            CursorSave => out.write_all(b"\x1b[s")?,
            CursorRestore => out.write_all(b"\x1b[u")?,
            EraseLineRight => out.write_all(b"\x1b[K")?,
            EraseLineLeft => out.write_all(b"\x1b[1K")?,
            EraseLine => out.write_all(b"\x1b[2K")?,
            EraseChars(count) => write!(out, "\x1b[{}X", count)?,
            Face(face) => {
                out.write_all(b"\x1b[00")?;
                if let Some(fg) = face.fg {
                    let (r, g, b) = fg.rgb_u8();
                    write!(out, ";38;2;{};{};{}", r, g, b)?;
                }
                if let Some(bg) = face.bg {
                    let (r, g, b) = bg.rgb_u8();
                    write!(out, ";48;2;{};{};{}", r, g, b)?;
                }
                if !face.attrs.is_empty() {
                    for (flag, code) in &[
                        (FaceAttrs::BOLD, b";01"),
                        (FaceAttrs::ITALIC, b";03"),
                        (FaceAttrs::UNDERLINE, b";04"),
                        (FaceAttrs::BLINK, b";05"),
                        (FaceAttrs::REVERSE, b";07"),
                    ] {
                        if face.attrs.contains(*flag) {
                            out.write_all(*code)?;
                        }
                    }
                }
                out.write_all(b"m")?;
            }
            Reset => out.write_all(b"\x1bc")?,
            Char(c) => write!(out, "{}", c)?,
        }

        Ok(())
    }
}
