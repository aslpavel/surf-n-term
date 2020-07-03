use crate::{error::Error, Color, FaceAttrs, TerminalCommand};
use std::io::Write;

pub trait Encoder {
    type Item;
    type Error: From<std::io::Error>;

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
            DecModeGet(mode) => {
                write!(out, "\x1b[?{}$p", mode as usize)?;
            }
            CursorTo(pos) => write!(out, "\x1b[{};{}H", pos.row + 1, pos.col + 1)?,
            CursorGet => out.write_all(b"\x1b[6n")?,
            CursorSave => out.write_all(b"\x1b[s")?,
            CursorRestore => out.write_all(b"\x1b[u")?,
            EraseLineRight => out.write_all(b"\x1b[K")?,
            EraseLineLeft => out.write_all(b"\x1b[1K")?,
            EraseLine => out.write_all(b"\x1b[2K")?,
            EraseChars(count) => write!(out, "\x1b[{}X", count)?,
            Face(face) => {
                out.write_all(b"\x1b[00")?;
                if let Some(fg) = face.fg {
                    let [r, g, b] = fg.rgb_u8();
                    write!(out, ";38;2;{};{};{}", r, g, b)?;
                }
                if let Some(bg) = face.bg {
                    let [r, g, b] = bg.rgb_u8();
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
            Scroll(count) => {
                if count > 0 {
                    write!(out, "\x1b[{}S", count)?;
                } else if count < 0 {
                    write!(out, "\x1b[{}T", -count)?;
                }
            }
            Image(_) | ImageErase(_) => {
                // image is ignored and must be handled by image storage
            }
            Termcap(caps) => {
                write!(out, "\x1bP+q")?;
                for (index, cap) in caps.iter().enumerate() {
                    if index != 0 {
                        out.write_all(b";")?;
                    }
                    for b in cap.as_bytes() {
                        write!(out, "{:x}", b)?;
                    }
                }
                write!(out, "\x1b\\")?;
            }
        }

        Ok(())
    }
}

const BASE64: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encode input as base64
pub fn base64_encode(
    mut out: impl Write,
    input: impl IntoIterator<Item = u8>,
) -> Result<(), std::io::Error> {
    let mut iter = input.into_iter();
    loop {
        let mut dst = [b'='; 4];
        if let Some(s0) = iter.next() {
            dst[0] = BASE64[(s0 >> 2) as usize];
            if let Some(s1) = iter.next() {
                dst[1] = BASE64[((s0 << 4 | s1 >> 4) & 0x3f) as usize];
                if let Some(s2) = iter.next() {
                    dst[2] = BASE64[((s1 << 2 | s2 >> 6) & 0x3f) as usize];
                    dst[3] = BASE64[(s2 & 0x3f) as usize];
                } else {
                    dst[2] = BASE64[((s1 << 2) & 0x3f) as usize];
                }
            } else {
                dst[1] = BASE64[((s0 << 4) & 0x3f) as usize];
            }
            out.write_all(&dst)?;
        } else {
            break;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base64() -> Result<(), Error> {
        let mut base64 = Vec::new();

        base64_encode(&mut base64, b"term".iter().copied())?;
        assert_eq!(base64, b"dGVybQ==");

        base64.clear();
        base64_encode(&mut base64, b"ter".iter().copied())?;
        assert_eq!(base64, b"dGVy");

        base64.clear();
        base64_encode(&mut base64, b"ab".iter().copied())?;
        assert_eq!(base64, b"YWI=");

        Ok(())
    }
}
