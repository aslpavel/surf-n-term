//! Encoders
use crate::{
    decoder::KEYBOARD_LEVEL, error::Error, Color, ColorLinear, DecMode, FaceAttrs, TerminalCaps,
    TerminalColor, TerminalCommand,
};
use std::{cmp::Ordering, io::Write, str::FromStr};

/// Encoder interface
pub trait Encoder {
    type Item;
    type Error: From<std::io::Error>;

    /// Encode item and write result to Write object
    fn encode<W: Write>(&mut self, out: W, item: Self::Item) -> Result<(), Self::Error>;
}

/// TTY encoder
///
/// References:
/// - [XTerm Control Sequences](https://www.invisible-island.net/xterm/ctlseqs/ctlseqs.html)
/// - [ANSI Escape Code](https://en.wikipedia.org/wiki/ANSI_escape_code)
#[derive(Debug)]
pub struct TTYEncoder {
    caps: TerminalCaps,
}

impl Default for TTYEncoder {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl TTYEncoder {
    pub fn new(caps: TerminalCaps) -> Self {
        Self { caps }
    }

    fn kitty_level<W: Write>(&self, mut out: W, level: usize) -> Result<(), Error> {
        if self.caps.kitty_keyboard {
            write!(out, "\x1b[={}u", level)?;
        }
        Ok(())
    }
}

impl Encoder for TTYEncoder {
    type Item = TerminalCommand;
    type Error = Error;

    fn encode<W: Write>(&mut self, mut out: W, cmd: Self::Item) -> Result<(), Self::Error> {
        use TerminalCommand::*;

        match cmd {
            DecModeSet { enable, mode } => {
                // kitty keyboard level maintained separately for alt-screen
                if !enable && mode == DecMode::AltScreen {
                    self.kitty_level(&mut out, 0)?;
                }

                let flag = if enable { "h" } else { "l" };
                write!(out, "\x1b[?{}{}", mode as usize, flag)?;

                if enable && mode == DecMode::AltScreen {
                    self.kitty_level(out, KEYBOARD_LEVEL)?;
                }
            }
            DecModeGet(mode) => {
                write!(out, "\x1b[?{}$p", mode as usize)?;
            }
            CursorTo(pos) => write!(out, "\x1b[{};{}H", pos.row + 1, pos.col + 1)?,
            CursorMove { row, col } => {
                match col.cmp(&0) {
                    Ordering::Greater => write!(out, "\x1b[{}C", col)?,
                    Ordering::Less => write!(out, "\x1b[{}D", -col)?,
                    _ => {}
                }
                match row.cmp(&0) {
                    Ordering::Greater => write!(out, "\x1b[{}B", row)?,
                    Ordering::Less => write!(out, "\x1b[{}A", -row)?,
                    _ => {}
                }
            }
            CursorGet => out.write_all(b"\x1b[6n")?,
            CursorSave => out.write_all(b"\x1b7")?,
            CursorRestore => out.write_all(b"\x1b8")?,
            EraseLineRight => out.write_all(b"\x1b[K")?,
            EraseLineLeft => out.write_all(b"\x1b[1K")?,
            EraseLine => out.write_all(b"\x1b[2K")?,
            EraseChars(count) => write!(out, "\x1b[{}X", count)?,
            Face(face) => {
                out.write_all(b"\x1b[00")?;
                if let Some(fg) = face.fg {
                    color_sgr_encode(&mut out, fg, self.caps.depth, true)?;
                }
                if let Some(bg) = face.bg {
                    color_sgr_encode(&mut out, bg, self.caps.depth, false)?;
                }
                if !face.attrs.is_empty() {
                    for (flag, code) in &[
                        (FaceAttrs::BOLD, b";1"),
                        (FaceAttrs::ITALIC, b";3"),
                        (FaceAttrs::UNDERLINE, b";4"),
                        (FaceAttrs::BLINK, b";5"),
                        (FaceAttrs::REVERSE, b";7"),
                        (FaceAttrs::STRIKE, b";9"),
                    ] {
                        if face.attrs.contains(*flag) {
                            out.write_all(*code)?;
                        }
                    }
                }
                out.write_all(b"m")?;
            }
            FaceGet => {
                // DECRQSS - Request Selection or Setting with description set to `m`
                out.write_all(b"\x1bP$qm\x1b\\")?;
            }
            Reset => out.write_all(b"\x1bc")?,
            Char(c) => write!(out, "{}", c)?,
            Scroll(count) => match count.cmp(&0) {
                Ordering::Less => write!(out, "\x1b[{}T", -count)?,
                Ordering::Greater => write!(out, "\x1b[{}S", count)?,
                _ => (),
            },
            ScrollRegion { start, end } => {
                if end > start {
                    write!(out, "\x1b[{};{}r", start + 1, end + 1)?;
                } else {
                    write!(out, "\x1b[r")?;
                }
            }
            Image(_, _) | ImageErase(_, _) => {
                // image commands are ignored and must be handled by image handler
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
            Color { name, color } => {
                write!(out, "\x1b]")?;
                match name {
                    TerminalColor::Background => write!(out, "11;")?,
                    TerminalColor::Foreground => write!(out, "10;")?,
                    TerminalColor::Palette(index) => write!(out, "4;{};", index)?,
                }
                match color {
                    Some(color) => write!(out, "{}", color)?,
                    None => write!(out, "?")?,
                }
                write!(out, "\x1b\\")?;
            }
            Title(title) => {
                write!(out, "\x1b]0;{}\x1b\\", title)?;
            }
            DeviceAttrs => {
                write!(out, "\x1b[c")?;
            }
            KeyboardLevel(level) => {
                self.kitty_level(out, level)?;
            }
        }

        Ok(())
    }
}

const BASE64: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Writable object which encodes input to base64 and writes it in underlying stream
pub struct Base64Encoder<W> {
    inner: W,
    buffer: Vec<u8>,
}

impl<W: Write> Base64Encoder<W> {
    pub fn new(inner: W) -> Self {
        Self {
            inner,
            buffer: Vec::with_capacity(3),
        }
    }

    /// finalize base64 stream, returning underlying stream
    pub fn finish(self) -> std::io::Result<W> {
        let Self { mut inner, buffer } = self;
        let mut dst = [b'='; 4];
        let mut iter = buffer.into_iter();
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
            inner.write_all(&dst)?;
        }
        Ok(inner)
    }
}

impl<W: Write> Write for Base64Encoder<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        for b in buf.iter().copied() {
            self.buffer.push(b);
            if self.buffer.len() == 3 {
                match self.buffer.as_slice() {
                    [s0, s1, s2] => {
                        let mut dst = [b'='; 4];
                        dst[0] = BASE64[(s0 >> 2) as usize];
                        dst[1] = BASE64[((s0 << 4 | s1 >> 4) & 0x3f) as usize];
                        dst[2] = BASE64[((s1 << 2 | s2 >> 6) & 0x3f) as usize];
                        dst[3] = BASE64[(s2 & 0x3f) as usize];
                        self.inner.write_all(&dst)?;
                    }
                    _ => unreachable!(),
                }
                self.buffer.clear();
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Color depth
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ColorDepth {
    TrueColor,
    EightBit,
    Gray,
}

impl FromStr for ColorDepth {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use ColorDepth::*;
        match s.to_ascii_lowercase().as_str() {
            "truecolor" => Ok(TrueColor),
            "24" => Ok(TrueColor),
            "256" => Ok(EightBit),
            "8" => Ok(EightBit),
            "gray" => Ok(Gray),
            "2" => Ok(Gray),
            _ => Err(Error::ParseError(
                "ColorDepth",
                format!("invalid color depth: {}", s),
            )),
        }
    }
}

/// Color cube grid [0, 95, 135, 175, 215, 255] converted to linear RGB colors space.
const CUBE: &[f64] = &[0.0, 0.114435, 0.242281, 0.42869, 0.679542, 1.0];

/// Grey colors available in 256 color mode converted to linear RGB color space.
const GREYS: &[f64] = &[
    0.002428, 0.006049, 0.011612, 0.019382, 0.029557, 0.042311, 0.057805, 0.076185, 0.097587,
    0.122139, 0.14996, 0.181164, 0.215861, 0.254152, 0.296138, 0.341914, 0.391572, 0.445201,
    0.502886, 0.564712, 0.630757, 0.701102, 0.775822, 0.854993,
];

fn nearest(v: f64, vs: &[f64]) -> usize {
    match vs.binary_search_by(|c| c.partial_cmp(&v).unwrap()) {
        Ok(index) => index,
        Err(index) => {
            if index == 0 {
                0
            } else if index >= vs.len() {
                vs.len() - 1
            } else if (v - vs[index - 1]) < (vs[index] - v) {
                index - 1
            } else {
                index
            }
        }
    }
}

/// Encode color as SGR sequence
pub fn color_sgr_encode<C: Color, W: Write>(
    mut out: W,
    color: C,
    depth: ColorDepth,
    foreground: bool,
) -> Result<(), Error> {
    match depth {
        ColorDepth::TrueColor => {
            let [r, g, b] = color.rgb_u8();
            if foreground {
                out.write_all(b";38")?;
            } else {
                out.write_all(b";48")?;
            }
            write!(out, ";2;{};{};{}", r, g, b)?;
        }
        ColorDepth::EightBit => {
            let color: ColorLinear = color.into();
            let ColorLinear([r, g, b, _]) = color;

            // color in the color cube
            let c_red = nearest(r, CUBE);
            let c_green = nearest(g, CUBE);
            let c_blue = nearest(b, CUBE);
            let c_color = ColorLinear::new(CUBE[c_red], CUBE[c_green], CUBE[c_blue], 1.0);

            // nearest grey color
            let g_index = nearest((r + g + b) / 3.0, GREYS);
            let g_color = ColorLinear::new(GREYS[g_index], GREYS[g_index], GREYS[g_index], 1.0);

            // pick grey or cube based on the distance
            let index = if color.distance(&g_color) < color.distance(&c_color) {
                232 + g_index
            } else {
                16 + 36 * c_red + 6 * c_green + c_blue
            };

            if foreground {
                out.write_all(b";38")?;
            } else {
                out.write_all(b";48")?;
            }
            write!(out, ";5;{}", index)?;
        }
        ColorDepth::Gray => {
            let luma = color.luma();
            let index = match nearest(luma, &[0.0, 0.33, 0.66, 1.0]) {
                0 => 30,
                1 => 90,
                2 => 37,
                _ => 97,
            };
            let index = if foreground { index } else { index + 10 };
            write!(out, ";{}", index)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base64() -> Result<(), Error> {
        let mut base64 = Base64Encoder::new(Vec::new());
        base64.write_all(b"te")?;
        base64.write_all(b"rm")?;
        assert_eq!(base64.finish()?, b"dGVybQ==");

        let mut base64 = Base64Encoder::new(Vec::new());
        base64.write_all(b"ter")?;
        assert_eq!(base64.finish()?, b"dGVy");

        let mut base64 = Base64Encoder::new(Vec::new());
        base64.write(b"ab")?;
        assert_eq!(base64.finish()?, b"YWI=");

        Ok(())
    }

    #[test]
    fn test_gray_sgr() -> Result<(), Error> {
        let mut encoder = TTYEncoder::new(TerminalCaps {
            depth: ColorDepth::Gray,
            ..TerminalCaps::default()
        });
        let mut out = Vec::new();
        encoder.encode(
            &mut out,
            TerminalCommand::Face("bg=#ebdbb2,fg=#282828".parse()?),
        )?;

        assert_eq!(
            std::str::from_utf8(out.as_ref()).as_deref(),
            Ok("\x1b[00;30;107m")
        );
        Ok(())
    }
}
