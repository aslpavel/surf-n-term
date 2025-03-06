//! Encoders
use crate::{
    Color, DecMode, FaceAttrs, LinColor, TerminalCaps, TerminalColor, TerminalCommand,
    UnderlineStyle, decoder::KEYBOARD_LEVEL, error::Error,
};
use std::{
    cmp::Ordering,
    io::{self, Write},
    str::FromStr,
};

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
pub struct TTYEncoder {
    caps: TerminalCaps,
    chunks: Chunks,
}

impl Default for TTYEncoder {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl TTYEncoder {
    pub fn new(caps: TerminalCaps) -> Self {
        Self {
            caps,
            chunks: Chunks::default(),
        }
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
            EraseScreen => out.write_all(b"\x1b[2J")?,
            EraseChars(count) => write!(out, "\x1b[{}X", count)?,
            Face(face) => {
                self.chunks.clear();
                self.chunks.push(b"0");
                if let Some(fg) = face.fg {
                    color_sgr_encode(
                        &mut self.chunks,
                        fg,
                        self.caps.depth,
                        SGRColorType::Foreground,
                    )?;
                }
                if let Some(bg) = face.bg {
                    color_sgr_encode(
                        &mut self.chunks,
                        bg,
                        self.caps.depth,
                        SGRColorType::Background,
                    )?;
                }
                match face.attrs.underline() {
                    UnderlineStyle::Straight => self.chunks.push(b"4"),
                    UnderlineStyle::Double => self.chunks.push(b"4:2"),
                    UnderlineStyle::Curly => self.chunks.push(b"4:3"),
                    UnderlineStyle::Dotted => self.chunks.push(b"4:4"),
                    UnderlineStyle::Dashed => self.chunks.push(b"4:5"),
                    _ => {}
                }
                if !face.attrs.is_empty() {
                    for (flag, code) in [
                        (FaceAttrs::BOLD, b"1"),
                        (FaceAttrs::ITALIC, b"3"),
                        (FaceAttrs::BLINK, b"5"),
                        (FaceAttrs::REVERSE, b"7"),
                        (FaceAttrs::STRIKE, b"9"),
                    ] {
                        if face.attrs.contains(flag) {
                            self.chunks.push(code);
                        }
                    }
                }
                out.write_all(b"\x1b[")?;
                self.chunks.drain(b";", &mut out)?;
                out.write_all(b"m")?;
            }
            FaceModify(face_modify) => {
                self.chunks.clear();
                if face_modify.reset {
                    self.chunks.push(b"0");
                }
                if let Some(fg) = face_modify.fg {
                    color_sgr_encode(
                        &mut self.chunks,
                        fg,
                        self.caps.depth,
                        SGRColorType::Foreground,
                    )?;
                }
                if let Some(bg) = face_modify.bg {
                    color_sgr_encode(
                        &mut self.chunks,
                        bg,
                        self.caps.depth,
                        SGRColorType::Background,
                    )?;
                }
                match face_modify.underline {
                    None => {}
                    Some(UnderlineStyle::None) => self.chunks.push(b"24"),
                    Some(UnderlineStyle::Straight) => self.chunks.push(b"4"),
                    Some(UnderlineStyle::Double) => self.chunks.push(b"4:2"),
                    Some(UnderlineStyle::Curly) => self.chunks.push(b"4:3"),
                    Some(UnderlineStyle::Dotted) => self.chunks.push(b"4:4"),
                    Some(UnderlineStyle::Dashed) => self.chunks.push(b"4:5"),
                }
                if let Some(color) = face_modify.underline_color {
                    color_sgr_encode(
                        &mut self.chunks,
                        color,
                        self.caps.depth,
                        SGRColorType::Underline,
                    )?;
                }
                for (flag, on, off) in [
                    (face_modify.bold, b"1", b"21"),
                    (face_modify.italic, b"3", b"23"),
                    (face_modify.blink, b"5", b"25"),
                    (face_modify.strike, b"9", b"29"),
                ] {
                    match flag {
                        None => {}
                        Some(true) => self.chunks.push(on),
                        Some(false) => self.chunks.push(off),
                    }
                }
                if !self.chunks.is_empty() {
                    out.write_all(b"\x1b[")?;
                    self.chunks.drain(b";", &mut out)?;
                    out.write_all(b"m")?;
                }
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
            Raw(data) => out.write_all(&data)?,
        }

        Ok(())
    }
}

#[derive(Default)]
struct Chunks {
    buffer: Vec<u8>,
    offsets: Vec<usize>,
}

impl Chunks {
    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
        self.offsets.clear();
    }

    pub fn mark(&mut self) {
        self.offsets.push(self.buffer.len());
    }

    pub fn push(&mut self, chunk: &[u8]) {
        self.buffer.extend(chunk);
        self.mark();
    }

    pub fn iter(&self) -> impl Iterator<Item = &[u8]> {
        let mut index = 0;
        let mut start = 0;
        std::iter::from_fn(move || {
            if index >= self.offsets.len() {
                return None;
            }
            let end = self.offsets[index];
            let chunk = &self.buffer[start..end];
            start = end;
            index += 1;
            Some(chunk)
        })
    }

    pub fn drain(&mut self, sep: &[u8], mut out: impl Write) -> io::Result<()> {
        for (index, chunk) in self.iter().enumerate() {
            if index != 0 {
                out.write_all(sep)?;
            }
            out.write_all(chunk)?
        }
        self.clear();
        Ok(())
    }
}

impl Write for Chunks {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.buffer.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

const BASE64_ENCODE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Writable object which encodes input to base64 and writes it in underlying stream
pub struct Base64Encoder<W> {
    inner: W,
    buffer: [u8; 3],
    size: usize,
}

impl<W: Write> Base64Encoder<W> {
    pub fn new(inner: W) -> Self {
        Self {
            inner,
            buffer: [0u8; 3],
            size: 0,
        }
    }

    /// finalize base64 stream, returning underlying stream
    pub fn finish(self) -> std::io::Result<W> {
        let Self {
            mut inner,
            buffer,
            size,
        } = self;
        let mut dst = [b'='; 4];
        let mut iter = buffer[..size].iter();
        if let Some(s0) = iter.next() {
            dst[0] = BASE64_ENCODE[(s0 >> 2) as usize];
            if let Some(s1) = iter.next() {
                dst[1] = BASE64_ENCODE[(((s0 << 4) | (s1 >> 4)) & 0x3f) as usize];
                if let Some(s2) = iter.next() {
                    dst[2] = BASE64_ENCODE[(((s1 << 2) | (s2 >> 6)) & 0x3f) as usize];
                    dst[3] = BASE64_ENCODE[(s2 & 0x3f) as usize];
                } else {
                    dst[2] = BASE64_ENCODE[((s1 << 2) & 0x3f) as usize];
                }
            } else {
                dst[1] = BASE64_ENCODE[((s0 << 4) & 0x3f) as usize];
            }
            inner.write_all(&dst)?;
        }
        Ok(inner)
    }
}

impl<W: Write> Write for Base64Encoder<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        for b in buf.iter().copied() {
            self.buffer[self.size] = b;
            self.size += 1;
            if self.size == 3 {
                let [s0, s1, s2] = self.buffer;
                let mut dst = [b'='; 4];
                dst[0] = BASE64_ENCODE[(s0 >> 2) as usize];
                dst[1] = BASE64_ENCODE[(((s0 << 4) | (s1 >> 4)) & 0x3f) as usize];
                dst[2] = BASE64_ENCODE[(((s1 << 2) | (s2 >> 6)) & 0x3f) as usize];
                dst[3] = BASE64_ENCODE[(s2 & 0x3f) as usize];
                self.inner.write_all(&dst)?;
                self.size = 0;
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
const CUBE: &[f32] = &[0.0, 0.114435, 0.242281, 0.42869, 0.679542, 1.0];

/// Grey colors available in 256 color mode converted to linear RGB color space.
const GREYS: &[f32] = &[
    0.002428, 0.006049, 0.011612, 0.019382, 0.029557, 0.042311, 0.057805, 0.076185, 0.097587,
    0.122139, 0.14996, 0.181164, 0.215861, 0.254152, 0.296138, 0.341914, 0.391572, 0.445201,
    0.502886, 0.564712, 0.630757, 0.701102, 0.775822, 0.854993,
];

fn nearest(v: f32, vs: &[f32]) -> usize {
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

pub enum SGRColorType {
    Foreground,
    Background,
    Underline,
}

/// Encode color as SGR sequence
fn color_sgr_encode<C: Color>(
    chunks: &mut Chunks,
    color: C,
    depth: ColorDepth,
    sgr_color_type: SGRColorType,
) -> Result<(), Error>
where
    LinColor: From<C>,
{
    match depth {
        ColorDepth::TrueColor => {
            let [r, g, b] = color.to_rgb();
            match sgr_color_type {
                SGRColorType::Foreground => chunks.push(b"38"),
                SGRColorType::Background => chunks.push(b"48"),
                SGRColorType::Underline => chunks.push(b"58"),
            }
            chunks.push(b"2");
            for c in [r, g, b] {
                write!(chunks, "{}", c)?;
                chunks.mark();
            }
        }
        ColorDepth::EightBit => {
            let color = LinColor::from(color);
            let [r, g, b, _]: [f32; 4] = color.into();

            // color in the color cube
            let c_red = nearest(r, CUBE);
            let c_green = nearest(g, CUBE);
            let c_blue = nearest(b, CUBE);
            let c_color = LinColor::new(CUBE[c_red], CUBE[c_green], CUBE[c_blue], 1.0);

            // nearest grey color
            let g_index = nearest((r + g + b) / 3.0, GREYS);
            let g_color = LinColor::new(GREYS[g_index], GREYS[g_index], GREYS[g_index], 1.0);

            // pick grey or cube based on the distance
            let index = if color.distance(g_color) < color.distance(c_color) {
                232 + g_index
            } else {
                16 + 36 * c_red + 6 * c_green + c_blue
            };

            match sgr_color_type {
                SGRColorType::Foreground => chunks.push(b"38"),
                SGRColorType::Background => chunks.push(b"48"),
                SGRColorType::Underline => chunks.push(b"58"),
            }
            chunks.push(b"5");
            write!(chunks, "{}", index)?;
            chunks.mark();
        }
        ColorDepth::Gray => {
            let luma = color.luma();
            let index = match nearest(luma, &[0.0, 0.33, 0.66, 1.0]) {
                0 => 30,
                1 => 90,
                2 => 37,
                _ => 97,
            };
            let index = match sgr_color_type {
                SGRColorType::Foreground => index,
                SGRColorType::Background => index + 10,
                SGRColorType::Underline => return Ok(()),
            };
            write!(chunks, "{}", index)?;
            chunks.mark();
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
        base64.write_all(b"ab")?;
        assert_eq!(base64.finish()?, b"YWI=");

        Ok(())
    }

    fn tty_encode_assert(
        encoder: &mut TTYEncoder,
        cmd: TerminalCommand,
        reference: &str,
    ) -> Result<(), Error> {
        let mut out = Vec::new();
        encoder.encode(&mut out, cmd)?;
        assert_eq!(std::str::from_utf8(out.as_ref()).as_deref(), Ok(reference));
        Ok(())
    }

    #[test]
    fn test_gray_sgr() -> Result<(), Error> {
        let mut encoder = TTYEncoder::new(TerminalCaps {
            depth: ColorDepth::Gray,
            ..TerminalCaps::default()
        });

        tty_encode_assert(
            &mut encoder,
            TerminalCommand::Face("bg=#ebdbb2,fg=#282828".parse()?),
            "\x1b[0;30;107m",
        )?;

        Ok(())
    }

    #[test]
    fn test_tty_encoder() -> Result<(), Error> {
        let mut encoder = TTYEncoder::new(TerminalCaps {
            depth: ColorDepth::Gray,
            ..TerminalCaps::default()
        });

        tty_encode_assert(
            &mut encoder,
            TerminalCommand::Face("underline_curly".parse()?),
            "\x1b[0;4:3m",
        )?;

        Ok(())
    }
}
