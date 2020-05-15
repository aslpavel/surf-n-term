use crate::{decoder::Decoder, Position, ViewMutExt};
use std::{fmt, str::FromStr};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Color {
    RGBA(u8, u8, u8, u8),
}

impl Color {
    pub fn rgb_u8(self) -> (u8, u8, u8) {
        let (r, g, b, _) = self.rgba_u8();
        (r, g, b)
    }

    pub fn rgba_u8(self) -> (u8, u8, u8, u8) {
        match self {
            Self::RGBA(r, g, b, a) => (r, g, b, a),
        }
    }

    pub fn from_str(rgba: &str) -> Option<Self> {
        if rgba.len() < 7 || !rgba.starts_with('#') || rgba.len() > 9 {
            return None;
        }
        let mut hex = crate::decoder::hex_decode(rgba[1..].as_bytes());
        let red = hex.next()?;
        let green = hex.next()?;
        let blue = hex.next()?;
        let alpha = if rgba.len() == 9 { hex.next()? } else { 255 };
        Some(Self::RGBA(red, green, blue, alpha))
    }
}

impl FromStr for Color {
    type Err = crate::error::Error;

    fn from_str(color: &str) -> Result<Color, Self::Err> {
        Self::from_str(color).ok_or(crate::error::Error::InvalidColor)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FaceAttrs {
    bits: u16,
}

impl FaceAttrs {
    pub const EMPTY: Self = FaceAttrs { bits: 0 };
    pub const BOLD: Self = FaceAttrs { bits: 1 };
    pub const ITALIC: Self = FaceAttrs { bits: 2 };
    pub const UNDERLINE: Self = FaceAttrs { bits: 4 };
    pub const BLINK: Self = FaceAttrs { bits: 8 };
    pub const REVERSE: Self = FaceAttrs { bits: 16 };

    pub fn is_empty(self) -> bool {
        self == Self::EMPTY
    }

    pub fn contains(self, other: Self) -> bool {
        self.bits & other.bits == other.bits
    }
}

impl std::ops::BitOr for FaceAttrs {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self {
            bits: self.bits | rhs.bits,
        }
    }
}

impl fmt::Debug for FaceAttrs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            write!(f, "Empty")?;
        } else {
            let mut first = true;
            for (flag, name) in &[
                (Self::BOLD, "Bold"),
                (Self::ITALIC, "Italic"),
                (Self::UNDERLINE, "Underline"),
                (Self::BLINK, "Blink"),
                (Self::REVERSE, "Reverse"),
            ] {
                if self.contains(*flag) {
                    if first {
                        first = false;
                        write!(f, "{}", name)?;
                    } else {
                        write!(f, " | {}", name)?;
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Face {
    pub fg: Option<Color>,
    pub bg: Option<Color>,
    pub attrs: FaceAttrs,
}

impl Face {
    pub fn new(fg: Option<Color>, bg: Option<Color>, attrs: FaceAttrs) -> Self {
        Self { fg, bg, attrs }
    }

    pub fn with_bg(&self, bg: Option<Color>) -> Self {
        Face { bg, ..*self }
    }

    pub fn with_fg(&self, fg: Option<Color>) -> Self {
        Face { fg, ..*self }
    }
}

impl Default for Face {
    fn default() -> Self {
        Self {
            fg: None,
            bg: None,
            attrs: FaceAttrs::EMPTY,
        }
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Cell {
    pub face: Face,
    pub glyph: Option<char>,
}

impl Cell {
    pub fn new(face: Face, glyph: Option<char>) -> Self {
        Self { face, glyph }
    }

    pub fn with_face(self, face: Face) -> Self {
        Self { face, ..self }
    }
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            face: Default::default(),
            glyph: None,
        }
    }
}

pub trait TerminalView: ViewMutExt<Item = Cell> {
    fn draw_box(&mut self, face: Option<Face>) {
        let shape = self.shape();
        if shape.width < 2 || shape.height < 2 {
            return;
        }
        let face = face.unwrap_or_default();

        let h = Cell::new(face, Some('─'));
        let v = Cell::new(face, Some('│'));
        self.view_mut(..1, 1..-1).fill(h.clone());
        self.view_mut(-1.., 1..-1).fill(h.clone());
        self.view_mut(1..-1, ..1).fill(v.clone());
        self.view_mut(1..-1, -1..).fill(v.clone());

        self.view_mut(..1, ..1).fill(Cell::new(face, Some('┌')));
        self.view_mut(..1, -1..).fill(Cell::new(face, Some('┐')));
        self.view_mut(-1.., -1..).fill(Cell::new(face, Some('┘')));
        self.view_mut(-1.., ..1).fill(Cell::new(face, Some('└')));
    }

    fn writer(&mut self, pos: Position, face: Option<Face>) -> TerminalViewWriter<'_> {
        let offset = self.shape().width * pos.row + pos.col;
        let mut iter = self.iter_mut();
        if offset > 0 {
            iter.nth(offset - 1);
        }
        TerminalViewWriter {
            face: face.unwrap_or_default(),
            iter,
            decoder: crate::decoder::Utf8Decoder::new(),
        }
    }
}

impl<T: ViewMutExt<Item = Cell> + ?Sized> TerminalView for T {}

pub struct TerminalViewWriter<'a> {
    face: Face,
    iter: crate::surface::ViewMutIter<'a, Cell>,
    decoder: crate::decoder::Utf8Decoder,
}

impl<'a> std::io::Write for TerminalViewWriter<'a> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut cur = std::io::Cursor::new(buf);
        while let Some(glyph) = self.decoder.decode(&mut cur)? {
            let glyph = if glyph == ' ' { None } else { Some(glyph) };
            match self.iter.next() {
                Some(cell) => *cell = Cell::new(self.face, glyph),
                None => return Ok(buf.len()),
            }
        }
        Ok(cur.position() as usize)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_color() -> Result<(), crate::error::Error> {
        assert_eq!("#d3869b".parse::<Color>()?, Color::RGBA(211, 134, 155, 255));
        assert_eq!(
            "#b8bb2680".parse::<Color>()?,
            Color::RGBA(184, 187, 38, 128)
        );
        Ok(())
    }
}
