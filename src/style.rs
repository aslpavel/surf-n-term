use crate::error::Error;
use std::{fmt, str::FromStr};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Color {
    RGBA([u8; 4]),
}

impl Color {
    pub fn rgb_u8(self) -> (u8, u8, u8) {
        let (r, g, b, _) = self.rgba_u8();
        (r, g, b)
    }

    pub fn rgba_u8(self) -> (u8, u8, u8, u8) {
        match self {
            Self::RGBA([r, g, b, a]) => (r, g, b, a),
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
        Some(Self::RGBA([red, green, blue, alpha]))
    }
}

impl FromStr for Color {
    type Err = Error;

    fn from_str(string: &str) -> Result<Self, Self::Err> {
        Self::from_str(string).ok_or(Error::ParseColorError)
    }
}

impl fmt::Debug for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Color::RGBA([r, g, b, a]) => {
                write!(f, "#{:02x}{:02x}{:02x}", r, g, b)?;
                if *a != 255 {
                    write!(f, "{:02x}", a)?;
                }
            }
        }
        Ok(())
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

    pub fn names(&self) -> impl Iterator<Item = &'static str> {
        let names = [
            (Self::BOLD, "bold"),
            (Self::ITALIC, "italic"),
            (Self::UNDERLINE, "underline"),
            (Self::BLINK, "blink"),
            (Self::REVERSE, "reverse"),
        ];
        let mut index = 0;
        let flags = *self;
        std::iter::from_fn(move || {
            while index < names.len() {
                let (flag, name) = names[index];
                if flags.contains(flag) {
                    return Some(name);
                }
                index += 1;
            }
            None
        })
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
        let names: Vec<_> = self.names().collect();
        write!(f, "FaceAttrs({})", names.join(","))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

impl FromStr for Face {
    type Err = Error;

    fn from_str(string: &str) -> Result<Self, Self::Err> {
        string
            .split(',')
            .try_fold(Face::default(), |mut face, attrs| {
                let mut iter = attrs.splitn(2, '=');
                let key = iter.next().unwrap_or_default().trim().to_lowercase();
                let value = iter.next().unwrap_or_default().trim();
                match key.as_str() {
                    "fg" => face.fg = Some(value.parse()?),
                    "bg" => face.bg = Some(value.parse()?),
                    "bold" => face.attrs = face.attrs | FaceAttrs::BOLD,
                    "italic" => face.attrs = face.attrs | FaceAttrs::ITALIC,
                    "underline" => face.attrs = face.attrs | FaceAttrs::UNDERLINE,
                    "blink" => face.attrs = face.attrs | FaceAttrs::BLINK,
                    "reverse" => face.attrs = face.attrs | FaceAttrs::REVERSE,
                    _ => return Err(Error::ParseFaceError),
                }
                Ok(face)
            })
    }
}

impl fmt::Debug for Face {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = Vec::new();
        if let Some(fg) = self.fg {
            result.push(format!("fg={:?}", fg));
        }
        if let Some(bg) = self.bg {
            result.push(format!("bg={:?}", bg));
        }
        result.extend(self.attrs.names().map(String::from));
        write!(f, "Face({})", result.join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_color() -> Result<(), Error> {
        assert_eq!(
            "#d3869b".parse::<Color>()?,
            Color::RGBA([211, 134, 155, 255])
        );
        assert_eq!(
            "#b8bb2680".parse::<Color>()?,
            Color::RGBA([184, 187, 38, 128])
        );
        Ok(())
    }

    #[test]
    fn test_parse_face() -> Result<(), Error> {
        assert_eq!(
            "fg=#98971a, bg=#bdae93, bold ,underline".parse::<Face>()?,
            Face {
                fg: Some(Color::RGBA([152, 151, 26, 255])),
                bg: Some(Color::RGBA([189, 174, 147, 255])),
                attrs: FaceAttrs::BOLD | FaceAttrs::UNDERLINE,
            }
        );
        Ok(())
    }
}