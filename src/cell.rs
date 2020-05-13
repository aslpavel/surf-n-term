use std::{fmt, str::FromStr};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Color {
    pub red: u8,
    pub green: u8,
    pub blue: u8,
    pub alpha: u8,
}

impl Color {
    pub fn new(red: u8, green: u8, blue: u8, alpha: u8) -> Self {
        Self {
            red,
            green,
            blue,
            alpha,
        }
    }

    pub fn rgb_u8(self) -> (u8, u8, u8) {
        (self.red, self.green, self.blue)
    }
}

impl fmt::Debug for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{:02x}{:02x}{:02x}", self.red, self.green, self.blue)?;
        if self.alpha != 255 {
            write!(f, "{:02x}", self.alpha)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct ColorError(String);

impl fmt::Display for ColorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for ColorError {}

impl FromStr for Color {
    type Err = ColorError;

    fn from_str(rgba: &str) -> Result<Color, Self::Err> {
        // FIXME: support different color notations
        if rgba.len() < 7 || !rgba.starts_with('#') || rgba.len() > 9 {
            return Err(ColorError(format!("invalid color `{}`", rgba)));
        }
        let red = u8::from_str_radix(rgba.get(1..3).unwrap(), 16)
            .map_err(|err| ColorError(format!("invalid red component `{}`: {}", rgba, err)))?;
        let green = u8::from_str_radix(rgba.get(3..5).unwrap(), 16)
            .map_err(|err| ColorError(format!("invalid green component `{}`: {}", rgba, err)))?;
        let blue = u8::from_str_radix(rgba.get(5..7).unwrap(), 16)
            .map_err(|err| ColorError(format!("invalid blue component `{}`: {}", rgba, err)))?;
        let alpha = match rgba.get(7..9) {
            None => 255,
            Some(alpha) => u8::from_str_radix(alpha, 16).map_err(|err| {
                ColorError(format!("invalid alpha component `{}`: {}", rgba, err))
            })?,
        };

        Ok(Color {
            red,
            green,
            blue,
            alpha,
        })
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

#[cfg(test)]
mod tests {
    use super::{Color, ColorError};

    #[test]
    fn test_parse_color() -> Result<(), ColorError> {
        assert_eq!("#d3869b".parse::<Color>()?, Color::new(211, 134, 155, 255));
        assert_eq!("#b8bb2680".parse::<Color>()?, Color::new(184, 187, 38, 128));
        Ok(())
    }
}
