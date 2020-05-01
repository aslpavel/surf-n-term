use std::{fmt, ops::Deref, str::FromStr};

#[derive(Clone, Copy, PartialEq, Eq)]
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
        if rgba.len() < 7 || !rgba.starts_with("#") || rgba.len() > 9 {
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

/*
pub enum FaceAttr {
    Bold,
    Italic,
    Underline,
    Blink,
    Reverse,
    None,
}
*/

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaceAttrs {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
            attrs: FaceAttrs {},
        }
    }
}

#[derive(Clone, Copy)]
pub struct Glyph([u8; 5]);

impl Deref for Glyph {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Clone)]
pub struct Cell {
    pub face: Face,
    pub glyph: Option<Glyph>,
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
