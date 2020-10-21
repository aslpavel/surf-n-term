use crate::{Blend, Color, Error, RGBA};
use std::{
    fmt,
    ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign},
    str::FromStr,
};

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
    pub const STRIKE: Self = FaceAttrs { bits: 32 }; // aka Crossed-Out
    const ALL: Self = FaceAttrs { bits: 63 };

    pub fn is_empty(self) -> bool {
        self == Self::EMPTY
    }

    pub fn contains(self, other: Self) -> bool {
        self.bits & other.bits == other.bits
    }

    pub fn insert(self, other: Self) -> Self {
        self | other
    }

    pub fn remove(self, other: Self) -> Self {
        self & (other ^ Self::ALL)
    }

    pub fn names(&self) -> impl Iterator<Item = &'static str> {
        let names = [
            (Self::BOLD, "bold"),
            (Self::ITALIC, "italic"),
            (Self::UNDERLINE, "underline"),
            (Self::BLINK, "blink"),
            (Self::REVERSE, "reverse"),
            (Self::STRIKE, "strike"),
        ];
        let mut index = 0;
        let flags = *self;
        std::iter::from_fn(move || {
            while index < names.len() {
                let (flag, name) = names[index];
                index += 1;
                if flags.contains(flag) {
                    return Some(name);
                }
            }
            None
        })
    }
}

impl BitAnd for FaceAttrs {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self {
            bits: self.bits & rhs.bits,
        }
    }
}

impl BitAndAssign for FaceAttrs {
    fn bitand_assign(&mut self, rhs: Self) {
        self.bits &= rhs.bits
    }
}

impl BitOr for FaceAttrs {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self {
            bits: self.bits | rhs.bits,
        }
    }
}

impl BitOrAssign for FaceAttrs {
    fn bitor_assign(&mut self, rhs: Self) {
        self.bits |= rhs.bits
    }
}

impl BitXor for FaceAttrs {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self {
            bits: self.bits ^ rhs.bits,
        }
    }
}

impl BitXorAssign for FaceAttrs {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.bits ^= rhs.bits
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
    pub fg: Option<RGBA>,
    pub bg: Option<RGBA>,
    pub attrs: FaceAttrs,
}

impl Face {
    pub fn new(fg: Option<RGBA>, bg: Option<RGBA>, attrs: FaceAttrs) -> Self {
        Self { fg, bg, attrs }
    }

    pub fn with_bg(&self, bg: Option<RGBA>) -> Self {
        Face { bg, ..*self }
    }

    pub fn with_fg(&self, fg: Option<RGBA>) -> Self {
        Face { fg, ..*self }
    }

    pub fn with_attrs(&self, attrs: FaceAttrs) -> Self {
        Face { attrs, ..*self }
    }

    pub fn invert(&self) -> Self {
        Face {
            fg: self.bg,
            bg: self.fg,
            ..*self
        }
    }

    /// Overlay `other` face on top of `self`
    pub fn overlay(&self, other: &Self) -> Self {
        let fg = match (self.fg, other.fg) {
            (Some(dst), Some(src)) => Some(dst.blend(src, Blend::Over)),
            (fg, None) => fg,
            (None, fg) => fg,
        };
        let bg = match (self.bg, other.bg) {
            (Some(dst), Some(src)) => Some(dst.blend(src, Blend::Over)),
            (bg, None) => bg,
            (None, bg) => bg,
        };
        Face { fg, bg, ..*other }
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
                    "bold" => face.attrs |= FaceAttrs::BOLD,
                    "italic" => face.attrs |= FaceAttrs::ITALIC,
                    "underline" => face.attrs |= FaceAttrs::UNDERLINE,
                    "blink" => face.attrs |= FaceAttrs::BLINK,
                    "reverse" => face.attrs |= FaceAttrs::REVERSE,
                    "strike" => face.attrs |= FaceAttrs::STRIKE,
                    "" => {}
                    _ => return Err(Error::ParseError("Face", string.to_string())),
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
    fn test_parse_face() -> Result<(), Error> {
        assert_eq!(
            "fg=#98971a,bg=#bdae93, bold ,underline".parse::<Face>()?,
            Face {
                fg: Some(RGBA::new(152, 151, 26, 255)),
                bg: Some(RGBA::new(189, 174, 147, 255)),
                attrs: FaceAttrs::BOLD | FaceAttrs::UNDERLINE,
            }
        );
        Ok(())
    }
}
