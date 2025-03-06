//! Type describing foreground/background/style-attrs of the terminal cell
use rasterize::{SVG_COLORS, utils::ArrayIter};
use serde::{Deserialize, Serialize, de::DeserializeSeed};

use crate::{Color, Error, RGBA};
use std::{
    borrow::Cow,
    collections::HashMap,
    fmt,
    ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign},
    str::FromStr,
};

/// Underline style
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum UnderlineStyle {
    #[default]
    None,
    Straight,
    Double,
    Curly,
    Dotted,
    Dashed,
}

impl From<UnderlineStyle> for FaceAttrs {
    fn from(value: UnderlineStyle) -> Self {
        FaceAttrs::pack(value, 0)
    }
}

/// Face style attributes
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct FaceAttrs {
    // first 3bit is underline style rest are flags
    bits: u16,
}

impl FaceAttrs {
    pub const EMPTY: Self = FaceAttrs { bits: 0 };

    pub const UNDERLINE: Self = FaceAttrs { bits: 1 };
    pub const UNDERLINE_DOUBLE: Self = FaceAttrs { bits: 2 };
    pub const UNDERLINE_CURLY: Self = FaceAttrs { bits: 3 };
    pub const UNDERLINE_DOTTED: Self = FaceAttrs { bits: 4 };
    pub const UNDERLINE_DASHED: Self = FaceAttrs { bits: 5 };
    const UNDERLINE_BITS: u16 = 3;

    pub const BOLD: Self = FaceAttrs {
        bits: 1 << Self::UNDERLINE_BITS,
    };
    pub const ITALIC: Self = FaceAttrs {
        bits: 2 << Self::UNDERLINE_BITS,
    };
    pub const BLINK: Self = FaceAttrs {
        bits: 4 << Self::UNDERLINE_BITS,
    };
    pub const REVERSE: Self = FaceAttrs {
        bits: 8 << Self::UNDERLINE_BITS,
    };
    pub const STRIKE: Self = FaceAttrs {
        bits: 16 << Self::UNDERLINE_BITS,
    }; // aka Crossed-Out

    const ALL_FLAGS: u16 = 31;

    /// Empty/Default style
    pub fn is_empty(self) -> bool {
        self == Self::EMPTY
    }

    /// Check if self contains any of the other attributes
    pub fn contains(self, other: Self) -> bool {
        let (self_under, self_flags) = self.unpack();
        let (other_under, other_flags) = other.unpack();
        if other_under != UnderlineStyle::None && self_under != other_under {
            return false;
        }
        self_flags & other_flags == other_flags
    }

    /// Add all attributes set in the other
    pub fn insert(self, other: Self) -> Self {
        let (self_under, self_flags) = self.unpack();
        let (other_under, other_flags) = other.unpack();
        let under = if other_under != UnderlineStyle::None {
            other_under
        } else {
            self_under
        };
        Self::pack(under, self_flags | other_flags)
    }

    /// Remove all attributes set in the other
    pub fn remove(self, other: Self) -> Self {
        let (self_under, self_flags) = self.unpack();
        let (other_under, other_flags) = other.unpack();
        let under = if other_under != UnderlineStyle::None {
            UnderlineStyle::None
        } else {
            self_under
        };
        Self::pack(under, self_flags & (other_flags ^ Self::ALL_FLAGS))
    }

    /// List names of all set attributes
    pub fn names(&self) -> impl Iterator<Item = &'static str> {
        let mut iter: ArrayIter<&'static str, 6> = ArrayIter::new();
        let (under, _) = self.unpack();
        match under {
            UnderlineStyle::None => (),
            UnderlineStyle::Straight => iter.push("underline"),
            UnderlineStyle::Double => iter.push("underline_double"),
            UnderlineStyle::Curly => iter.push("underline_curly"),
            UnderlineStyle::Dotted => iter.push("underline_dotted"),
            UnderlineStyle::Dashed => iter.push("underline_dashed"),
        }
        for (flag, name) in [
            (Self::BOLD, "bold"),
            (Self::ITALIC, "italic"),
            (Self::BLINK, "blink"),
            (Self::REVERSE, "reverse"),
            (Self::STRIKE, "strike"),
        ] {
            if self.bits & flag.bits != 0 {
                iter.push(name);
            }
        }
        iter
    }

    pub fn underline(&self) -> UnderlineStyle {
        match 0b111 & self.bits {
            1 => UnderlineStyle::Straight,
            2 => UnderlineStyle::Double,
            3 => UnderlineStyle::Curly,
            4 => UnderlineStyle::Dotted,
            5 => UnderlineStyle::Dashed,
            _ => UnderlineStyle::None,
        }
    }

    fn unpack(self) -> (UnderlineStyle, u16) {
        (self.underline(), self.bits >> 3)
    }

    fn pack(underline: UnderlineStyle, flags: u16) -> Self {
        let underline_bits = match underline {
            UnderlineStyle::None => 0,
            UnderlineStyle::Straight => 1,
            UnderlineStyle::Double => 2,
            UnderlineStyle::Curly => 3,
            UnderlineStyle::Dotted => 4,
            UnderlineStyle::Dashed => 5,
        };
        Self {
            bits: underline_bits | (flags << 3),
        }
    }
}

impl BitAnd for FaceAttrs {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        let (lhs_under, lhs_flags) = self.unpack();
        let (rhs_under, rhs_flags) = rhs.unpack();
        let under = if lhs_under == rhs_under {
            lhs_under
        } else {
            UnderlineStyle::None
        };
        Self::pack(under, lhs_flags & rhs_flags)
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
        let (lhs_under, lhs_flags) = self.unpack();
        let (rhs_under, rhs_flags) = rhs.unpack();
        let under = if rhs_under == UnderlineStyle::None {
            lhs_under
        } else {
            rhs_under
        };
        Self::pack(under, lhs_flags | rhs_flags)
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
        let (lhs_under, lhs_flags) = self.unpack();
        let (rhs_under, rhs_flags) = rhs.unpack();
        let under = if rhs_under == UnderlineStyle::None {
            lhs_under
        } else {
            rhs_under
        };
        Self::pack(under, lhs_flags ^ rhs_flags)
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

/// Type describing foreground/background/style-attrs of the terminal cell
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Face {
    /// Foreground color
    pub fg: Option<RGBA>,
    /// Background color
    pub bg: Option<RGBA>,
    /// Style attributes
    pub attrs: FaceAttrs,
}

impl Face {
    pub fn new(fg: Option<RGBA>, bg: Option<RGBA>, attrs: FaceAttrs) -> Self {
        Self { fg, bg, attrs }
    }

    /// Override background color
    pub fn with_bg(&self, bg: Option<RGBA>) -> Self {
        Face { bg, ..*self }
    }

    /// Override foreground color
    pub fn with_fg(&self, fg: Option<RGBA>) -> Self {
        Face { fg, ..*self }
    }

    /// Override style attributes
    pub fn with_attrs(&self, attrs: FaceAttrs) -> Self {
        Face { attrs, ..*self }
    }

    /// Swap foreground and background colors
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
            (Some(dst), Some(src)) => Some(dst.blend_over(src)),
            (fg, None) => fg,
            (None, fg) => fg,
        };
        let bg = match (self.bg, other.bg) {
            (Some(dst), Some(src)) => Some(dst.blend_over(src)),
            (bg, None) => bg,
            (None, bg) => bg,
        };
        let attrs = if other.attrs == FaceAttrs::EMPTY {
            self.attrs
        } else {
            other.attrs
        };
        Face { fg, bg, attrs }
    }

    pub fn from_str_named(string: &str, colors: &HashMap<String, RGBA>) -> Result<Face, Error> {
        string
            .split(',')
            .try_fold(Face::default(), |mut face, attrs| {
                let mut iter = attrs.splitn(2, '=');
                let key = iter.next().unwrap_or_default().trim();
                let value = iter.next().unwrap_or_default().trim();
                match key {
                    "fg" => face.fg = Some(RGBA::from_str_named(value, colors)?),
                    "bg" => face.bg = Some(RGBA::from_str_named(value, colors)?),
                    "underline" => face.attrs |= FaceAttrs::UNDERLINE,
                    "underline_double" => face.attrs |= FaceAttrs::UNDERLINE_DOUBLE,
                    "underline_curly" => face.attrs |= FaceAttrs::UNDERLINE_CURLY,
                    "underline_dotted" => face.attrs |= FaceAttrs::UNDERLINE_DOTTED,
                    "underline_dashed" => face.attrs |= FaceAttrs::UNDERLINE_DASHED,
                    "bold" => face.attrs |= FaceAttrs::BOLD,
                    "italic" => face.attrs |= FaceAttrs::ITALIC,
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

impl FromStr for Face {
    type Err = Error;

    fn from_str(string: &str) -> Result<Self, Self::Err> {
        Self::from_str_named(string, &SVG_COLORS)
    }
}

impl fmt::Display for Face {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        if let Some(fg) = self.fg {
            write!(f, "fg={}", fg)?;
            first = false;
        }
        if let Some(bg) = self.bg {
            if !first {
                write!(f, ",")?;
            }
            write!(f, "bg={}", bg)?;
            first = false;
        }
        for attr in self.attrs.names() {
            if !first {
                write!(f, ",")?;
            }
            write!(f, "{}", attr)?;
            first = false;
        }
        Ok(())
    }
}

impl fmt::Debug for Face {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Face(")?;
        let mut first = true;
        if let Some(fg) = self.fg {
            write!(f, "fg={:?}", fg)?;
            first = false;
        }
        if let Some(bg) = self.bg {
            if !first {
                write!(f, ",")?;
            }
            write!(f, "bg={:?}", bg)?;
            first = false;
        }
        for attr in self.attrs.names() {
            if !first {
                write!(f, ",")?;
            }
            write!(f, "{}", attr)?;
            first = false;
        }
        write!(f, ")")?;
        Ok(())
    }
}

impl Serialize for Face {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for Face {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Cow::<'de, str>::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Clone)]
pub struct FaceDeserializer<'a> {
    pub colors: &'a HashMap<String, RGBA>,
}

impl<'de> DeserializeSeed<'de> for FaceDeserializer<'_> {
    type Value = Face;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let color = std::borrow::Cow::<'de, str>::deserialize(deserializer)?;
        Face::from_str_named(color.as_ref(), self.colors).map_err(serde::de::Error::custom)
    }
}

/// Command that modifies current face
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct FaceModify {
    pub reset: bool,
    pub fg: Option<RGBA>,
    pub bg: Option<RGBA>,
    pub underline: Option<UnderlineStyle>,
    pub underline_color: Option<RGBA>,
    pub bold: Option<bool>,
    pub italic: Option<bool>,
    pub blink: Option<bool>,
    pub strike: Option<bool>,
}

impl FaceModify {
    pub fn apply(&self, mut face: Face) -> Face {
        if self.reset {
            face = Face::default();
        }
        if let Some(fg) = self.fg {
            face.fg = Some(fg);
        }
        if let Some(bg) = self.bg {
            face.bg = Some(bg);
        }
        if let Some(underline) = self.underline {
            face.attrs |= underline.into();
        }
        // TODO: underline_color
        for (update, flag) in [
            (self.bold, FaceAttrs::BOLD),
            (self.italic, FaceAttrs::ITALIC),
            (self.blink, FaceAttrs::BLINK),
            (self.strike, FaceAttrs::BOLD),
        ] {
            match update {
                Some(true) => face.attrs = face.attrs.insert(flag),
                Some(false) => face.attrs = face.attrs.remove(flag),
                _ => {}
            }
        }
        face
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_face() -> Result<(), Error> {
        let face = Face {
            fg: Some(RGBA::new(152, 151, 26, 255)),
            bg: Some(RGBA::new(189, 174, 147, 255)),
            attrs: FaceAttrs::BOLD | FaceAttrs::UNDERLINE,
        };
        let face_str: Face = "fg=#98971a,bg=#bdae93, bold ,underline".parse()?;
        assert_eq!(face, face_str);

        let face_str: Face = face.to_string().parse()?;
        assert_eq!(face, face_str);

        assert_eq!(
            "fg=#b22222,bg=#f0f8ff".parse::<Face>()?,
            "fg=firebrick,bg=aliceblue".parse()?
        );

        Ok(())
    }

    #[test]
    fn test_serde() -> Result<(), Box<dyn std::error::Error>> {
        use serde_json::de::StrRead;

        let mut colors = HashMap::new();
        colors.insert("purple".to_owned(), "#b16286".parse()?);

        let face_str = r#""bg=purple/.3,fg=#282828""#;

        let mut deserializer = serde_json::Deserializer::new(StrRead::new(face_str));
        let color = FaceDeserializer { colors: &colors }.deserialize(&mut deserializer)?;
        assert_eq!(color, "bg=#b16286/.3,fg=#282828".parse()?);

        Ok(())
    }
}
