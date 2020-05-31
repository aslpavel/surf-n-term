use crate::common::clamp;
use crate::error::Error;
use std::{
    fmt,
    ops::{Add, Mul},
    str::FromStr,
};

/// Color in linear RGB color space with premultiplied alpha
#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct ColorLinear([f32; 4]);

impl Mul<f32> for ColorLinear {
    type Output = Self;

    #[inline]
    fn mul(self, v: f32) -> Self::Output {
        let Self([r, g, b, a]) = self;
        Self([r * v, g * v, b * v, a * v])
    }
}

impl Add<Self> for ColorLinear {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self::Output {
        let Self([r0, g0, b0, a0]) = self;
        let Self([r1, g1, b1, a1]) = other;
        Self([r0 + r1, g0 + g1, b0 + b1, a0 + a1])
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Blend {
    Over,
    Out,
    In,
    Atop,
    Xor,
}

impl<C: From<ColorLinear> + Into<ColorLinear>> ColorExt for C {}

pub trait ColorExt: From<ColorLinear> + Into<ColorLinear> {
    fn blend(self, other: impl Into<ColorLinear>, method: Blend) -> Self {
        // Reference:
        // https://ciechanow.ski/alpha-compositing/
        // http://ssp.impulsetrain.com/porterduff.html
        let dst = self.into();
        let dst_a = dst.0[3];
        let src = other.into();
        let src_a = src.0[3];
        let color = match method {
            Blend::Over => src + dst * (1.0 - src_a),
            Blend::Out => src * (1.0 - dst_a),
            Blend::In => src * dst_a,
            Blend::Atop => src * dst_a + dst * (1.0 - src_a),
            Blend::Xor => src * (1.0 - dst_a) + dst * (1.0 - src_a),
        };
        color.into()
    }

    fn lerp(self, other: impl Into<ColorLinear>, t: f32) -> Self {
        let start = self.into();
        let end = other.into();
        let color = start * (1.0 - t) + end * t;
        color.into()
    }

    fn luma(self) -> f32 {
        let ColorLinear([r, g, b, _]) = self.into();
        r * 0.2126 + g * 0.7152 + b * 0.0722
    }
}

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(value: f32) -> f32 {
    if value <= 0.0031308 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Color {
    RGBA([u8; 4]),
}

impl Default for Color {
    fn default() -> Self {
        Self::RGBA([0, 0, 0, 0])
    }
}

impl Color {
    pub fn rgb_u8(self) -> [u8; 3] {
        let [r, g, b, a] = self.rgba_u8();
        if a == 255 {
            [r, g, b]
        } else {
            let alpha = a as f32 / 255.0;
            let [r, g, b, _] = Color::RGBA([0, 0, 0, 255]).lerp(self, alpha).rgba_u8();
            [r, g, b]
        }
    }

    pub fn rgba_u8(self) -> [u8; 4] {
        match self {
            Self::RGBA([r, g, b, a]) => [r, g, b, a],
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

impl From<Color> for ColorLinear {
    fn from(color: Color) -> Self {
        let [r, g, b, a] = color.rgba_u8();
        let a = (a as f32) / 255.0;
        let r = srgb_to_linear((r as f32) / 255.0) * a;
        let g = srgb_to_linear((g as f32) / 255.0) * a;
        let b = srgb_to_linear((b as f32) / 255.0) * a;
        ColorLinear([r, g, b, a])
    }
}

impl From<ColorLinear> for Color {
    fn from(color: ColorLinear) -> Self {
        let ColorLinear([r, g, b, a]) = color;
        if a < std::f32::EPSILON {
            Color::RGBA([0, 0, 0, 0])
        } else {
            let a = clamp(a, 0.0, 1.0);
            let r = (linear_to_srgb(clamp(r / a, 0.0, 1.0)) * 255.0) as u8;
            let g = (linear_to_srgb(clamp(g / a, 0.0, 1.0)) * 255.0) as u8;
            let b = (linear_to_srgb(clamp(b / a, 0.0, 1.0)) * 255.0) as u8;
            let a = (a * 255.0) as u8;
            Color::RGBA([r, g, b, a])
        }
    }
}

impl FromStr for Color {
    type Err = Error;

    fn from_str(string: &str) -> Result<Self, Self::Err> {
        Self::from_str(string).ok_or(Error::ParseColorError)
    }
}

impl FromStr for ColorLinear {
    type Err = Error;

    fn from_str(string: &str) -> Result<Self, Self::Err> {
        Color::from_str(string)
            .map(ColorLinear::from)
            .ok_or(Error::ParseColorError)
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

    /// Overlay `other` face on top of `self`
    pub fn overlay(&self, other: &Self) -> Self {
        let fg = match (self.fg, other.fg) {
            (Some(dst), Some(src)) => Some(dst.blend(src, Blend::Over)),
            (_, src) => src,
        };
        let bg = match (self.bg, other.bg) {
            (Some(dst), Some(src)) => Some(dst.blend(src, Blend::Over)),
            (_, src) => src,
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
            "fg=#98971a,bg=#bdae93, bold ,underline".parse::<Face>()?,
            Face {
                fg: Some(Color::RGBA([152, 151, 26, 255])),
                bg: Some(Color::RGBA([189, 174, 147, 255])),
                attrs: FaceAttrs::BOLD | FaceAttrs::UNDERLINE,
            }
        );
        Ok(())
    }

    #[test]
    fn test_color_linear() -> Result<(), Error> {
        let color = "#fe801970".parse()?;
        assert_eq!(Color::from(ColorLinear::from(color)), color);
        Ok(())
    }
}
