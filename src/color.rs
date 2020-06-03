use crate::common::clamp;
use crate::error::Error;
use std::{
    fmt,
    ops::{Add, Mul},
    str::FromStr,
};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Blend {
    Over,
    Out,
    In,
    Atop,
    Xor,
}

pub trait Color: From<ColorLinear> + Into<ColorLinear> + Copy {
    /// Convert color into 32-bit sRGB array with alpha (channels are not pre-multiplied).
    fn rgba_u8(self) -> [u8; 4];

    /// Convert color into 24-bit sRGB array without alpha (channels are not pre-multiplied).
    fn rgb_u8(self) -> [u8; 3] {
        let [r, g, b, a] = self.rgba_u8();
        if a == 255 {
            [r, g, b]
        } else {
            let alpha = a as f32 / 255.0;
            ColorLinear([0.0, 0.0, 0.0, 1.0]).lerp(self, alpha).rgb_u8()
        }
    }

    /// Blend current color with the other color, with the specified blend method.
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

    /// Linear interpolation between self and other colors.
    fn lerp(self, other: impl Into<ColorLinear>, t: f32) -> Self {
        let start = self.into();
        let end = other.into();
        let color = start * (1.0 - t) + end * t;
        color.into()
    }

    /// Calculate luma of the color.
    fn luma(self) -> f32 {
        let ColorLinear([r, g, b, _]) = self.into();
        r * 0.2126 + g * 0.7152 + b * 0.0722
    }
}

/// Convert SRGB color component into a Linear RGB color component.
fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert Linear RGB color component into a SRGB color component.
fn linear_to_srgb(value: f32) -> f32 {
    if value <= 0.0031308 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

/// Color in linear RGB color space with premultiplied alpha
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
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

impl ColorLinear {
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self([r, g, b, a])
    }
}

impl Color for ColorLinear {
    fn rgba_u8(self) -> [u8; 4] {
        RGBA::from(self).rgba_u8()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RGBA([u8; 4]);

impl Default for RGBA {
    fn default() -> Self {
        Self([0, 0, 0, 0])
    }
}

impl RGBA {
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        RGBA([r, g, b, a])
    }

    pub fn from_str_opt(rgba: &str) -> Option<Self> {
        if rgba.len() < 7 || !rgba.starts_with('#') || rgba.len() > 9 {
            return None;
        }
        let mut hex = crate::decoder::hex_decode(rgba[1..].as_bytes());
        let red = hex.next()?;
        let green = hex.next()?;
        let blue = hex.next()?;
        let alpha = if rgba.len() == 9 { hex.next()? } else { 255 };
        Some(Self([red, green, blue, alpha]))
    }
}

impl Color for RGBA {
    fn rgba_u8(self) -> [u8; 4] {
        self.0
    }
}

impl From<RGBA> for ColorLinear {
    fn from(color: RGBA) -> Self {
        let [r, g, b, a] = color.rgba_u8();
        let a = (a as f32) / 255.0;
        let r = srgb_to_linear((r as f32) / 255.0) * a;
        let g = srgb_to_linear((g as f32) / 255.0) * a;
        let b = srgb_to_linear((b as f32) / 255.0) * a;
        ColorLinear([r, g, b, a])
    }
}

impl From<ColorLinear> for RGBA {
    fn from(color: ColorLinear) -> Self {
        let ColorLinear([r, g, b, a]) = color;
        if a < std::f32::EPSILON {
            Self([0, 0, 0, 0])
        } else {
            let a = clamp(a, 0.0, 1.0);
            let r = (linear_to_srgb(clamp(r / a, 0.0, 1.0)) * 255.0) as u8;
            let g = (linear_to_srgb(clamp(g / a, 0.0, 1.0)) * 255.0) as u8;
            let b = (linear_to_srgb(clamp(b / a, 0.0, 1.0)) * 255.0) as u8;
            let a = (a * 255.0) as u8;
            Self([r, g, b, a])
        }
    }
}

impl FromStr for RGBA {
    type Err = Error;

    fn from_str(string: &str) -> Result<Self, Self::Err> {
        Self::from_str_opt(string).ok_or(Error::ParseColorError)
    }
}

impl FromStr for ColorLinear {
    type Err = Error;

    fn from_str(string: &str) -> Result<Self, Self::Err> {
        RGBA::from_str(string).map(ColorLinear::from)
    }
}

impl fmt::Debug for RGBA {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [r, g, b, a] = self.rgba_u8();
        write!(f, "#{:02x}{:02x}{:02x}", r, g, b)?;
        if a != 255 {
            write!(f, "{:02x}", a)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_color() -> Result<(), Error> {
        assert_eq!("#d3869b".parse::<RGBA>()?, RGBA([211, 134, 155, 255]));
        assert_eq!("#b8bb2680".parse::<RGBA>()?, RGBA([184, 187, 38, 128]));
        Ok(())
    }

    #[test]
    fn test_color_linear() -> Result<(), Error> {
        let color = "#fe801970".parse()?;
        assert_eq!(RGBA::from(ColorLinear::from(color)), color);
        Ok(())
    }
}
