use std::io::Write;

pub mod render;
pub mod surface;

pub use render::{Align, Curve, FillRule, Path, Scalar, Transform};
pub use surface::{Surface, SurfaceMut, SurfaceOwned};

#[inline]
pub fn clamp<T>(val: T, min: T, max: T) -> T
where
    T: PartialOrd,
{
    if val < min {
        min
    } else if val > max {
        max
    } else {
        val
    }
}

pub fn timeit<F: FnOnce() -> R, R>(msg: &str, f: F) -> R {
    let start = std::time::Instant::now();
    let result = f();
    log::debug!("{} {:?}", msg, start.elapsed());
    result
}

/// Save surface as PPM
pub fn surf_to_ppm<S, W>(surf: S, mut w: W) -> Result<(), std::io::Error>
where
    S: Surface,
    S::Item: Color,
    W: Write,
{
    write!(w, "P6 {} {} 255 ", surf.width(), surf.height())?;
    for color in surf.iter() {
        w.write_all(&color.to_rgb())?;
    }
    Ok(())
}

/// Save surface as PNG
pub fn surf_to_png<S, W>(surf: S, w: W) -> Result<(), png::EncodingError>
where
    S: Surface,
    S::Item: Color,
    W: Write,
{
    let mut encoder = png::Encoder::new(w, surf.width() as u32, surf.height() as u32);
    encoder.set_color(png::ColorType::RGBA);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    let mut stream_writer = writer.stream_writer();
    for color in surf.iter() {
        stream_writer.write_all(&color.to_rgba())?;
    }
    stream_writer.flush()?;
    Ok(())
}

fn linear_to_srgb(value: Scalar) -> Scalar {
    if value <= 0.0031308 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

pub trait Color {
    fn to_rgb(&self) -> [u8; 3];
    fn to_rgba(&self) -> [u8; 4];
}

impl Color for Scalar {
    fn to_rgb(&self) -> [u8; 3] {
        let color = (linear_to_srgb(1.0 - *self) * 255.0).round() as u8;
        [color; 3]
    }

    fn to_rgba(&self) -> [u8; 4] {
        // let color = (clamp(1.0 - *self, 0.0, 1.0) * 255.0).round() as u8;
        let color = (linear_to_srgb(1.0 - *self) * 255.0).round() as u8;
        [color, color, color, 255]
    }
}
