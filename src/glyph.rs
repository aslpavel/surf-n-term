use crate::{
    Color, ColorLinear, Face, Image, Size, Surface, SurfaceMut, SurfaceOwned, TerminalSize, RGBA,
};
use rasterize::{ActiveEdgeRasterizer, Align, Rasterizer, Transform};
pub use rasterize::{BBox, FillRule, Path};
use serde::{Deserialize, Serialize};
use std::{
    hash::{Hash, Hasher},
    io::Write,
    sync::Arc,
};
use tracing::debug_span;

/// Glyph defined as an SVG path
#[derive(Debug, Clone)]
pub struct Glyph {
    /// Rasterize path representing the glyph
    path: Arc<Path>,
    /// View box
    view_box: BBox,
    /// Fill rule
    fill_rule: FillRule,
    /// Glyph size in cells
    size: Size,
    /// Hash that is used to determine equality of the paths
    hash: u64,
}

impl Glyph {
    pub fn new(
        path: impl Into<Arc<Path>>,
        fill_rule: FillRule,
        view_box: Option<BBox>,
        size: Size,
    ) -> Self {
        let path = path.into();
        let view_box = view_box
            .or_else(|| path.bbox(Transform::identity()))
            .unwrap_or_else(|| BBox::new((0.0, 0.0), (1.0, 1.0)));

        let mut hasher = GlyphHasher::new();
        path.write_svg_path(&mut hasher).unwrap();
        write!(hasher, "{:?}", view_box).unwrap();
        let hash = hasher.finish();

        Self {
            path,
            view_box,
            hash,
            size,
            fill_rule,
        }
    }

    /// Rasterize glyph into an image with provided face.
    pub fn rasterize(&self, face: Face, term_size: TerminalSize) -> Image {
        let pixel_size = term_size.cells_in_pixels(self.size);
        let size = rasterize::Size {
            height: pixel_size.height,
            width: pixel_size.width,
        };
        let tr = Transform::fit_size(self.view_box, size, Align::Mid);

        let bg_rgba = face.bg.unwrap_or_else(|| RGBA::new(0, 0, 0, 0));
        let bg: ColorLinear = bg_rgba.into();
        let fg: ColorLinear = face
            .fg
            .unwrap_or_else(|| RGBA::new(255, 255, 255, 255))
            .into();

        let _ = debug_span!("glyph rasterize", path=%self.path.to_svg_path(), ?face, ?size).enter();
        let rasterizer = ActiveEdgeRasterizer::default();
        let mut surf = SurfaceOwned::new_with(size.height, size.width, |_, _| bg_rgba);
        let shape = surf.shape();
        let data = surf.data_mut();
        for pixel in rasterizer.mask_iter(&self.path, tr, size, self.fill_rule) {
            data[shape.offset(pixel.y, pixel.x)] = bg.lerp(fg, pixel.alpha).into();
        }

        Image::new(surf)
    }

    /// Size of the glyph in cells
    pub fn size(&self) -> Size {
        self.size
    }
}

impl PartialEq for Glyph {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash && self.fill_rule == other.fill_rule && self.size == other.size
    }
}

impl Eq for Glyph {}

impl Hash for Glyph {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash.hash(state);
        self.fill_rule.hash(state);
    }
}

struct GlyphHasher {
    hasher: fnv::FnvHasher,
}

impl GlyphHasher {
    fn new() -> Self {
        Self {
            hasher: Default::default(),
        }
    }

    fn finish(&self) -> u64 {
        self.hasher.finish()
    }
}

impl Write for GlyphHasher {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.hasher.write(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
struct GlyphSerde {
    path: Arc<Path>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    view_box: Option<BBox>,
    #[serde(default, skip_serializing_if = "is_default")]
    fill_rule: FillRule,
    size: Size,
}

impl Serialize for Glyph {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let view_box = match self.path.bbox(Transform::identity()) {
            Some(bbox) if bbox == self.view_box => None,
            _ => Some(self.view_box),
        };
        GlyphSerde {
            path: self.path.clone(),
            view_box,
            fill_rule: self.fill_rule,
            size: self.size,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Glyph {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let glyph = GlyphSerde::deserialize(deserializer)?;
        Ok(Glyph::new(
            glyph.path,
            glyph.fill_rule,
            glyph.view_box,
            glyph.size,
        ))
    }
}

/// Check if value is equal to default
/// useful for skipping serialization if value is equal to default value
/// by adding `#[serde(default, skip_serializing_if = "is_default")]`
pub(crate) fn is_default<T: Default + PartialEq>(val: &T) -> bool {
    val == &T::default()
}

#[cfg(test)]
mod tests {
    use super::*;
    const TEST_ICON: &str = "M1,1 h18 v18 h-18 Z";

    #[test]
    fn test_glyph_serde() -> Result<(), Box<dyn std::error::Error>> {
        let path: Path = TEST_ICON.parse()?;
        let term = Glyph::new(
            path,
            FillRule::NonZero,
            Some(BBox::new((1.0, 0.0), (25.0, 21.0))),
            Size::new(1, 2),
        );
        let term_str = serde_json::to_string(&term)?;
        assert_eq!(term, serde_json::from_str(term_str.as_ref())?);

        let term_json = serde_json::json!({
            "path": TEST_ICON,
            "fill_rule": "nonzero",
            "view_box": [1, 0, 24, 21],
            "size": [1, 2],
        });
        assert_eq!(term, serde_json::from_value(term_json)?);

        Ok(())
    }
}
