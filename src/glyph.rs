use crate::{
    Color, ColorLinear, Face, Image, Size, Surface, SurfaceMut, SurfaceOwned, TerminalSize, RGBA,
};
use rasterize::{ActiveEdgeRasterizer, Align, Rasterizer, Scalar, Transform};
pub use rasterize::{BBox, FillRule, Path};
use serde::{de, ser::SerializeStruct, Deserialize, Serialize};
use std::{
    borrow::Cow,
    hash::{Hash, Hasher},
    io::Write,
    sync::Arc,
};
use tracing::debug;

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
    pub fn new(path: Path, fill_rule: FillRule, view_box: Option<BBox>, size: Size) -> Self {
        let view_box = view_box
            .or_else(|| path.bbox(Transform::identity()))
            .unwrap_or_else(|| BBox::new((0.0, 0.0), (1.0, 1.0)));

        let mut hasher = GlyphHasher::new();
        path.write_svg_path(&mut hasher).unwrap();
        write!(&mut hasher, "{:?}", view_box).unwrap();
        let hash = hasher.finish();

        Self {
            path: Arc::new(path),
            view_box,
            hash,
            size,
            fill_rule,
        }
    }

    /// Rasterize glyph into an image with provied face.
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

        let rasterizer = ActiveEdgeRasterizer::default();
        let mut surf = SurfaceOwned::new_with(size.height, size.width, |_, _| bg_rgba);
        let shape = surf.shape();
        let data = surf.data_mut();
        for pixel in rasterizer.mask_iter(&self.path, tr, size, self.fill_rule) {
            data[shape.offset(pixel.y, pixel.x)] = bg.lerp(fg, pixel.alpha).into();
        }

        debug!(path=%self.path.to_svg_path(), ?face, ?size, "glyph rasterized");
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

impl Serialize for Glyph {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut glyph = serializer.serialize_struct("Glyph", 4)?;

        // serialize as SVG d attribute
        glyph.serialize_field("path", &self.path.to_svg_path())?;

        // same as SVG viewBox attribute
        glyph.serialize_field(
            "view_box",
            &(
                self.view_box.x(),
                self.view_box.y(),
                self.view_box.width(),
                self.view_box.height(),
            ),
        )?;

        // use SVG names for fill-rule attribute
        let fill_rule = match self.fill_rule {
            FillRule::EvenOdd => "evenodd",
            FillRule::NonZero => "nonzero",
        };
        glyph.serialize_field("fill_rule", fill_rule)?;

        // size in cells as (height, width)
        glyph.serialize_field("size", &(self.size.height, self.size.width))?;

        glyph.end()
    }
}

impl<'de> Deserialize<'de> for Glyph {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct GlyphVisitor;
        const FIELDS: &[&str] = &["path", "view_box", "fill_rule", "size"];

        impl<'de> de::Visitor<'de> for GlyphVisitor {
            type Value = Glyph;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("Glyph struct with {path|view_box|fill_rule|size} attributes")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: de::MapAccess<'de>,
            {
                let mut path: Option<Path> = None;
                let mut view_box: Option<BBox> = None;
                let mut fill_rule: Option<FillRule> = None;
                let mut size: Option<Size> = None;
                while let Some(field) = map.next_key::<Cow<'de, str>>()? {
                    match field.as_ref() {
                        "path" => {
                            let path_str: String = map.next_value()?;
                            path.replace(match path_str.parse() {
                                Ok(path) => path,
                                Err(error) => return Err(de::Error::custom(error)),
                            });
                        }
                        "view_box" => {
                            let (minx, miny, width, height): (Scalar, Scalar, Scalar, Scalar) =
                                map.next_value()?;
                            view_box
                                .replace(BBox::new((minx, miny), (minx + width, miny + height)));
                        }
                        "fill_rule" => {
                            fill_rule.replace(match map.next_value::<Cow<'de, str>>()?.as_ref() {
                                "evenodd" => FillRule::EvenOdd,
                                "nonzero" => FillRule::NonZero,
                                fill_rule => {
                                    return Err(de::Error::custom(&format!(
                                    "failed to parse fill_rule {} (expected {{evenodd|nonzero}})",
                                    fill_rule
                                )))
                                }
                            });
                        }
                        "size" => {
                            let (height, width): (usize, usize) = map.next_value()?;
                            size.replace(Size { height, width });
                        }
                        name => {
                            return Err(de::Error::unknown_field(name, FIELDS));
                        }
                    }
                }
                Ok(Glyph::new(
                    path.unwrap_or_default(),
                    fill_rule.unwrap_or_default(),
                    view_box,
                    size.unwrap_or_else(|| Size::new(1, 1)),
                ))
            }
        }

        deserializer.deserialize_struct("Glyph", FIELDS, GlyphVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TERMINAL_ICON: &str = r#"
    M20,19V7H4V19H20M20,3A2,2 0 0,1 22,5V19A2,2 0 0,1 20,21H4A2,2 0 0,1 2,19V5C2,3.89 2.9,3 4,3H20
    M13,17V15H18V17H13M9.58,13L5.57,9H8.4L11.7,12.3C12.09,12.69 12.09,13.33 11.7,13.72L8.42,17H5.59L9.58,13Z
    "#;

    #[test]
    fn test_glyph_serde() -> Result<(), Box<dyn std::error::Error>> {
        let path: Path = TERMINAL_ICON.parse()?;
        let term = Glyph::new(
            path,
            FillRule::NonZero,
            Some(BBox::new((1.0, 0.0), (25.0, 21.0))),
            Size::new(1, 2),
        );
        let term_str = serde_json::to_string(&term)?;
        assert_eq!(term, serde_json::from_str(term_str.as_ref())?);

        let term_json = serde_json::json!({
            "path": TERMINAL_ICON,
            "fill_rule": "nonzero",
            "view_box": [1, 0, 24, 21],
            "size": [1, 2],
        });
        assert_eq!(term, serde_json::from_value(term_json)?);

        Ok(())
    }
}
