use crate::{
    Color, Error, Face, Image, LinColor, Size, Surface, SurfaceMut, SurfaceOwned, TerminalSize,
    RGBA,
};
use rasterize::{
    ActiveEdgeRasterizer, Align, Image as _, Point, Rasterizer, Scalar, Scene, Transform,
};
pub use rasterize::{BBox, FillRule, Path};
use serde::{de, Deserialize, Serialize};
use std::{
    hash::{Hash, Hasher},
    str::FromStr,
    sync::Arc,
};
use tracing::debug_span;

#[derive(Clone, Debug)]
pub enum GlyphScene {
    Symbol { path: Path, fill_rule: FillRule },
    Scene(Scene),
}

impl GlyphScene {
    fn bbox(&self, tr: Transform) -> Option<BBox> {
        match self {
            GlyphScene::Symbol { path, .. } => path.bbox(tr),
            GlyphScene::Scene(scene) => scene.bbox(tr),
        }
    }
}

#[derive(Debug, Clone)]
struct GlyphInner {
    /// Scene to by rasterized
    scene: GlyphScene,
    /// View box
    view_box: BBox,
    /// Glyph size in cells
    size: Size,
    /// Fallback text (used when image is not supported)
    fallback: String,
}

/// Glyph defined as an SVG path
#[derive(Clone)]
pub struct Glyph {
    inner: Arc<GlyphInner>,
}

impl Glyph {
    pub fn new(
        path: Path,
        fill_rule: FillRule,
        view_box: Option<BBox>,
        size: Size,
        fallback: String,
    ) -> Self {
        let view_box = view_box
            .or_else(|| {
                let bbox = path.bbox(Transform::identity())?;
                let padding = bbox.width().min(bbox.height()) * 0.15;
                let offset = Point::new(padding, padding);
                Some(BBox::new(bbox.min() - offset, bbox.max() + offset))
            })
            .unwrap_or_else(|| BBox::new((0.0, 0.0), (1.0, 1.0)));
        Self {
            inner: Arc::new(GlyphInner {
                scene: GlyphScene::Symbol { path, fill_rule },
                view_box,
                size,
                fallback,
            }),
        }
    }

    /// Rasterize glyph into an image with provided face.
    pub fn rasterize(&self, face: Face, term_size: TerminalSize) -> Image {
        let pixel_size = term_size.cells_in_pixels(self.inner.size);
        let size = rasterize::Size {
            height: pixel_size.height,
            width: pixel_size.width,
        };
        let tr = Transform::fit_size(self.inner.view_box, size, Align::Mid);

        let bg_rgba = face.bg.unwrap_or_else(|| RGBA::new(0, 0, 0, 0));
        let bg: LinColor = bg_rgba.into();
        let fg: LinColor = face
            .fg
            .unwrap_or_else(|| RGBA::new(255, 255, 255, 255))
            .into();

        let _ = debug_span!("glyph rasterize", path=?self, ?face, ?size).enter();
        let rasterizer = ActiveEdgeRasterizer::default();
        let mut surf = SurfaceOwned::new_with(size.height, size.width, |_, _| bg_rgba);
        let shape = surf.shape();
        let data = surf.data_mut();
        match &self.inner.scene {
            GlyphScene::Symbol { path, fill_rule } => {
                for pixel in rasterizer.mask_iter(path, tr, size, *fill_rule) {
                    data[shape.offset(pixel.y, pixel.x)] = bg.lerp(fg, pixel.alpha as f32).into();
                }
            }
            GlyphScene::Scene(scene) => {
                let image = scene.render(
                    &rasterizer,
                    Transform::identity(),
                    Some(BBox::new(
                        (0.0, 0.0),
                        (size.width as Scalar, size.height as Scalar),
                    )),
                    None,
                );
                let image_data = image.data();
                let image_shape = image.shape();
                for row in 0..image.height() {
                    for col in 0..image.width() {
                        let pixel = image_data[image_shape.offset(row, col)];
                        data[shape.offset(row, col)] = bg.lerp(pixel, pixel.alpha()).into();
                    }
                }
            }
        }

        Image::new(surf)
    }

    /// Size of the glyph in cells
    pub fn size(&self) -> Size {
        self.inner.size
    }

    /// View box used to render scene
    pub fn view_box(&self) -> BBox {
        self.inner.view_box
    }

    /// Scene use to render glyph
    pub fn scene(&self) -> &GlyphScene {
        &self.inner.scene
    }

    pub fn fallback_str(&self) -> &str {
        &self.inner.fallback
    }
}

impl std::fmt::Debug for Glyph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}

impl PartialEq for Glyph {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for Glyph {}

impl Hash for Glyph {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.inner).hash(state);
    }
}

impl FromStr for Glyph {
    type Err = Error;

    fn from_str(string: &str) -> Result<Self, Self::Err> {
        struct Attrs {
            path: Option<Path>,
            fill_rule: FillRule,
            view_box: Option<BBox>,
            size: Option<Size>,
            fallback: String,
        }

        let attrs = string.split(';').try_fold(
            Attrs {
                path: None,
                fill_rule: FillRule::default(),
                view_box: None,
                size: None,
                fallback: String::new(),
            },
            |mut attrs, attr| {
                let mut iter = attr.splitn(2, '=');
                let key = iter.next().unwrap_or_default().trim();
                let value = iter.next().unwrap_or_default();
                match key {
                    "path" => {
                        attrs.path.replace(value.parse()?);
                    }
                    "fill_rule" => attrs.fill_rule = value.parse()?,
                    "view_box" => {
                        attrs.view_box.replace(value.parse()?);
                    }
                    "size" => {
                        attrs.size.replace(value.parse()?);
                    }
                    "fallback" => attrs.fallback = value.to_owned(),
                    "" => {}
                    _ => return Err(Error::ParseError("Glyph", string.to_owned())),
                }
                Ok(attrs)
            },
        )?;

        let Some(path) = attrs.path else {
            return Err(Error::ParseError("Glyph", format!("path is requred: {}",string)))
        };
        Ok(Glyph::new(
            path,
            attrs.fill_rule,
            attrs.view_box,
            attrs.size.unwrap_or_else(glyph_default_size),
            attrs.fallback,
        ))
    }
}

#[derive(Serialize, Deserialize)]
struct GlyphSerde {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    scene: Option<Scene>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    path: Option<Path>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    view_box: Option<BBox>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    fallback: Option<String>,
    #[serde(default, skip_serializing_if = "is_default")]
    fill_rule: FillRule,
    #[serde(default = "glyph_default_size")]
    size: Size,
}

fn glyph_default_size() -> Size {
    Size::new(1, 3)
}

impl Serialize for Glyph {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let view_box = match self.inner.scene.bbox(Transform::identity()) {
            Some(bbox) if bbox == self.inner.view_box => None,
            _ => Some(self.inner.view_box),
        };
        let fallback = self
            .fallback_str()
            .is_empty()
            .then_some(self.fallback_str().to_owned());
        match &self.inner.scene {
            GlyphScene::Symbol { path, fill_rule } => GlyphSerde {
                path: Some(path.clone()),
                view_box,
                fill_rule: *fill_rule,
                size: self.inner.size,
                scene: None,
                fallback,
            }
            .serialize(serializer),
            GlyphScene::Scene(scene) => GlyphSerde {
                scene: Some(scene.clone()),
                view_box,
                size: self.inner.size,
                fill_rule: FillRule::default(),
                path: None,
                fallback,
            }
            .serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for Glyph {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let glyph = GlyphSerde::deserialize(deserializer)?;
        if let Some(path) = glyph.path {
            Ok(Glyph::new(
                path,
                glyph.fill_rule,
                glyph.view_box,
                glyph.size,
                glyph.fallback.unwrap_or_default(),
            ))
        } else if let Some(scene) = glyph.scene {
            let view_box = glyph
                .view_box
                .or_else(|| scene.bbox(Transform::identity()))
                .unwrap_or_else(|| BBox::new((0.0, 0.0), (1.0, 1.0)));
            Ok(Glyph {
                inner: Arc::new(GlyphInner {
                    scene: GlyphScene::Scene(scene),
                    view_box,
                    size: glyph.size,
                    fallback: glyph.fallback.unwrap_or_default(),
                }),
            })
        } else {
            Err(de::Error::custom("must contain either scene or path"))
        }
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
        let glyph = Glyph::new(
            path.clone(),
            FillRule::NonZero,
            Some(BBox::new((1.0, 0.0), (25.0, 21.0))),
            Size::new(1, 2),
            String::new(),
        );
        let glyph_str = serde_json::to_string(&glyph)?;
        let glyph_de: Glyph = serde_json::from_str(glyph_str.as_ref())?;
        assert_eq!(glyph.view_box(), glyph_de.view_box());
        assert_eq!(glyph.size(), glyph_de.size());
        match glyph_de.scene() {
            GlyphScene::Symbol {
                path: path_de,
                fill_rule: fill_rule_de,
            } => {
                assert_eq!(&path, path_de);
                assert_eq!(FillRule::NonZero, *fill_rule_de);
            }
            _ => panic!("symbol expected"),
        }

        Ok(())
    }
}
