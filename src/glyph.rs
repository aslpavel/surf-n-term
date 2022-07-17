use crate::{
    Color, ColorLinear, Face, Image, Size, Surface, SurfaceMut, SurfaceOwned, TerminalSize, RGBA,
};
use rasterize::{ActiveEdgeRasterizer, Align, Image as _, Rasterizer, Scalar, Scene, Transform};
pub use rasterize::{BBox, FillRule, Path};
use serde::{de, Deserialize, Serialize};
use std::{
    hash::{Hash, Hasher},
    sync::Arc,
};
use tracing::debug_span;

#[derive(Debug, Clone)]
enum GlyphScene {
    Symbol {
        path: Arc<Path>,
        fill_rule: FillRule,
    },
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

impl PartialEq for GlyphScene {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Symbol {
                    path: l_path,
                    fill_rule: l_fill_rule,
                },
                Self::Symbol {
                    path: r_path,
                    fill_rule: r_fill_rule,
                },
            ) => Arc::ptr_eq(l_path, r_path) && l_fill_rule == r_fill_rule,
            (Self::Scene(l0), Self::Scene(r0)) => l0 == r0,
            _ => false,
        }
    }
}

impl Eq for GlyphScene {}

impl Hash for GlyphScene {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match &self {
            GlyphScene::Symbol { path, fill_rule } => {
                Arc::as_ptr(path).hash(state);
                fill_rule.hash(state)
            }
            GlyphScene::Scene(scene) => scene.hash(state),
        }
    }
}

/// Glyph defined as an SVG path
#[derive(Debug, Clone, Hash)]
pub struct Glyph {
    /// Scene to by rasterized
    scene: GlyphScene,
    /// View box
    view_box: BBox,
    /// Glyph size in cells
    size: Size,
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
        Self {
            scene: GlyphScene::Symbol { path, fill_rule },
            view_box,
            size,
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

        let _ = debug_span!("glyph rasterize", path=?self, ?face, ?size).enter();
        let rasterizer = ActiveEdgeRasterizer::default();
        let mut surf = SurfaceOwned::new_with(size.height, size.width, |_, _| bg_rgba);
        let shape = surf.shape();
        let data = surf.data_mut();
        match &self.scene {
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
        self.size
    }
}

impl PartialEq for Glyph {
    fn eq(&self, other: &Self) -> bool {
        self.scene == other.scene && self.view_box == other.view_box && self.size == other.size
    }
}

impl Eq for Glyph {}

#[derive(Serialize, Deserialize)]
struct GlyphSerde {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    scene: Option<Scene>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    path: Option<Arc<Path>>,
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
        let view_box = match self.scene.bbox(Transform::identity()) {
            Some(bbox) if bbox == self.view_box => None,
            _ => Some(self.view_box),
        };
        match &self.scene {
            GlyphScene::Symbol { path, fill_rule } => GlyphSerde {
                path: Some(path.clone()),
                view_box,
                fill_rule: *fill_rule,
                size: self.size,
                scene: None,
            }
            .serialize(serializer),
            GlyphScene::Scene(scene) => GlyphSerde {
                scene: Some(scene.clone()),
                view_box,
                size: self.size,
                fill_rule: FillRule::default(),
                path: None,
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
            ))
        } else if let Some(scene) = glyph.scene {
            let view_box = glyph
                .view_box
                .or_else(|| scene.bbox(Transform::identity()))
                .unwrap_or_else(|| BBox::new((0.0, 0.0), (1.0, 1.0)));
            Ok(Glyph {
                scene: GlyphScene::Scene(scene),
                view_box,
                size: glyph.size,
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
        );
        let glyph_str = serde_json::to_string(&glyph)?;
        let glyph_de: Glyph = serde_json::from_str(glyph_str.as_ref())?;
        assert_eq!(glyph.view_box, glyph_de.view_box);
        assert_eq!(glyph.size, glyph_de.size);
        match glyph_de.scene {
            GlyphScene::Symbol {
                path: path_de,
                fill_rule: fill_rule_de,
            } => {
                assert_eq!(&path, path_de.as_ref());
                assert_eq!(FillRule::NonZero, fill_rule_de);
            }
            _ => panic!("symbol expected"),
        }

        Ok(())
    }
}
