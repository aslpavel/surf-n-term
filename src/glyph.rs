use crate::{
    view::{BoxConstraint, Layout, View, ViewContext, ViewLayout, ViewMutLayout},
    Cell, Color, Error, Face, Image, LinColor, Position, Size, Surface, SurfaceMut, SurfaceMutView,
    SurfaceOwned, TerminalSize, TerminalSurface, TerminalSurfaceExt, RGBA,
};
use rasterize::{
    ActiveEdgeRasterizer, Align, Image as _, LineCap, PathBuilder, Point, RGBADeserializer,
    Rasterizer, Scalar, Scene, StrokeStyle, Transform, SVG_COLORS,
};
pub use rasterize::{BBox, FillRule, Path, EPSILON};
use serde::{
    de::{self, DeserializeSeed, IgnoredAny},
    ser::SerializeMap,
    Deserialize, Serialize,
};
use std::{
    borrow::Cow,
    collections::HashMap,
    hash::{Hash, Hasher},
    io::Write,
    str::FromStr,
    sync::Arc,
};

#[derive(Clone, Debug)]
pub enum GlyphScene {
    Symbol { path: Path, fill_rule: FillRule },
    Scene(Scene),
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
    /// Render a frame around the glyph
    frame: Option<GlyphFrame>,
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
        frame: Option<GlyphFrame>,
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
                frame,
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
        let _ = tracing::debug_span!("[Glyph.rasterize]", glyph=?self, ?face, ?size).enter();

        let bg: LinColor = face.bg.unwrap_or_default().into();
        let fg: LinColor = face.fg.unwrap_or_default().into();

        let rasterizer = ActiveEdgeRasterizer::default();
        let mut surf = SurfaceOwned::new_with(
            Size {
                height: size.height,
                width: size.width,
            },
            |_| bg,
        );

        // draw frame
        let scene_bbox = if let Some(frame) = &self.inner.frame {
            frame.rasterize(&rasterizer, surf.as_mut())
        } else {
            BBox::new((0.0, 0.0), (size.width as Scalar, size.height as Scalar))
        };

        // draw scene
        let shape = surf.shape();
        let data = surf.data_mut();
        let tr = Transform::fit_bbox(self.inner.view_box, scene_bbox, Align::Mid);
        match &self.inner.scene {
            GlyphScene::Symbol { path, fill_rule } => {
                for pixel in rasterizer.mask_iter(path, tr, size, *fill_rule) {
                    let pos = Position::new(pixel.y, pixel.x);
                    let offset = shape.offset(pos);
                    data[offset] = data[offset].blend_over(fg.with_alpha(pixel.alpha));
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
                        let pos = Position::new(row, col);
                        let pixel = image_data[image_shape.offset(pos.row, pos.col)];
                        let offset = shape.offset(pos);
                        data[offset] = data[offset].lerp(pixel, pixel.alpha());
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

impl View for Glyph {
    fn render(
        &self,
        ctx: &ViewContext,
        surf: TerminalSurface<'_>,
        layout: ViewLayout<'_>,
    ) -> Result<(), Error> {
        let mut surf = layout.apply_to(surf);
        if ctx.has_glyphs() {
            if let Some(cell) = surf.get_mut(Position::origin()) {
                cell.overlay(Cell::new_glyph(Face::default(), self.clone()));
            }
        } else {
            let mut writer = surf.writer(ctx);
            write!(&mut writer, "{}", self.fallback_str())?;
        }
        Ok(())
    }

    fn layout(
        &self,
        ctx: &ViewContext,
        ct: BoxConstraint,
        mut layout: ViewMutLayout<'_>,
    ) -> Result<(), Error> {
        if ctx.has_glyphs() {
            *layout = Layout::new().with_size(ct.clamp(self.size()));
        } else {
            self.fallback_str().layout(ctx, ct, layout)?;
        }
        Ok(())
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
                    "fallback" => value.clone_into(&mut attrs.fallback),
                    "" => {}
                    _ => return Err(Error::ParseError("Glyph", string.to_owned())),
                }
                Ok(attrs)
            },
        )?;

        let Some(path) = attrs.path else {
            return Err(Error::ParseError(
                "Glyph",
                format!("path is requred: {}", string),
            ));
        };
        Ok(Glyph::new(
            path,
            attrs.fill_rule,
            attrs.view_box,
            attrs.size.unwrap_or(Size::new(1, 3)),
            attrs.fallback,
            None,
        ))
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct GlyphFrame {
    /// Margins (top height%, right width%, bottom height%, left width%)
    #[serde(default, skip_serializing_if = "is_default")]
    pub margin: [Scalar; 4],
    /// Border width (top px, right px, bottom px, left px)
    #[serde(default, skip_serializing_if = "is_default")]
    pub border_width: [Scalar; 4],
    /// Border radius (top-left min%, top-right min%, bottom-right %min, bottom-left min%)
    #[serde(default, skip_serializing_if = "is_default")]
    pub border_radius: [Scalar; 4],
    /// Border color
    #[serde(default, skip_serializing_if = "is_default")]
    pub border_color: Option<RGBA>,
    /// Padding (top height%, right width%, bottom height%, left width%)
    #[serde(default, skip_serializing_if = "is_default")]
    pub padding: [Scalar; 4],
    /// Fill color inside the frame
    #[serde(default, skip_serializing_if = "is_default")]
    pub fill_color: Option<RGBA>,
}

impl GlyphFrame {
    /// Set same border width for all sides
    pub fn with_border_width(self, width: Scalar) -> Self {
        Self {
            border_width: [width; 4],
            ..self
        }
    }

    /// Set same border radius for all corners
    pub fn with_border_radius(self, radius: Scalar) -> Self {
        Self {
            border_radius: [radius; 4],
            ..self
        }
    }

    pub fn rasterize(
        &self,
        rasterizer: &dyn Rasterizer,
        mut surf: SurfaceMutView<'_, LinColor>,
    ) -> BBox {
        let width = surf.width() as Scalar;
        let height = surf.height() as Scalar;

        // margins
        let [m_top, m_right, m_bottom, m_left] = self.margin;
        let m_top = (m_top.clamp(0.0, 100.0) / 100.0 * height).round();
        let m_right = (m_right.clamp(0.0, 100.0) / 100.0 * width).round();
        let m_bottom = (m_bottom.clamp(0.0, 100.0) / 100.0 * height).round();
        let m_left = (m_left.clamp(0.0, 100.0) / 100.0 * width).round();

        // border width
        let [bw_top, bw_right, bw_bottom, bw_left] = self.border_width;
        let bw_top = bw_top.max(0.0);
        let bw_right = bw_right.max(0.0);
        let bw_bottom = bw_bottom.max(0.0);
        let bw_left = bw_left.max(0.0);

        // border radius
        let size = (width - m_right - m_left - (bw_left + bw_right) / 2.0)
            .min(height - m_top - m_bottom - (bw_top + bw_bottom) / 2.0);
        let [br_tl, br_tr, br_br, br_bl] = self.border_radius;
        let br_tl = br_tl.clamp(0.0, 100.0) / 100.0 * size;
        let br_tr = br_tr.clamp(0.0, 100.0) / 100.0 * size;
        let br_br = br_br.clamp(0.0, 100.0) / 100.0 * size;
        let br_bl = br_bl.clamp(0.0, 100.0) / 100.0 * size;

        // paddings
        let [p_top, p_right, p_bottom, p_left] = self.padding;
        let p_top = (p_top.clamp(0.0, 100.0) / 100.0 * height).round();
        let p_right = (p_right.clamp(0.0, 100.0) / 100.0 * width).round();
        let p_bottom = (p_bottom.clamp(0.0, 100.0) / 100.0 * height).round();
        let p_left = (p_left.clamp(0.0, 100.0) / 100.0 * width).round();

        // border path
        let lx = m_left + bw_left / 2.0; // low x
        let ly = m_top + bw_top / 2.0; // low y
        let hx = width - m_right - bw_right / 2.0; // high x
        let hy = height - m_bottom - bw_bottom / 2.0; // high y

        let shape = surf.shape();
        let size = rasterize::Size {
            height: shape.height,
            width: shape.width,
        };
        let data = surf.data_mut();

        // fill
        if let Some(fill_color) = self.fill_color {
            let mut builder = PathBuilder::new();
            builder.move_to((lx + br_tl, ly)).line_to((hx - br_tr, ly));
            if br_tr > EPSILON {
                builder.arc_to((br_tr, br_tr), 0.0, false, true, (hx, ly + br_tr));
            }
            builder.line_to((hx, hy - br_br));
            if br_br > EPSILON {
                builder.arc_to((br_br, br_br), 0.0, false, true, (hx - br_br, hy));
            }
            builder.line_to((lx + br_bl, hy));
            if br_bl > EPSILON {
                builder.arc_to((br_bl, br_bl), 0.0, false, true, (lx, hy - br_bl));
            }
            builder.line_to((lx, ly + br_tl));
            if br_tl > EPSILON {
                builder.arc_to((br_tl, br_tl), 0.0, false, true, (lx + br_tl, ly));
            }
            let path = builder.close().build();

            let fill_color = LinColor::from(fill_color);
            for pixel in
                rasterizer.mask_iter(&path, Transform::identity(), size, FillRule::default())
            {
                let pos = Position::new(pixel.y, pixel.x);
                let offset = shape.offset(pos);
                data[offset] = data[offset].blend_over(fill_color.with_alpha(pixel.alpha));
            }
        }

        // border
        if let Some(border_color) = self.border_color {
            let mut subpaths = Vec::new();
            let stroke_style = StrokeStyle {
                line_cap: LineCap::Round,
                ..Default::default()
            };
            // top
            let top_path = PathBuilder::new()
                .move_to((lx + br_tl, ly))
                .line_to((hx - br_tr, ly))
                .build()
                .stroke(StrokeStyle {
                    width: bw_top,
                    ..stroke_style
                });
            subpaths.extend(top_path);
            // right
            let mut right_build = PathBuilder::new();
            right_build.move_to((hx - br_tr, ly));
            if br_tr > EPSILON {
                right_build.arc_to((br_tr, br_tr), 0.0, false, true, (hx, ly + br_tr));
            }
            right_build.line_to((hx, hy - br_br));
            if br_br > EPSILON {
                right_build.arc_to((br_br, br_br), 0.0, false, true, (hx - br_br, hy));
            }
            let right_path = right_build.build().stroke(StrokeStyle {
                width: bw_right,
                ..stroke_style
            });
            subpaths.extend(right_path);
            // bottom
            let bottom_path = PathBuilder::new()
                .move_to((hx - br_br, hy))
                .line_to((lx + br_bl, hy))
                .build()
                .stroke(StrokeStyle {
                    width: bw_bottom,
                    ..stroke_style
                });
            subpaths.extend(bottom_path);
            // left
            let mut left_build = PathBuilder::new();
            left_build.move_to((lx + br_bl, hy));
            if br_bl > EPSILON {
                left_build.arc_to((br_bl, br_bl), 0.0, false, true, (lx, hy - br_bl));
            }
            left_build.line_to((lx, ly + br_tl));
            if br_tl > EPSILON {
                left_build.arc_to((br_tl, br_tl), 0.0, false, true, (lx + br_tl, ly));
            }
            let left_path = left_build.build().stroke(StrokeStyle {
                width: bw_left,
                ..stroke_style
            });
            subpaths.extend(left_path);
            let path = Path::new(subpaths);

            let border_color = LinColor::from(border_color);
            for pixel in
                rasterizer.mask_iter(&path, Transform::identity(), size, FillRule::default())
            {
                let pos = Position::new(pixel.y, pixel.x);
                let offset = shape.offset(pos);
                data[offset] = data[offset].blend_over(border_color.with_alpha(pixel.alpha));
            }
        }

        BBox::new(
            (m_left + p_left, m_top + p_top),
            (width - m_right - p_right, height - m_bottom - p_bottom),
        )
    }
}

pub struct GlyphFrameDeserializer<'a> {
    pub colors: &'a HashMap<String, RGBA>,
}

impl<'de> Deserialize<'de> for GlyphFrame {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        GlyphFrameDeserializer {
            colors: &SVG_COLORS,
        }
        .deserialize(deserializer)
    }
}

impl<'de> de::DeserializeSeed<'de> for &GlyphFrameDeserializer<'_> {
    type Value = GlyphFrame;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        deserializer.deserialize_map(self)
    }
}

impl<'de> de::Visitor<'de> for &GlyphFrameDeserializer<'_> {
    type Value = GlyphFrame;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("mapping")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: de::MapAccess<'de>,
    {
        let mut margin: Option<[Scalar; 4]> = None;
        let mut border_width: Option<[Scalar; 4]> = None;
        let mut border_radius: Option<[Scalar; 4]> = None;
        let mut border_color: Option<RGBA> = None;
        let mut padding: Option<[Scalar; 4]> = None;
        let mut fill_color: Option<RGBA> = None;

        let color_seed = RGBADeserializer {
            colors: self.colors,
        };

        while let Some(name) = map.next_key::<Cow<'de, str>>()? {
            match name.as_ref() {
                "margin" => {
                    margin.replace(map.next_value()?);
                }
                "border_width" => {
                    border_width.replace(map.next_value()?);
                }
                "border_radius" => {
                    border_radius.replace(map.next_value()?);
                }
                "border_color" => {
                    border_color.replace(map.next_value_seed(color_seed.clone())?);
                }
                "padding" => {
                    padding.replace(map.next_value()?);
                }
                "fill_color" => {
                    fill_color.replace(map.next_value_seed(color_seed.clone())?);
                }
                _ => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }

        Ok(GlyphFrame {
            margin: margin.unwrap_or_default(),
            border_width: border_width.unwrap_or_default(),
            border_radius: border_radius.unwrap_or_default(),
            border_color,
            padding: padding.unwrap_or_default(),
            fill_color,
        })
    }
}

pub struct GlyphDeserializer<'a> {
    pub colors: &'a HashMap<String, RGBA>,
}

impl<'de> Deserialize<'de> for Glyph {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        GlyphDeserializer {
            colors: &SVG_COLORS,
        }
        .deserialize(deserializer)
    }
}

impl<'de> de::DeserializeSeed<'de> for &GlyphDeserializer<'_> {
    type Value = Glyph;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        deserializer.deserialize_map(self)
    }
}

impl<'de> de::Visitor<'de> for &GlyphDeserializer<'_> {
    type Value = Glyph;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("mapping")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: de::MapAccess<'de>,
    {
        let mut scene: Option<Scene> = None;
        let mut path: Option<Path> = None;
        let mut view_box: Option<BBox> = None;
        let mut fallback: Option<String> = None;
        let mut fill_rule: Option<FillRule> = None;
        let mut size: Option<Size> = None;
        let mut frame: Option<GlyphFrame> = None;

        while let Some(name) = map.next_key::<Cow<'de, str>>()? {
            match name.as_ref() {
                "scene" => {
                    // TODO: if scene deserializer is implemented use it here
                    scene.replace(map.next_value()?);
                }
                "path" => {
                    path.replace(map.next_value()?);
                }
                "view_box" => {
                    view_box.replace(map.next_value()?);
                }
                "fallback" => {
                    fallback.replace(map.next_value()?);
                }
                "fill_rule" => {
                    fill_rule.replace(map.next_value()?);
                }
                "size" => {
                    size.replace(map.next_value()?);
                }
                "frame" => {
                    frame.replace(map.next_value_seed(&GlyphFrameDeserializer {
                        colors: self.colors,
                    })?);
                }
                _ => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }

        let size = size.unwrap_or(Size::new(1, 3));
        match (path, scene) {
            (Some(path), None) => Ok(Glyph::new(
                path,
                fill_rule.unwrap_or_default(),
                view_box,
                size,
                fallback.unwrap_or_default(),
                frame,
            )),
            (None, Some(scene)) => {
                let view_box = view_box
                    .or_else(|| scene.bbox(Transform::identity()))
                    .unwrap_or_else(|| BBox::new((0.0, 0.0), (1.0, 1.0)));
                Ok(Glyph {
                    inner: Arc::new(GlyphInner {
                        scene: GlyphScene::Scene(scene),
                        view_box,
                        size,
                        fallback: fallback.unwrap_or_default(),
                        frame,
                    }),
                })
            }
            _ => Err(de::Error::custom("must contain either scene or path")),
        }
    }
}

impl Serialize for Glyph {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut attrs = serializer.serialize_map(Some(6))?;
        match &self.inner.scene {
            GlyphScene::Symbol { path, fill_rule } => {
                attrs.serialize_entry("path", path)?;
                if fill_rule != &FillRule::default() {
                    attrs.serialize_entry("fill_rule", fill_rule)?;
                }
            }
            GlyphScene::Scene(scene) => {
                attrs.serialize_entry("scene", scene)?;
            }
        }
        attrs.serialize_entry("view_box", &self.inner.view_box)?;
        if !self.fallback_str().is_empty() {
            attrs.serialize_entry("fallback", self.fallback_str())?;
        }
        attrs.serialize_entry("size", &self.inner.size)?;
        if let Some(frame) = &self.inner.frame {
            attrs.serialize_entry("frame", frame)?;
        }
        attrs.end()
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
            None,
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
