//! Defines [View] that represents anything that can be rendered into a terminal.
//! As well as some useful implementations such as [Text], [Flex], [Container], ...
mod layout;
pub use layout::{
    Layout, Tree, TreeId, TreeMut, TreeMutView, TreeNode, TreeStore, TreeView, ViewLayout,
    ViewLayoutStore, ViewMutLayout,
};

mod container;
pub use container::{Align, Container, Margins};

pub mod flex;
pub use flex::{Flex, FlexChild, FlexRef, Justify};

mod scrollbar;
use rasterize::{RGBADeserializer, SVG_COLORS};
pub use scrollbar::{ScrollBar, ScrollBarFn, ScrollBarPosition};

mod text;
use serde_json::Value;
pub use text::Text;

mod dynamic;
pub use dynamic::Dynamic;

mod frame;
pub use frame::Frame;

mod offscreen;
pub use offscreen::Offscreen;

pub use either::{self, Either};

use crate::{
    Cell, Error, Face, FaceAttrs, FaceDeserializer, Image, Position, RGBA, Size, Surface,
    SurfaceMut, SurfaceOwned, SurfaceView, Terminal, TerminalSurface, TerminalSurfaceExt,
    encoder::ColorDepth, glyph::GlyphDeserializer, image::ImageAsciiView,
};
use serde::{
    Deserialize, Deserializer, Serialize,
    de::{self, DeserializeSeed},
};
use std::{any::Any, collections::HashMap, fmt::Debug, sync::Arc};

use self::text::TextDeserializer;

pub type BoxView<'a> = Box<dyn View + 'a>;
pub type ArcView<'a> = Arc<dyn View + 'a>;

/// View is anything that can be layed out and rendered to the terminal
pub trait View: Send + Sync {
    /// Render view into a given surface with the provided layout
    fn render(
        &self,
        ctx: &ViewContext,
        surf: TerminalSurface<'_>,
        layout: ViewLayout<'_>,
    ) -> Result<(), Error>;

    /// Compute layout of the view based on the constraints
    fn layout(
        &self,
        ctx: &ViewContext,
        ct: BoxConstraint,
        layout: ViewMutLayout<'_>,
    ) -> Result<(), Error>;

    fn layout_new<'a>(
        &self,
        ctx: &ViewContext,
        ct: BoxConstraint,
        store: &'a mut ViewLayoutStore,
    ) -> Result<ViewMutLayout<'a>, Error> {
        let mut layout = ViewMutLayout::new(store, Layout::default());
        self.layout(ctx, ct, layout.view_mut())?;
        Ok(layout)
    }

    /// Convert into boxed view
    fn boxed<'a>(self) -> BoxView<'a>
    where
        Self: Sized + Send + Sync + 'a,
    {
        Box::new(self)
    }

    /// Convert into Arc-ed view
    fn arc<'a>(self) -> ArcView<'a>
    where
        Self: Sized + Send + Sync + 'a,
    {
        Arc::new(self)
    }

    /// Wrapper around view that implements [std::fmt::Debug] which renders
    /// view. Only supposed to be used for debugging.
    fn debug(&self, size: Size) -> ViewDebug<&'_ Self>
    where
        Self: Sized,
    {
        ViewDebug { view: self, size }
    }

    /// Wrapper around view that calls trace function on every layout call
    fn trace_layout<T>(self, trace: T) -> TraceLayout<Self, T>
    where
        T: Fn(&BoxConstraint, ViewLayout<'_>),
        Self: Sized,
    {
        TraceLayout { view: self, trace }
    }

    /// Tag the view
    fn tag<T>(self, tag: T) -> Tag<T, Self>
    where
        T: Any + Clone,
        Self: Sized,
    {
        Tag::new(tag, self)
    }

    /// Convert into left side of [Either] view
    fn left_view<R>(self) -> Either<Self, R>
    where
        Self: Sized,
    {
        Either::Left(self)
    }

    /// Convert into right side of [Either] view
    fn right_view<L>(self) -> Either<L, Self>
    where
        Self: Sized,
    {
        Either::Right(self)
    }
}

impl<V: View + ?Sized> View for &V {
    fn render(
        &self,
        ctx: &ViewContext,
        surf: TerminalSurface<'_>,
        layout: ViewLayout<'_>,
    ) -> Result<(), Error> {
        (**self).render(ctx, surf, layout)
    }

    fn layout(
        &self,
        ctx: &ViewContext,
        ct: BoxConstraint,
        layout: ViewMutLayout<'_>,
    ) -> Result<(), Error> {
        (**self).layout(ctx, ct, layout)
    }
}

impl<T: View + ?Sized> View for Box<T> {
    fn render(
        &self,
        ctx: &ViewContext,
        surf: TerminalSurface<'_>,
        layout: ViewLayout<'_>,
    ) -> Result<(), Error> {
        (**self).render(ctx, surf, layout)
    }

    fn layout(
        &self,
        ctx: &ViewContext,
        ct: BoxConstraint,
        layout: ViewMutLayout<'_>,
    ) -> Result<(), Error> {
        (**self).layout(ctx, ct, layout)
    }
}

impl<T: View + ?Sized> View for Arc<T> {
    fn render(
        &self,
        ctx: &ViewContext,
        surf: TerminalSurface<'_>,
        layout: ViewLayout<'_>,
    ) -> Result<(), Error> {
        (**self).render(ctx, surf, layout)
    }

    fn layout(
        &self,
        ctx: &ViewContext,
        ct: BoxConstraint,
        layout: ViewMutLayout<'_>,
    ) -> Result<(), Error> {
        (**self).layout(ctx, ct, layout)
    }
}

#[derive(Clone)]
pub struct ViewDebug<V> {
    view: V,
    size: Size,
}

impl<V: View> std::fmt::Debug for ViewDebug<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ctx = ViewContext::dummy();
        let mut surf = SurfaceOwned::new(self.size);
        surf.draw_check_pattern(
            "fg=#282828,bg=#3c3836"
                .parse()
                .expect("[ViewDebug] failed parse face"),
        );
        surf.draw_view(&ctx, None, &self.view)
            .map_err(|_| std::fmt::Error)?;
        surf.debug().fmt(f)
    }
}

#[derive(Clone)]
pub struct TraceLayout<V, T> {
    view: V,
    trace: T,
}

impl<V, S> View for TraceLayout<V, S>
where
    V: View,
    S: Fn(&BoxConstraint, ViewLayout<'_>) + Send + Sync,
{
    fn render(
        &self,
        ctx: &ViewContext,
        surf: TerminalSurface<'_>,
        layout: ViewLayout<'_>,
    ) -> Result<(), Error> {
        self.view.render(ctx, surf, layout)
    }

    fn layout(
        &self,
        ctx: &ViewContext,
        ct: BoxConstraint,
        mut layout: ViewMutLayout<'_>,
    ) -> Result<(), Error> {
        self.view.layout(ctx, ct, layout.view_mut())?;
        (self.trace)(&ct, layout.view());
        Ok(())
    }
}

/// Something that can be converted to a [View]
pub trait IntoView {
    /// Result view type
    type View: View;

    /// Convert into a [View]
    fn into_view(self) -> Self::View;
}

impl<V: View> IntoView for V {
    type View = V;

    fn into_view(self) -> Self::View {
        self
    }
}

#[derive(Debug, Clone)]
pub struct ViewContext {
    pub(crate) pixels_per_cell: Size,
    pub(crate) has_glyphs: bool,
    pub(crate) color_depth: ColorDepth,
}

impl ViewContext {
    pub fn new(term: &dyn Terminal) -> Result<Self, Error> {
        let caps = term.capabilities();
        Ok(Self {
            pixels_per_cell: term.size()?.pixels_per_cell(),
            has_glyphs: caps.glyphs,
            color_depth: caps.depth,
        })
    }

    /// Dummy view context for debug purposes
    pub fn dummy() -> Self {
        Self {
            pixels_per_cell: Size {
                height: 37,
                width: 15,
            },
            has_glyphs: true,
            color_depth: ColorDepth::TrueColor,
        }
    }

    /// Number of pixels in the single terminal cell
    pub fn pixels_per_cell(&self) -> Size {
        self.pixels_per_cell
    }

    /// Whether terminal supports glyph rendering
    pub fn has_glyphs(&self) -> bool {
        self.has_glyphs
    }

    /// Color depth supported by terminal
    pub fn color_depth(&self) -> ColorDepth {
        self.color_depth
    }
}

/// Constraint that specify size of the view that it can take. Any view when layout
/// should that the size between `min` and `max` sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BoxConstraint {
    min: Size,
    max: Size,
}

impl BoxConstraint {
    /// Create new constraint
    pub fn new(min: Size, max: Size) -> Self {
        Self { min, max }
    }

    /// Constraint min and max equal to size
    pub fn tight(size: Size) -> Self {
        Self {
            min: size,
            max: size,
        }
    }

    /// Constraint with zero min constrain
    pub fn loose(size: Size) -> Self {
        Self {
            min: Size::empty(),
            max: size,
        }
    }

    /// Minimal size
    pub fn min(&self) -> Size {
        self.min
    }

    /// Maximum size
    pub fn max(&self) -> Size {
        self.max
    }

    /// Remove minimal size constraint
    pub fn loosen(&self) -> Self {
        Self {
            min: Size::empty(),
            max: self.max,
        }
    }

    /// Clamp size with constraint
    pub fn clamp(&self, size: Size) -> Size {
        size.clamp(self.min, self.max)
    }
}

/// Major axis of the [Flex] and [ScrollBar] views
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Axis {
    #[serde(rename = "horizontal")]
    Horizontal,
    #[serde(rename = "vertical")]
    Vertical,
}

/// Helper to get values along [Axis]
pub trait AlongAxis {
    /// Axis value
    type Value;

    /// Get value along major axis
    fn major(&self, axis: Axis) -> Self::Value;

    /// Get mutable reference along major axis
    fn major_mut(&mut self, axis: Axis) -> &mut Self::Value;

    /// Get value along minor axis
    fn minor(&self, axis: Axis) -> Self::Value {
        self.major(axis.cross())
    }

    /// Get mutable reference along minor axis
    fn minor_mut(&mut self, axis: Axis) -> &mut Self::Value {
        self.major_mut(axis.cross())
    }

    /// Construct new value given [Axis] and values along major and minor axes
    fn from_axes(axis: Axis, major: Self::Value, minor: Self::Value) -> Self;
}

impl AlongAxis for Size {
    type Value = usize;

    fn major(&self, axis: Axis) -> Self::Value {
        match axis {
            Axis::Horizontal => self.width,
            Axis::Vertical => self.height,
        }
    }

    fn major_mut(&mut self, axis: Axis) -> &mut Self::Value {
        match axis {
            Axis::Horizontal => &mut self.width,
            Axis::Vertical => &mut self.height,
        }
    }

    fn from_axes(axis: Axis, major: Self::Value, minor: Self::Value) -> Self {
        match axis {
            Axis::Horizontal => Size {
                width: major,
                height: minor,
            },
            Axis::Vertical => Size {
                width: minor,
                height: major,
            },
        }
    }
}

impl AlongAxis for Position {
    type Value = usize;

    fn major(&self, axis: Axis) -> Self::Value {
        match axis {
            Axis::Horizontal => self.col,
            Axis::Vertical => self.row,
        }
    }

    fn major_mut(&mut self, axis: Axis) -> &mut Self::Value {
        match axis {
            Axis::Horizontal => &mut self.col,
            Axis::Vertical => &mut self.row,
        }
    }

    fn from_axes(axis: Axis, major: Self::Value, minor: Self::Value) -> Self {
        match axis {
            Axis::Horizontal => Position {
                col: major,
                row: minor,
            },
            Axis::Vertical => Position {
                col: minor,
                row: major,
            },
        }
    }
}

impl Axis {
    /// Flip axis
    pub fn cross(self) -> Self {
        match self {
            Self::Horizontal => Self::Vertical,
            Self::Vertical => Self::Horizontal,
        }
    }

    /// Get major axis value
    pub fn major<T: AlongAxis>(&self, target: T) -> T::Value {
        target.major(*self)
    }

    /// Get minor axis value
    pub fn minor<T: AlongAxis>(&self, target: T) -> T::Value {
        target.minor(*self)
    }

    /// Change constraint along axis
    pub fn constraint(&self, ct: BoxConstraint, min: usize, max: usize) -> BoxConstraint {
        match self {
            Self::Horizontal => BoxConstraint::new(
                Size {
                    height: ct.min().height,
                    width: min,
                },
                Size {
                    height: ct.max().height,
                    width: max,
                },
            ),
            Self::Vertical => BoxConstraint::new(
                Size {
                    height: min,
                    width: ct.min().width,
                },
                Size {
                    height: max,
                    width: ct.max().width,
                },
            ),
        }
    }
}

impl<L, R> View for either::Either<L, R>
where
    L: View,
    R: View,
{
    fn render(
        &self,
        ctx: &ViewContext,
        surf: TerminalSurface<'_>,
        layout: ViewLayout<'_>,
    ) -> Result<(), Error> {
        either::for_both!(self, view => view.render(ctx, surf, layout))
    }

    fn layout(
        &self,
        ctx: &ViewContext,
        ct: BoxConstraint,
        layout: ViewMutLayout<'_>,
    ) -> Result<(), Error> {
        either::for_both!(self, view => view.layout(ctx, ct, layout))
    }
}

impl<V: View> View for Option<V> {
    fn render(
        &self,
        ctx: &ViewContext,
        surf: TerminalSurface<'_>,
        layout: ViewLayout<'_>,
    ) -> Result<(), Error> {
        if let Some(view) = self.as_ref() {
            view.render(ctx, surf, layout)?
        }
        Ok(())
    }

    fn layout(
        &self,
        ctx: &ViewContext,
        ct: BoxConstraint,
        mut layout: ViewMutLayout<'_>,
    ) -> Result<(), Error> {
        match self.as_ref() {
            Some(view) => view.layout(ctx, ct, layout)?,
            None => {
                *layout = Layout::new();
            }
        }
        Ok(())
    }
}

/// View that does not effect rendering only adds tag as `Layout::data`
/// for generated layout.
#[derive(Clone)]
pub struct Tag<T, V> {
    view: V,
    tag: T,
}

impl<T: Clone + Any, V: View> Tag<T, V> {
    pub fn new(tag: T, view: impl IntoView<View = V>) -> Self {
        Self {
            view: view.into_view(),
            tag,
        }
    }
}

fn tag_from_json_value(
    seed: &ViewDeserializer<'_>,
    value: &serde_json::Value,
) -> Result<Tag<serde_json::Value, ArcView<'static>>, Error> {
    let view = value
        .get("view")
        .ok_or_else(|| Error::ParseError("Tag", "must include view attribute".to_owned()))?;
    let tag = value
        .get("tag")
        .ok_or_else(|| Error::ParseError("Tag", "must include tag attribute".to_owned()))?;
    Ok(Tag::new(tag.clone(), seed.deserialize(view)?))
}

impl<T, V> View for Tag<T, V>
where
    T: Clone + Any + Send + Sync,
    V: View,
{
    fn render(
        &self,
        ctx: &ViewContext,
        surf: TerminalSurface<'_>,
        layout: ViewLayout<'_>,
    ) -> Result<(), Error> {
        let surf = layout.apply_to(surf);
        let child_layout = layout.children().next().ok_or(Error::InvalidLayout)?;
        self.view.render(ctx, surf, child_layout)?;
        Ok(())
    }

    fn layout(
        &self,
        ctx: &ViewContext,
        ct: BoxConstraint,
        mut layout: ViewMutLayout<'_>,
    ) -> Result<(), Error> {
        let mut layout_child = layout.push_default();
        self.view.layout(ctx, ct, layout_child.view_mut())?;
        *layout = Layout::new()
            .with_size(layout_child.size())
            .with_data(self.tag.clone());
        Ok(())
    }
}

/// Renders nothing takes all space
impl View for () {
    fn render(
        &self,
        _ctx: &ViewContext,
        _surf: TerminalSurface<'_>,
        _layout: ViewLayout<'_>,
    ) -> Result<(), Error> {
        Ok(())
    }

    fn layout(
        &self,
        _ctx: &ViewContext,
        ct: BoxConstraint,
        mut layout: ViewMutLayout<'_>,
    ) -> Result<(), Error> {
        *layout = Layout::new().with_size(ct.max());
        Ok(())
    }
}

/// Fills with color takes all space
impl View for RGBA {
    fn render(
        &self,
        _ctx: &ViewContext,
        surf: TerminalSurface<'_>,
        layout: ViewLayout<'_>,
    ) -> Result<(), Error> {
        let cell = Cell::new_char(Face::new(None, Some(*self), FaceAttrs::default()), ' ');
        layout.apply_to(surf).fill(cell);
        Ok(())
    }

    fn layout(
        &self,
        _ctx: &ViewContext,
        ct: BoxConstraint,
        mut layout: ViewMutLayout<'_>,
    ) -> Result<(), Error> {
        *layout = Layout::new().with_size(ct.max());
        Ok(())
    }
}

impl View for SurfaceView<'_, Cell> {
    fn render(
        &self,
        _ctx: &ViewContext,
        surf: TerminalSurface<'_>,
        layout: ViewLayout<'_>,
    ) -> Result<(), Error> {
        let src_data = self.data();
        let src_shape = self.shape();

        let mut dst_surf = layout.apply_to(surf);
        let width = dst_surf.width().min(src_shape.width);
        let height = dst_surf.height().min(src_shape.height);
        dst_surf
            .view_mut(..height, ..width)
            .fill_with(|pos, mut dst| {
                let src = src_data[src_shape.offset(pos)].clone();
                dst.overlay(src);
                dst
            });

        Ok(())
    }

    fn layout(
        &self,
        _ctx: &ViewContext,
        ct: BoxConstraint,
        mut layout: ViewMutLayout<'_>,
    ) -> Result<(), Error> {
        *layout = Layout::new().with_size(ct.clamp(self.shape().size()));
        Ok(())
    }
}

/// Passed to `ViewDeserializer` and represents cache, when view with type `ref`
/// is deserialized this cache is used to resolve `uid`s into corresponding view
/// objects.
pub trait ViewCache: Send + Sync {
    fn get(&self, uid: i64) -> Option<ArcView<'static>>;
}

impl<T: ViewCache + ?Sized> ViewCache for Arc<T> {
    fn get(&self, uid: i64) -> Option<ArcView<'static>> {
        (**self).get(uid)
    }
}

#[derive(Clone)]
struct ViewCached {
    cache: Option<Arc<dyn ViewCache>>,
    uid: i64,
}

impl View for ViewCached {
    fn render(
        &self,
        ctx: &ViewContext,
        surf: TerminalSurface<'_>,
        layout: ViewLayout<'_>,
    ) -> Result<(), Error> {
        if let Some(view) = layout.data::<ArcView<'static>>() {
            view.render(ctx, surf, layout.view())?;
        }
        Ok(())
    }

    fn layout(
        &self,
        ctx: &ViewContext,
        ct: BoxConstraint,
        mut layout: ViewMutLayout<'_>,
    ) -> Result<(), Error> {
        if let Some(view) = self.cache.as_ref().and_then(|c| c.get(self.uid)) {
            view.layout(ctx, ct, layout.view_mut())?;
            layout.set_data(view);
        }
        Ok(())
    }
}

/// View deserializer
pub struct ViewDeserializer<'a> {
    colors: &'a HashMap<String, RGBA>,
    view_cache: Option<Arc<dyn ViewCache>>,
    handlers: HashMap<
        String,
        Box<dyn for<'b> Fn(&'b ViewDeserializer<'_>, &'b serde_json::Value) -> ArcView<'static>>,
    >,
}

impl<'a> ViewDeserializer<'a> {
    pub fn new(
        colors: Option<&'a HashMap<String, RGBA>>,
        view_cache: Option<Arc<dyn ViewCache>>,
    ) -> Self {
        Self {
            colors: colors.unwrap_or(&SVG_COLORS),
            view_cache,
            handlers: HashMap::default(),
        }
    }

    pub fn face<'de, D>(&self, deserializer: D) -> Result<Face, D::Error>
    where
        D: Deserializer<'de>,
    {
        FaceDeserializer {
            colors: self.colors,
        }
        .deserialize(deserializer)
    }

    pub fn register<H>(&mut self, name: impl Into<String>, handler: H)
    where
        H: for<'b> Fn(&'b ViewDeserializer<'_>, &'b serde_json::Value) -> ArcView<'static>
            + 'static,
    {
        self.handlers.insert(name.into(), Box::new(handler));
    }
}

impl<'de> de::DeserializeSeed<'de> for &ViewDeserializer<'_> {
    type Value = ArcView<'static>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let Some(view_type) = value.get("type").and_then(Value::as_str) else {
            return Err(de::Error::custom("map expected with \"type\" attribute"));
        };
        let view = match view_type {
            "text" => TextDeserializer {
                colors: self.colors,
            }
            .deserialize(&value)
            .map_err(de::Error::custom)?
            .arc(),
            "trace-layout" => {
                // Generate debug message with calculated constraints and layout
                let view_value = value.get("view").ok_or_else(|| {
                    let err = Error::ParseError(
                        "TracelLayout",
                        "must include `view` attribute".to_owned(),
                    );
                    de::Error::custom(format!("[TraceLayout] {err}"))
                })?;
                let msg = value
                    .get("msg")
                    .and_then(|v| v.as_str())
                    .unwrap_or("trace-layout")
                    .to_owned();
                self.deserialize(view_value)
                    .map(move |view| {
                        view.trace_layout(move |ct, layout| {
                            tracing::debug!(?ct, ?layout, "{}", msg);
                        })
                    })
                    .map_err(|err| de::Error::custom(format!("[TraceLayout] {err}")))?
                    .arc()
            }
            "flex" => Flex::from_json_value(self, &value)
                .map_err(|err| de::Error::custom(format!("[Flex] {err}")))?
                .arc(),
            "container" => container::from_json_value(self, &value)
                .map_err(|err| de::Error::custom(format!("[Container] {err}")))?
                .arc(),
            "glyph" => GlyphDeserializer {
                colors: self.colors,
            }
            .deserialize(value)
            .map_err(|err| de::Error::custom(format!("[Glyph] {err}")))?
            .arc(),
            "image" => Image::deserialize(value)
                .map_err(|err| de::Error::custom(format!("[Image] {err}")))?
                .arc(),
            "image_ascii" => ImageAsciiView::deserialize(value)
                .map_err(|err| de::Error::custom(format!("[ImageAscii] {err}")))?
                .arc(),
            "color" => RGBADeserializer {
                colors: self.colors,
            }
            .deserialize(value)
            .map_err(|err| de::Error::custom(format!("[RGBA] {err}")))?
            .arc(),
            "tag" => tag_from_json_value(self, &value)
                .map_err(|err| de::Error::custom(format!("[Tag] {err}")))?
                .arc(),
            "ref" => {
                let uid = value.get("ref").and_then(|v| v.as_i64()).ok_or_else(|| {
                    de::Error::custom("[ref] must include `uid: u64` attribute".to_string())
                })?;
                ViewCached {
                    cache: self.view_cache.clone(),
                    uid,
                }
                .arc()
            }
            name => match self.handlers.get(name) {
                None => return Err(de::Error::custom(format!("unknown view type: {name}"))),
                Some(handler) => handler(self, &value),
            },
        };
        Ok(view)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn render(
        ctx: &ViewContext,
        view: &dyn View,
        size: Size,
    ) -> Result<SurfaceOwned<Cell>, Error> {
        let mut surf = SurfaceOwned::new(size);

        let mut layout_store = ViewLayoutStore::new();
        let layout = view.layout_new(ctx, BoxConstraint::loose(size), &mut layout_store)?;

        view.render(ctx, surf.as_mut(), layout.view())?;
        Ok(surf)
    }

    #[test]
    fn test_references() -> Result<(), Error> {
        fn witness<V: View>(_: V) {
            println!("{}", std::any::type_name::<V>());
        }
        let color = "#ff0000".parse::<RGBA>()?;
        let color_boxed = color.boxed();
        let color_arc: ArcView<'static> = Arc::new(color);

        witness(color);
        witness(color);
        witness(&color as &dyn View);
        witness(&color_boxed);
        witness(color_boxed);
        witness(&color_arc);
        witness(color_arc);

        Ok(())
    }

    #[test]
    fn test_box_constraint() -> Result<(), Error> {
        let ct = BoxConstraint::new(Size::new(10, 0), Size::new(10, 100));
        let size = Size::new(1, 4);
        assert_eq!(ct.clamp(size), Size::new(10, 4));
        Ok(())
    }

    #[test]
    fn test_view_deserialize() -> Result<(), Error> {
        let view_value = serde_json::json!({
            "type": "flex",
            "direction": "vertical",
            "justify": "center",
            "children": [
                {
                    "face": "bg=#ff0000/.2",
                    "view": {
                        "type": "container",
                        "horizontal": "center",
                        "child": {
                            "type": "text",
                            "text": {
                                "face": "bg=white,fg=black,bold",
                                "text": [
                                    {
                                        "glyph": {
                                            "size": [1, 3],
                                            "view_box": [0, 0, 128, 128],
                                            "path": "M20.33 68.63L20.33 68.63L107.67 68.63Q107.26 73.17 106.02 77.49L106.02 77.49L72.86 77.49L72.86 86.35L86.04 86.35L86.04 95.00L77.18 95.00L77.18 103.86L66.27 103.86L66.27 108.18L64 108.18Q52.88 108.18 43.20 102.93Q33.51 97.68 27.44 88.72Q21.36 79.76 20.33 68.63ZM107.67 59.98L107.67 59.98L20.33 59.98Q21.36 48.86 27.44 39.90Q33.51 30.94 43.20 25.69Q52.88 20.43 64 20.43L64 20.43Q74.51 20.43 83.77 25.17L83.77 25.17L83.77 33.62L92.63 33.62L92.63 42.27L99.22 42.27L99.22 51.13L106.02 51.13Q107.26 55.45 107.67 59.98ZM64 41.24L64 41.24Q64 36.71 60.81 33.51Q57.61 30.32 53.08 30.32Q48.55 30.32 45.26 33.51Q41.96 36.71 41.96 41.24Q41.96 45.77 45.26 48.96Q48.55 52.16 53.08 52.16Q57.61 52.16 60.81 48.96Q64 45.77 64 41.24Z",
                                        }
                                    },
                                    "Space Invaders "
                                ]
                            }
                        }
                    }
                },
                {
                    "align": "center",
                    "face": "bg=#00ff00/.05",
                    "view": {
                        "type": "image_ascii",
                        "data": "AAAAAAAAAAAAAAAAAAAAAP8AAAAAAP8AAAAAAAAA/wAAAP8AAAAAAAAA/////////wAAAAAA//8A////AP//AAAA//////////////8AAP8A/////////wD/AAD/AP8AAAAAAP8A/wAAAAAA//8A//8AAAAAAAAAAAAAAAAAAAAAAA==",
                        "channels": 1,
                        "size": [10, 13],
                    }
                }
            ]
        });

        let view = ViewDeserializer::new(None, None).deserialize(view_value)?;
        let size = Size::new(10, 20);
        println!("[view] deserialize: {:?}", view.debug(size));
        let mut layout_store = ViewLayoutStore::new();
        let layout = view.layout_new(
            &ViewContext::dummy(),
            BoxConstraint::loose(size),
            &mut layout_store,
        )?;
        let mut reference_store = ViewLayoutStore::new();
        let mut reference = ViewMutLayout::new(
            &mut reference_store,
            Layout::new().with_size(Size::new(8, 20)),
        );
        reference
            .push(
                Layout::new()
                    .with_position(Position::new(2, 0))
                    .with_size(Size::new(1, 20)),
            )
            .push(
                Layout::new()
                    .with_position(Position::new(0, 1))
                    .with_size(Size::new(1, 18)),
            );
        reference.push(
            Layout::new()
                .with_position(Position::new(3, 3))
                .with_size(Size::new(5, 13)),
        );
        assert_eq!(reference, layout);

        Ok(())
    }
}
