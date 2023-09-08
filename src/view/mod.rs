//! Defines [View] that represents anything that can be rendered into a terminal.
//! As well as some useful implementations such as [Text], [Flex], [Container], ...
mod container;
pub use container::{Align, Container, Margins};

mod flex;
pub use flex::{Flex, Justify};

mod scrollbar;
use rasterize::{RGBADeserializer, SVG_COLORS};
pub use scrollbar::ScrollBar;

mod text;
use serde_json::Value;
pub use text::Text;

mod dynamic;
pub use dynamic::Dynamic;

mod frame;
pub use frame::Frame;

use crate::{
    encoder::ColorDepth, image::ImageAsciiView, Cell, Error, Face, FaceAttrs, FaceDeserializer,
    Glyph, Image, Position, Size, SurfaceMut, SurfaceOwned, Terminal, TerminalSurface,
    TerminalSurfaceExt, RGBA,
};
use serde::{
    de::{self, DeserializeSeed},
    Deserialize, Deserializer, Serialize,
};
use std::{
    any::Any,
    collections::HashMap,
    fmt::Debug,
    ops::{Deref, DerefMut, Index, IndexMut},
};

use self::text::TextDeserializer;

/// View is anything that can be layed out and rendered to the terminal
pub trait View {
    /// Render view into a given surface with the provided layout
    fn render<'a>(
        &self,
        ctx: &ViewContext,
        surf: &'a mut TerminalSurface<'a>,
        layout: &Tree<Layout>,
    ) -> Result<(), Error>;

    /// Compute layout of the view based on the constraints
    fn layout(&self, ctx: &ViewContext, ct: BoxConstraint) -> Tree<Layout>;

    /// Convert into boxed view
    fn boxed<'a>(self) -> Box<dyn View + 'a>
    where
        Self: Sized + 'a,
    {
        Box::new(self)
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
        T: Fn(&BoxConstraint, &Tree<Layout>),
        Self: Sized,
    {
        TraceLayout { view: self, trace }
    }
}

impl<'a, V: View + ?Sized> View for &'a V {
    fn render<'b>(
        &self,
        ctx: &ViewContext,
        surf: &'b mut TerminalSurface<'b>,
        layout: &Tree<Layout>,
    ) -> Result<(), Error> {
        (**self).render(ctx, surf, layout)
    }

    fn layout(&self, ctx: &ViewContext, ct: BoxConstraint) -> Tree<Layout> {
        (**self).layout(ctx, ct)
    }
}

impl<'a> View for Box<dyn View + 'a> {
    fn render<'b>(
        &self,
        ctx: &ViewContext,
        surf: &'b mut TerminalSurface<'b>,
        layout: &Tree<Layout>,
    ) -> Result<(), Error> {
        (**self).render(ctx, surf, layout)
    }

    fn layout(&self, ctx: &ViewContext, ct: BoxConstraint) -> Tree<Layout> {
        (**self).layout(ctx, ct)
    }

    fn boxed<'b>(self) -> Box<dyn View + 'b>
    where
        Self: Sized + 'b,
    {
        self
    }
}

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
        surf.draw_view(&ctx, &self.view)
            .map_err(|_| std::fmt::Error)?;
        surf.debug().fmt(f)
    }
}

pub struct TraceLayout<V, T> {
    view: V,
    trace: T,
}

impl<V, S> View for TraceLayout<V, S>
where
    V: View,
    S: Fn(&BoxConstraint, &Tree<Layout>),
{
    fn render<'a>(
        &self,
        ctx: &ViewContext,
        surf: &'a mut TerminalSurface<'a>,
        layout: &Tree<Layout>,
    ) -> Result<(), Error> {
        self.view.render(ctx, surf, layout)
    }

    fn layout(&self, ctx: &ViewContext, ct: BoxConstraint) -> Tree<Layout> {
        let layout = self.view.layout(ctx, ct);
        (self.trace)(&ct, &layout);
        layout
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

/// Layout of the [View] determines its position and size
#[derive(Default)]
pub struct Layout {
    pos: Position,
    size: Size,
    data: Option<Box<dyn Any>>,
}

impl std::cmp::PartialEq for Layout {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos
            && self.size == other.size
            && self.data.is_none()
            && other.data.is_none()
    }
}

impl std::cmp::Eq for Layout {}

impl Debug for Layout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Layout")
            .field("row", &self.pos.row)
            .field("col", &self.pos.col)
            .field("height", &self.size.height)
            .field("width", &self.size.width)
            .finish()
    }
}

impl Layout {
    /// Create a new empty layout
    pub fn new() -> Self {
        Self::default()
    }

    /// Override layout position
    pub fn with_position(self, pos: Position) -> Self {
        Self { pos, ..self }
    }

    /// Get layout position
    pub fn pos(&self) -> Position {
        self.pos
    }

    /// Set layout position
    pub fn set_pos(&mut self, pos: Position) -> &mut Self {
        self.pos = pos;
        self
    }

    /// Override layout size
    pub fn with_size(self, size: Size) -> Self {
        Self { size, ..self }
    }

    /// Get layout size
    pub fn size(&self) -> Size {
        self.size
    }

    /// Set layout size
    pub fn set_size(&mut self, size: Size) -> &mut Self {
        self.size = size;
        self
    }

    /// Get layout data
    pub fn data<T: Any>(&self) -> Option<&T> {
        self.data.as_ref()?.downcast_ref()
    }

    /// Set layout data
    pub fn set_data(&mut self, data: impl Any) -> &mut Self {
        self.data = Some(Box::new(data));
        self
    }

    /// Constrain surface by the layout, that is create sub-subsurface view
    /// with offset `pos` and size of `size`.
    pub fn apply_to<'a, S>(&self, surf: &'a mut S) -> TerminalSurface<'a>
    where
        S: SurfaceMut<Item = Cell>,
    {
        surf.view_mut(
            self.pos.row..self.pos.row + self.size.height,
            self.pos.col..self.pos.col + self.size.width,
        )
    }
}

#[derive(PartialEq, Eq, Hash, Default)]
pub struct Tree<T> {
    pub value: T,
    pub children: Vec<Tree<T>>,
}

impl<T> Tree<T> {
    /// Construct a new node
    pub fn new(value: T, children: Vec<Tree<T>>) -> Self {
        Self { value, children }
    }

    /// Construct a new leaf node
    pub fn leaf(value: T) -> Self {
        Self {
            value,
            children: Vec::new(),
        }
    }

    /// Add child
    pub fn push(&mut self, child: Tree<T>) {
        self.children.push(child)
    }

    /// Get child by its index
    pub fn get(&self, index: usize) -> Option<&Tree<T>> {
        self.children.get(index)
    }
}

impl<T: Debug> Debug for Tree<T> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn debug_rec<T: Debug>(
            this: &Tree<T>,
            offset: usize,
            fmt: &mut std::fmt::Formatter<'_>,
        ) -> std::fmt::Result {
            writeln!(fmt, "{0:<1$}{2:?}", "", offset, this.value)?;
            for child in this.children.iter() {
                debug_rec(child, offset + 2, fmt)?;
            }
            Ok(())
        }
        writeln!(fmt)?;
        debug_rec(self, 0, fmt)?;
        Ok(())
    }
}

impl<T> Deref for Tree<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T> DerefMut for Tree<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

impl<T> Index<usize> for Tree<T> {
    type Output = Tree<T>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.children[index]
    }
}

impl<T> IndexMut<usize> for Tree<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.children[index]
    }
}

impl Tree<Layout> {
    /// Find path in the layout that leads to the position.
    pub fn find_path(&self, pos: Position) -> FindPath<'_> {
        FindPath {
            tree: Some(self),
            pos,
        }
    }
}

pub struct FindPath<'a> {
    tree: Option<&'a Tree<Layout>>,
    pos: Position,
}

impl<'a> Iterator for FindPath<'a> {
    type Item = &'a Layout;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.tree.take()?;
        for child in result.children.iter() {
            if child.pos.col <= self.pos.col
                && self.pos.col < child.pos.col + child.size.width
                && child.pos.row <= self.pos.row
                && self.pos.row < child.pos.row + child.size.height
            {
                self.pos = Position {
                    row: self.pos.row - child.pos.row,
                    col: self.pos.col - child.pos.col,
                };
                self.tree.replace(child);
                break;
            }
        }
        Some(result)
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

/// View that does not effect rendering only adds tag as `Layout::data`
/// for generated layout.
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
) -> Result<Tag<serde_json::Value, Box<dyn View>>, Error> {
    let view = value
        .get("view")
        .ok_or_else(|| Error::ParseError("Tag", "must include view attribute".to_owned()))?;
    let tag = value
        .get("tag")
        .ok_or_else(|| Error::ParseError("Tag", "must include tag attribute".to_owned()))?;
    Ok(Tag::new(tag.clone(), seed.deserialize(view)?))
}

impl<T: Clone + Any, V: View> View for Tag<T, V> {
    fn render<'a>(
        &self,
        ctx: &ViewContext,
        surf: &'a mut TerminalSurface<'a>,
        layout: &Tree<Layout>,
    ) -> Result<(), Error> {
        self.view.render(ctx, surf, layout)
    }

    fn layout(&self, ctx: &ViewContext, ct: BoxConstraint) -> Tree<Layout> {
        let mut layout = self.view.layout(ctx, ct);
        layout.set_data(self.tag.clone());
        layout
    }
}

/// Renders nothing takes all space
impl View for () {
    fn render<'a>(
        &self,
        _ctx: &ViewContext,
        _surf: &'a mut TerminalSurface<'a>,
        _layout: &Tree<Layout>,
    ) -> Result<(), Error> {
        Ok(())
    }

    fn layout(&self, _ctx: &ViewContext, ct: BoxConstraint) -> Tree<Layout> {
        Tree::leaf(Layout::new().with_size(ct.max()))
    }
}

/// Fills with color takes all space
impl View for RGBA {
    fn render<'a>(
        &self,
        _ctx: &ViewContext,
        surf: &'a mut TerminalSurface<'a>,
        layout: &Tree<Layout>,
    ) -> Result<(), Error> {
        let cell = Cell::new_char(Face::new(None, Some(*self), FaceAttrs::default()), ' ');
        layout.apply_to(surf).fill(cell);
        Ok(())
    }

    fn layout(&self, _ctx: &ViewContext, ct: BoxConstraint) -> Tree<Layout> {
        Tree::leaf(Layout::new().with_size(ct.max()))
    }
}

/// View deserializer
pub struct ViewDeserializer<'a> {
    colors: &'a HashMap<String, RGBA>,
    handlers: HashMap<
        String,
        Box<dyn for<'b> Fn(&'b ViewDeserializer<'_>, &'b serde_json::Value) -> Box<dyn View>>,
    >,
}

impl<'a> ViewDeserializer<'a> {
    pub fn new(colors: Option<&'a HashMap<String, RGBA>>) -> Self {
        Self {
            colors: colors.unwrap_or(&SVG_COLORS),
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
        H: for<'b> Fn(&'b ViewDeserializer<'_>, &'b serde_json::Value) -> Box<dyn View> + 'static,
    {
        self.handlers.insert(name.into(), Box::new(handler));
    }
}

impl<'de, 'a> de::DeserializeSeed<'de> for &'a ViewDeserializer<'_> {
    type Value = Box<dyn View>;

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
            .boxed(),
            "flex" => Flex::from_json_value(&self, &value)
                .map_err(|err| de::Error::custom(format!("[Flex] {err}")))?
                .boxed(),
            "container" => container::from_json_value(&self, &value)
                .map_err(|err| de::Error::custom(format!("[Container] {err}")))?
                .boxed(),
            "glyph" => Glyph::deserialize(value)
                .map_err(|err| de::Error::custom(format!("[Glyph] {err}")))?
                .boxed(),
            "image" => Image::deserialize(value)
                .map_err(|err| de::Error::custom(format!("[Image] {err}")))?
                .boxed(),
            "image_ascii" => ImageAsciiView::deserialize(value)
                .map_err(|err| de::Error::custom(format!("[ImageAscii] {err}")))?
                .boxed(),
            "color" => RGBADeserializer {
                colors: self.colors,
            }
            .deserialize(value)
            .map_err(|err| de::Error::custom(format!("[RGBA] {err}")))?
            .boxed(),
            "tag" => tag_from_json_value(self, &value)
                .map_err(|err| de::Error::custom(format!("[Tag] {err}")))?
                .boxed(),
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
        let layout = view.layout(ctx, BoxConstraint::loose(size));
        let mut surf = SurfaceOwned::new(size);
        view.render(ctx, &mut surf.view_mut(.., ..), &layout)?;
        Ok(surf)
    }

    #[test]
    fn test_references() -> Result<(), Error> {
        fn witness<V: View>(_: V) {
            println!("{}", std::any::type_name::<V>());
        }
        let color = "#ff0000".parse::<RGBA>()?;
        let color_boxed = color.boxed();

        witness(color);
        witness(&color);
        witness(&color as &dyn View);
        witness(&color_boxed);
        witness(color_boxed);

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
    fn test_layout_find_path() {
        let layout = Tree::new(
            Layout::new().with_size(Size::new(10, 10)),
            vec![
                Tree::leaf(Layout::new().with_size(Size::new(6, 6))),
                Tree::leaf(
                    Layout::new()
                        .with_position(Position::new(6, 0))
                        .with_size(Size::new(4, 5)),
                ),
                Tree::leaf(
                    Layout::new()
                        .with_position(Position::new(6, 5))
                        .with_size(Size::new(4, 5)),
                ),
                Tree::new(
                    Layout::new()
                        .with_position(Position::new(0, 6))
                        .with_size(Size::new(6, 4)),
                    vec![
                        Tree::leaf(Layout::new().with_size(Size::new(3, 4))),
                        Tree::leaf(
                            Layout::new()
                                .with_position(Position::new(3, 0))
                                .with_size(Size::new(3, 4)),
                        ),
                    ],
                ),
            ],
        );

        assert_eq!(
            layout.find_path(Position::new(4, 7)).collect::<Vec<_>>(),
            vec![
                &Layout::new().with_size(Size::new(10, 10)),
                &Layout::new()
                    .with_position(Position::new(0, 6))
                    .with_size(Size::new(6, 4)),
                &Layout::new()
                    .with_position(Position::new(3, 0))
                    .with_size(Size::new(3, 4)),
            ]
        );
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
                        "data": "eJxjYEAF/5FICA1nQwCUDeIygHn/EQCkGMaEcKCmIOliQAcA3jAt0w==",
                        "channels": 1,
                        "size": [10, 13],
                    }
                }
            ]
        });

        let view = ViewDeserializer::new(None).deserialize(view_value)?;
        let size = Size::new(10, 20);
        println!("[view] deserialize: {:?}", view.debug(size));
        let layout = view.layout(&ViewContext::dummy(), BoxConstraint::loose(size));
        assert_eq!(
            layout,
            Tree::new(
                Layout::new().with_size(Size::new(8, 20)),
                vec![
                    Tree::new(
                        Layout::new()
                            .with_position(Position::new(2, 0))
                            .with_size(Size::new(1, 20)),
                        vec![Tree::leaf(
                            Layout::new()
                                .with_position(Position::new(0, 1))
                                .with_size(Size::new(1, 18))
                        )]
                    ),
                    Tree::leaf(
                        Layout::new()
                            .with_position(Position::new(3, 3))
                            .with_size(Size::new(5, 13))
                    )
                ]
            )
        );

        Ok(())
    }
}
