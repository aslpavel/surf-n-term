use crate::{
    Cell, Error, Face, FaceAttrs, Position, Size, SurfaceMut, SurfaceOwned, TerminalSurface,
    TerminalSurfaceExt, TerminalWriter, RGBA,
};
use std::{
    borrow::Cow,
    cmp::{max, min},
    fmt::Debug,
    io::Write,
    ops::{Deref, DerefMut, Index, IndexMut},
};

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

#[derive(Debug, Clone, Copy, Default)]
pub struct Layout {
    pub pos: Position,
    pub size: Size,
}

impl Layout {
    /// Constrain surface by the layout
    /*

    pub fn view<'surf, S>(&self, mut surf: S) -> TerminalSurface<'surf>
    where
        S: SurfaceMut<Item = Cell> + 'surf,
    {
        surf.view_mut(
            self.pos.row..self.pos.row + self.size.height,
            self.pos.col..self.pos.col + self.size.width,
        )
    }
     */

    pub fn view<'a>(&self, surf: &'a mut TerminalSurface<'a>) -> TerminalSurface<'a> {
        surf.view_mut(
            self.pos.row..self.pos.row + self.size.height,
            self.pos.col..self.pos.col + self.size.width,
        )
    }
}

pub trait Widget: Debug {
    /// Render widget into a given surface with the provided layout
    fn render<'a>(
        &self,
        surf: &'a mut TerminalSurface<'a>,
        layout: &Tree<Layout>,
    ) -> Result<(), Error>;

    /// Compute layout of the widget based on the constraints
    fn layout(&self, ct: BoxConstraint) -> Tree<Layout>;

    /// Wrapper around widget that implements [std::fmt::Debug] which renders
    /// widget. Only supposed to be used for debugging.
    fn debug(&self, size: Size) -> WidgetPreview<&'_ Self>
    where
        Self: Sized,
    {
        WidgetPreview { widget: self, size }
    }
}

pub struct WidgetPreview<W> {
    widget: W,
    size: Size,
}

impl<W: Widget> std::fmt::Debug for WidgetPreview<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut surf = SurfaceOwned::new(self.size.height, self.size.width);
        let layout = self.widget.layout(BoxConstraint::loose(self.size));
        self.widget
            .render(&mut surf.view_mut(.., ..), &layout)
            .map_err(|_| std::fmt::Error)?;
        surf.debug().fmt(f)
    }
}

impl<'a, T: Widget + ?Sized> Widget for &'a T {
    fn render<'b>(
        &self,
        surf: &'b mut TerminalSurface<'b>,
        layout: &Tree<Layout>,
    ) -> Result<(), Error> {
        (**self).render(surf, layout)
    }

    fn layout(&self, ct: BoxConstraint) -> Tree<Layout> {
        (**self).layout(ct)
    }
}

impl<'a> Widget for Box<dyn Widget + 'a> {
    fn render<'b>(
        &self,
        surf: &'b mut TerminalSurface<'b>,
        layout: &Tree<Layout>,
    ) -> Result<(), Error> {
        (**self).render(surf, layout)
    }

    fn layout(&self, ct: BoxConstraint) -> Tree<Layout> {
        (**self).layout(ct)
    }
}

#[derive(Default, Debug)]
pub struct Tree<T> {
    pub value: T,
    pub children: Vec<Tree<T>>,
}

impl<T> Tree<T> {
    pub fn new(value: T, children: Vec<Tree<T>>) -> Self {
        Self { value, children }
    }

    pub fn push(&mut self, child: Tree<T>) {
        self.children.push(child)
    }

    pub fn get(&self, index: usize) -> Option<&Tree<T>> {
        self.children.get(index)
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Align {
    Start,
    Center,
    End,
}

impl Align {
    pub fn align(&self, leftover: usize) -> usize {
        match self {
            Self::Start => 0,
            Self::Center => leftover / 2,
            Self::End => leftover,
        }
    }
}

impl Default for Align {
    fn default() -> Self {
        Self::Center
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Axis {
    Horizontal,
    Vertical,
}

pub trait AlongAxis {
    type Value;

    fn major(&self, axis: Axis) -> Self::Value;
    fn major_mut(&mut self, axis: Axis) -> &mut Self::Value;

    fn minor(&self, axis: Axis) -> Self::Value {
        self.major(axis.cross())
    }
    fn minor_mut(&mut self, axis: Axis) -> &mut Self::Value {
        self.major_mut(axis.cross())
    }
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

    /// Create [Size] given the value for major and minor axis
    pub fn size(&self, major: usize, minor: usize) -> Size {
        match self {
            Self::Horizontal => Size {
                width: major,
                height: minor,
            },
            Self::Vertical => Size {
                width: minor,
                height: major,
            },
        }
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

#[derive(Debug)]
enum Child<'a> {
    Fixed {
        widget: Box<dyn Widget + 'a>,
    },
    Flex {
        widget: Box<dyn Widget + 'a>,
        flex: f64,
    },
}

impl<'a> Child<'a> {
    fn _widget(&self) -> &dyn Widget {
        match self {
            Self::Fixed { widget, .. } => &*widget,
            Self::Flex { widget, .. } => &*widget,
        }
    }
}

#[derive(Debug)]
pub struct Flex<'a> {
    direction: Axis,
    children: Vec<Child<'a>>,
}

impl<'a> Flex<'a> {
    /// Create new flex widget aligned along direction [Axis]
    pub fn new(direction: Axis) -> Self {
        Self {
            direction,
            children: Default::default(),
        }
    }

    pub fn row() -> Self {
        Self::new(Axis::Horizontal)
    }

    pub fn column() -> Self {
        Self::new(Axis::Vertical)
    }

    /// Add new fixed size child
    pub fn add_child(mut self, child: impl Widget + 'a) -> Self {
        self.children.push(Child::Fixed {
            widget: Box::new(child),
        });
        self
    }

    /// Add new flex size child
    pub fn add_flex_child(mut self, flex: f64, child: impl Widget + 'a) -> Self {
        if flex > 0.0 {
            self.children.push(Child::Flex {
                widget: Box::new(child),
                flex,
            });
            self
        } else {
            self.add_child(child)
        }
    }
}

impl<'a> Widget for Flex<'a>
where
    Self: 'a,
{
    fn render<'b>(
        &self,
        _surf: &'b mut TerminalSurface<'b>,
        _layout: &Tree<Layout>,
    ) -> Result<(), Error> {
        /*
        for (index, child) in self.children.iter().enumerate() {
            let child_layout = layout.get(index).ok_or(Error::InvalidLayout)?;
            {
                let surf = child_layout.view(surf);
                child.widget().render(&mut surf, layout)?;
            }
        }
        */
        Ok(())
    }

    fn layout(&self, ct: BoxConstraint) -> Tree<Layout> {
        let mut children_layout = Vec::new();
        children_layout.resize_with(self.children.len(), Tree::<Layout>::default);

        let mut flex_total = 0.0;
        let mut major_non_flex = 0;
        let mut minor = self.direction.minor(ct.min());
        let ct_loosen = ct.loosen();

        // layout non-flex
        for (index, child) in self.children.iter().enumerate() {
            match child {
                Child::Fixed { widget } => {
                    let child_layout = widget.layout(ct_loosen);
                    major_non_flex += self.direction.major(child_layout.size);
                    minor = max(minor, self.direction.minor(child_layout.size));
                    children_layout[index] = child_layout;
                }
                Child::Flex { flex, .. } => {
                    flex_total += flex;
                }
            }
        }

        // calculate available space for flex widgets
        let major_total = self.direction.major(ct.max());
        major_non_flex = min(major_total, major_non_flex);
        let major_remain = major_total - major_non_flex;

        // layout flex
        let mut major_flex = 0;
        if major_remain > 0 {
            let per_flex = (major_remain as f64) / flex_total;
            for (index, child) in self.children.iter().enumerate() {
                match child {
                    Child::Flex { widget, flex } => {
                        let child_major = (flex * per_flex).round() as usize;
                        if child_major == 0 {
                            continue;
                        }
                        let child_ct = self.direction.constraint(ct_loosen, 0, child_major);
                        let child_layout = widget.layout(child_ct);

                        major_flex += self.direction.major(child_layout.size);
                        minor = max(minor, self.direction.minor(child_layout.size));
                        children_layout[index] = child_layout;
                    }
                    _ => {}
                }
            }
        }

        // extra space to be filled
        let _extra = self.direction.major(ct.max()) - (major_non_flex + major_flex);

        // calculate offsets
        let mut offset = 0;
        for index in 0..self.children.len() {
            let child_layout = &mut children_layout[index];
            *child_layout.pos.major_mut(self.direction) = offset;
            offset += child_layout.size.major(self.direction);
        }

        // create layout tree
        Tree::new(
            Layout {
                pos: Default::default(),
                size: ct.clamp(self.direction.size(offset, minor)),
            },
            children_layout,
        )
    }
}

#[derive(Debug)]
pub struct Container<W> {
    widget: W,
    _color: Option<RGBA>,
    _align_vertical: Align,
    _align_horizontal: Align,
    size: Size,
}

impl<W: Widget> Container<W> {
    pub fn new(widget: W) -> Self {
        Self {
            size: Size::empty(),
            _color: None,
            _align_vertical: Align::default(),
            _align_horizontal: Align::default(),
            widget,
        }
    }

    pub fn width(self, width: usize) -> Self {
        Self {
            size: Size { width, ..self.size },
            ..self
        }
    }

    pub fn height(self, height: usize) -> Self {
        Self {
            size: Size {
                height,
                ..self.size
            },
            ..self
        }
    }
}

impl<W: Widget> Widget for Container<W> {
    fn render<'a>(
        &self,
        _surf: &'a mut TerminalSurface<'a>,
        _layout: &Tree<Layout>,
    ) -> Result<(), Error> {
        todo!()
    }

    fn layout(&self, ct: BoxConstraint) -> Tree<Layout> {
        let width_max = if self.size.width == 0 {
            ct.max().width
        } else {
            self.size.width.clamp(ct.min().width, ct.max().width)
        };
        let height_max = if self.size.height == 0 {
            ct.max().height
        } else {
            self.size.height.clamp(ct.min().height, ct.max().height)
        };
        let widget_ct = BoxConstraint::new(Size::empty(), Size::new(height_max, width_max));
        let _widget_layout = self.widget.layout(widget_ct);
        todo!()
    }
}

#[derive(Debug)]
pub struct Fixed<W> {
    widget: W,
    size: Size,
}

impl<W> Fixed<W> {
    pub fn new(size: Size, widget: W) -> Self {
        Self { widget, size }
    }
}

impl<W: Widget> Widget for Fixed<W> {
    fn render<'a>(
        &self,
        surf: &'a mut TerminalSurface<'a>,
        layout: &Tree<Layout>,
    ) -> Result<(), Error> {
        self.widget.render(surf, layout)
    }

    fn layout(&self, ct: BoxConstraint) -> Tree<Layout> {
        let size = ct.clamp(self.size);
        self.widget.layout(BoxConstraint::tight(size))
    }
}

impl Widget for RGBA {
    fn render<'a>(
        &self,
        surf: &'a mut TerminalSurface<'a>,
        layout: &Tree<Layout>,
    ) -> Result<(), Error> {
        let cell = Cell::new(Face::new(None, Some(*self), FaceAttrs::default()), None);
        layout.view(surf).fill(cell);
        Ok(())
    }

    fn layout(&self, ct: BoxConstraint) -> Tree<Layout> {
        Tree::new(
            Layout {
                pos: Position::origin(),
                size: ct.min(),
            },
            Vec::new(),
        )
    }
}

#[derive(Debug)]
pub struct Text<'a> {
    text: Cow<'a, str>,
    face: Face,
    children: Vec<Text<'a>>,
}

impl<'a> Text<'a> {
    pub fn new(text: impl Into<Cow<'a, str>>) -> Self {
        Self {
            text: text.into(),
            face: Face::default(),
            children: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.children
            .iter()
            .fold(self.text.len(), |len, child| len + child.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn with_face(self, face: Face) -> Self {
        Self { face, ..self }
    }

    pub fn add_text(mut self, text: impl Into<Text<'a>>) -> Self {
        self.children.push(text.into());
        self
    }
}

impl<'a> Widget for Text<'a> {
    fn render<'b>(
        &self,
        surf: &'b mut TerminalSurface<'b>,
        layout: &Tree<Layout>,
    ) -> Result<(), Error> {
        fn render_rec<'a>(
            writer: &mut TerminalWriter<'_>,
            face: &Face,
            this: &Text<'a>,
        ) -> Result<(), Error> {
            writer.face_set(face.overlay(&this.face));
            write!(writer, "{}", this.text.as_ref())?;
            for child in this.children.iter() {
                render_rec(writer, &this.face, child)?;
            }
            writer.face_set(*face);
            Ok(())
        }

        let mut surf = layout.view(surf);
        render_rec(&mut surf.writer(), &Face::default(), self)
    }

    fn layout(&self, ct: BoxConstraint) -> Tree<Layout> {
        fn chars<'a>(this: &'a Text<'a>) -> Box<dyn Iterator<Item = char> + 'a> {
            // TODO: implement FIFO based depth first iterator to reduce allocations?
            let iter = this
                .text
                .as_ref()
                .chars()
                .chain(this.children.iter().flat_map(chars));
            Box::new(iter)
        }
        Tree::new(layout_chars(ct, chars(self)), Vec::new())
    }
}

impl<'a, 'b: 'a> From<&'b str> for Text<'a> {
    fn from(string: &'b str) -> Self {
        Text::new(string)
    }
}

impl<'a> From<String> for Text<'a> {
    fn from(string: String) -> Self {
        Text::new(string)
    }
}

/// Layout iterator of chars
pub fn layout_chars(ct: BoxConstraint, chrs: impl IntoIterator<Item = char>) -> Layout {
    let mut size = Size::empty();
    let mut col = 0;
    for chr in chrs {
        match chr {
            '\r' => {}
            '\n' => {
                size.width = max(size.width, col);
                size.height += 1;
                col = 0;
            }
            _ => {
                if col < ct.max().width {
                    col += 1;
                } else {
                    size.width = max(size.width, col);
                    size.height += 1;
                    col = 1;
                }
            }
        }
    }
    size.width = max(size.width, col);
    if size.height != 0 || size.width != 0 {
        size.height += 1;
    }
    Layout {
        pos: Position::origin(),
        size,
    }
}

impl<'a> Widget for &'a str {
    fn render<'b>(
        &self,
        surf: &'b mut TerminalSurface<'b>,
        layout: &Tree<Layout>,
    ) -> Result<(), Error> {
        Text::new(*self).render(surf, layout)
    }

    fn layout(&self, ct: BoxConstraint) -> Tree<Layout> {
        Text::new(*self).layout(ct)
    }
}

impl Widget for String {
    fn render<'a>(
        &self,
        surf: &'a mut TerminalSurface<'a>,
        layout: &Tree<Layout>,
    ) -> Result<(), Error> {
        self.as_str().render(surf, layout)
    }

    fn layout(&self, ct: BoxConstraint) -> Tree<Layout> {
        self.as_str().layout(ct)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_references() -> Result<(), Error> {
        fn witness(_: impl Widget) {}
        let color = "#ff0000".parse::<RGBA>()?;

        witness(color);
        witness(&color);
        witness(&color as &dyn Widget);
        witness(Box::new(color) as Box<dyn Widget>);

        Ok(())
    }

    #[test]
    fn test_text() -> Result<(), Error> {
        let two = "two".to_string();
        let text = Text::new("one ")
            .with_face("fg=#3c3836,bg=#ebdbb2".parse()?)
            .add_text(Text::new(two.as_str()).with_face("fg=#af3a03,bold".parse()?))
            .add_text(" three".to_string())
            .add_text("\nfour")
            .add_text("");

        let size = Size::new(5, 10);
        println!("{:?}", text.debug(size));

        let layout = text.layout(BoxConstraint::loose(size));
        assert_eq!(layout.size, Size::new(3, 10));

        Ok(())
    }

    #[test]
    fn test_fixed() -> Result<(), Error> {
        let fixed = Fixed::new(Size::new(3, 5), "#ff0000".parse::<RGBA>()?);
        println!("{:?}", fixed.debug(Size::new(5, 10)));
        Ok(())
    }

    #[test]
    fn test_flex() {
        /*
        let text = "some text".to_string();
        let flex = Flex::row().add_child(&text);
        println!("{:?}", flex.debug(Size::new(5, 2)));
        */
    }
}
