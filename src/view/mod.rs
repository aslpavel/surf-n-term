//! Defines [View] that represents anything that can be rendered into a terminal.
//! As well as some useful implementations such as [Text], [Flex], [Container], ...
mod container;
mod flex;
mod scrollbar;
mod text;
pub use container::{Align, Container};
pub use flex::Flex;
pub use scrollbar::ScrollBar;
pub use text::Text;

use crate::{
    Cell, Error, Face, FaceAttrs, Position, Size, SurfaceMut, SurfaceOwned, TerminalSurface,
    TerminalSurfaceExt, RGBA,
};
use std::{
    fmt::Debug,
    ops::{Deref, DerefMut, Index, IndexMut},
};

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
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct Layout {
    pub pos: Position,
    pub size: Size,
}

impl Debug for Layout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Layout")
            .field("row", &self.pos.row)
            .field("col", &self.pos.col)
            .field("height", &self.size.height)
            .field("row", &self.size.width)
            .finish()
    }
}

impl Layout {
    /// Constrain surface by the layout, that is create sub-subsurface view
    /// with offset `pos` and size of `size`.
    pub fn view<'a, S>(&self, surf: &'a mut S) -> TerminalSurface<'a>
    where
        S: SurfaceMut<Item = Cell>,
    {
        surf.view_mut(
            self.pos.row..self.pos.row + self.size.height,
            self.pos.col..self.pos.col + self.size.width,
        )
    }
}

/// View is anything that can be layed out and rendered to the terminal
pub trait View: Debug {
    /// Render view into a given surface with the provided layout
    fn render<'a>(
        &self,
        surf: &'a mut TerminalSurface<'a>,
        layout: &Tree<Layout>,
    ) -> Result<(), Error>;

    /// Compute layout of the view based on the constraints
    fn layout(&self, ct: BoxConstraint) -> Tree<Layout>;

    /// Wrapper around view that implements [std::fmt::Debug] which renders
    /// view. Only supposed to be used for debugging.
    fn debug(&self, size: Size) -> Preview<&'_ Self>
    where
        Self: Sized,
    {
        Preview { view: self, size }
    }
}

pub struct Preview<V> {
    view: V,
    size: Size,
}

impl<V: View> std::fmt::Debug for Preview<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut surf = SurfaceOwned::new(self.size.height, self.size.width);
        surf.draw_view(&self.view).map_err(|_| std::fmt::Error)?;
        surf.debug().fmt(f)
    }
}

impl<'a, V: View + ?Sized> View for &'a V {
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

impl<'a> View for Box<dyn View + 'a> {
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

#[derive(Debug, PartialEq, Eq, Hash, Default)]
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

/// Major axis of the [Flex] view
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Axis {
    Horizontal,
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

#[derive(Debug)]
pub struct Fixed<V> {
    view: V,
    size: Size,
}

impl<V> Fixed<V> {
    pub fn new(size: Size, view: V) -> Self {
        Self { view, size }
    }
}

impl<V: View> View for Fixed<V> {
    fn render<'a>(
        &self,
        surf: &'a mut TerminalSurface<'a>,
        layout: &Tree<Layout>,
    ) -> Result<(), Error> {
        self.view.render(surf, layout)
    }

    fn layout(&self, ct: BoxConstraint) -> Tree<Layout> {
        let size = ct.clamp(self.size);
        self.view.layout(BoxConstraint::tight(size))
    }
}

impl View for RGBA {
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

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn render(view: &dyn View, size: Size) -> Result<SurfaceOwned<Cell>, Error> {
        let layout = view.layout(BoxConstraint::loose(size));
        let mut surf = SurfaceOwned::new(size.height, size.width);
        view.render(&mut surf.view_mut(.., ..), &layout)?;
        Ok(surf)
    }

    #[test]
    fn test_references() -> Result<(), Error> {
        fn witness(_: impl View) {}
        let color = "#ff0000".parse::<RGBA>()?;

        witness(color);
        witness(&color);
        witness(&color as &dyn View);
        witness(Box::new(color) as Box<dyn View>);

        Ok(())
    }

    #[test]
    fn test_fixed() -> Result<(), Error> {
        let size = Size::new(3, 5);
        let fixed = Fixed::new(size, "#ff0000".parse::<RGBA>()?);
        print!("{:?}", fixed.debug(size));
        Ok(())
    }
}
