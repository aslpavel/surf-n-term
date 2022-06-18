use crate::{Error, Position, Size, TerminalSurface};
use std::{
    cmp::{max, min},
    fmt::Debug,
    ops::{Deref, DerefMut, Index, IndexMut},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BoxConstraint {
    min: Size,
    max: Size,
}

impl BoxConstraint {
    pub fn new(min: Size, max: Size) -> Self {
        Self { min, max }
    }
    pub fn min(&self) -> Size {
        self.min
    }

    pub fn max(&self) -> Size {
        self.max
    }

    pub fn loosen(&self) -> Self {
        Self {
            min: Size::empty(),
            max: self.max,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Layout {
    pub pos: Position,
    pub size: Size,
}

pub trait Widget: Debug {
    fn render(&self, surf: &mut TerminalSurface<'_>, layout: &Tree<Layout>) -> Result<(), Error>;

    /// Compute layout of the widget based on the constraints
    fn layout(&self, ct: BoxConstraint) -> Tree<Layout>;
}

impl<'a, T: Widget> Widget for &'a T {
    fn render(&self, surf: &mut TerminalSurface<'_>, layout: &Tree<Layout>) -> Result<(), Error> {
        (**self).render(surf, layout)
    }

    fn layout(&self, ct: BoxConstraint) -> Tree<Layout> {
        (**self).layout(ct)
    }
}

impl Widget for Box<dyn Widget> {
    fn render(&self, surf: &mut TerminalSurface<'_>, layout: &Tree<Layout>) -> Result<(), Error> {
        (**self).render(surf, layout)
    }

    fn layout(&self, ct: BoxConstraint) -> Tree<Layout> {
        (**self).layout(ct)
    }
}

#[derive(Default)]
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

/*
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
*/

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
            Self::Vertical => Self::Vertical,
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
enum Child {
    Fixed { widget: Box<dyn Widget> },
    Flex { widget: Box<dyn Widget>, flex: f64 },
}

#[derive(Debug)]
pub struct Flex {
    direction: Axis,
    children: Vec<Child>,
}

impl Flex {
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
    pub fn add_child(&mut self, child: impl Widget + 'static) -> &mut Self {
        self.children.push(Child::Fixed {
            widget: Box::new(child),
        });
        self
    }

    /// Add new flex size child
    pub fn add_flex_child(&mut self, flex: f64, child: impl Widget + 'static) -> &mut Self {
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

impl Widget for Flex {
    fn render(&self, _surf: &mut TerminalSurface<'_>, _layout: &Tree<Layout>) -> Result<(), Error> {
        todo!()
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
        let _extra = self.direction.major(ct.min()) - (major_non_flex + major_flex);

        // calculate offsets
        let mut offset = 0;
        for index in 0..self.children.len() {
            let child_layout = &mut children_layout[index];
            *child_layout.pos.major_mut(self.direction) = offset;
            offset += child_layout.size.major(self.direction);
        }

        todo!()
    }
}
