use super::{BoxConstraint, Layout, Tree, View};
use crate::{Error, Position, Size, TerminalSurface};
use std::cmp::{max, min};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Align {
    Start,
    Center,
    End,
    Fill,
}

impl Align {
    pub fn align(&self, leftover: usize) -> usize {
        match self {
            Self::Start | Self::Fill => 0,
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
    Fixed { view: Box<dyn View + 'a> },
    Flex { view: Box<dyn View + 'a>, flex: f64 },
}

impl<'a> Child<'a> {
    fn view(&self) -> &dyn View {
        match self {
            Self::Fixed { view, .. } => &*view,
            Self::Flex { view, .. } => &*view,
        }
    }
}

#[derive(Debug)]
pub struct Flex<'a> {
    direction: Axis,
    children: Vec<Child<'a>>,
}

impl<'a> Flex<'a> {
    /// Create new flex view aligned along direction [Axis]
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
    pub fn add_child(mut self, child: impl View + 'a) -> Self {
        self.children.push(Child::Fixed {
            view: Box::new(child),
        });
        self
    }

    /// Add new flex size child
    pub fn add_flex_child(mut self, flex: f64, child: impl View + 'a) -> Self {
        if flex > 0.0 {
            self.children.push(Child::Flex {
                view: Box::new(child),
                flex,
            });
            self
        } else {
            self.add_child(child)
        }
    }
}

impl<'a> View for Flex<'a>
where
    Self: 'a,
{
    fn render<'b>(
        &self,
        surf: &'b mut TerminalSurface<'b>,
        layout: &Tree<Layout>,
    ) -> Result<(), Error> {
        let surf = &mut layout.view(surf);
        for (index, child) in self.children.iter().enumerate() {
            let child_layout = layout.get(index).ok_or(Error::InvalidLayout)?;
            if child_layout.size.is_empty() {
                continue;
            }
            child.view().render(&mut child_layout.view(surf), layout)?;
        }
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
                Child::Fixed { view } => {
                    let child_layout = view.layout(ct_loosen);
                    major_non_flex += self.direction.major(child_layout.size);
                    minor = max(minor, self.direction.minor(child_layout.size));
                    children_layout[index] = child_layout;
                }
                Child::Flex { flex, .. } => {
                    flex_total += flex;
                }
            }
        }

        // calculate available space for flex views
        let major_total = self.direction.major(ct.max());
        major_non_flex = min(major_total, major_non_flex);
        let major_remain = major_total - major_non_flex;

        // layout flex
        let mut major_flex = 0;
        if major_remain > 0 {
            let per_flex = (major_remain as f64) / flex_total;
            for (index, child) in self.children.iter().enumerate() {
                match child {
                    Child::Flex { view, flex } => {
                        let child_major = (flex * per_flex).round() as usize;
                        if child_major == 0 {
                            continue;
                        }
                        let child_ct = self.direction.constraint(ct_loosen, 0, child_major);
                        let child_layout = view.layout(child_ct);

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
                pos: Position::origin(),
                size: ct.clamp(self.direction.size(offset, minor)),
            },
            children_layout,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::Text;

    #[test]
    fn test_flex() -> Result<(), Error> {
        let text = "some text".to_string();
        let flex = Flex::column()
            .add_flex_child(2.0, Text::new(&text).with_face("fg=#ff0000".parse()?))
            .add_flex_child(1.0, "other text");

        let size = Size::new(5, 12);
        println!("{:#?}", flex.layout(BoxConstraint::loose(size)));
        println!("{:?}", flex.debug(size));
        Ok(())
    }
}
