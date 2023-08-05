use super::{AlongAxis, Axis, BoxConstraint, IntoView, Layout, Tree, View, ViewContext};
use crate::{Error, Size, SurfaceMut, TerminalSurface};
use std::{cmp::max, fmt};

enum Child<'a> {
    Fixed { view: Box<dyn View + 'a> },
    Flex { view: Box<dyn View + 'a>, flex: f64 },
}

impl<'a> fmt::Debug for Child<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fixed { .. } => write!(f, "Fixed"),
            Self::Flex { flex, .. } => write!(f, "Flex({flex})"),
        }
    }
}

impl<'a> Child<'a> {
    fn view(&self) -> &dyn View {
        match self {
            Self::Fixed { view, .. } => view,
            Self::Flex { view, .. } => view,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Justify {
    Start,
    Center,
    End,
    SpaceBetween,
    SpaceAround,
    SpaceEvenly,
}

#[derive(Debug)]
pub struct Flex<'a> {
    direction: Axis,
    justify: Justify,
    children: Vec<Child<'a>>,
}

impl<'a> Flex<'a> {
    /// Create new flex view aligned along direction [Axis]
    pub fn new(direction: Axis) -> Self {
        Self {
            direction,
            justify: Justify::Start,
            children: Default::default(),
        }
    }

    pub fn justify(self, justify: Justify) -> Self {
        Self { justify, ..self }
    }

    pub fn row() -> Self {
        Self::new(Axis::Horizontal)
    }

    pub fn column() -> Self {
        Self::new(Axis::Vertical)
    }

    pub fn push_child(&mut self, child: impl IntoView + 'a) {
        self.children.push(Child::Fixed {
            view: child.into_view().boxed(),
        });
    }

    pub fn push_flex_child(&mut self, flex: f64, child: impl IntoView + 'a) {
        if flex > 0.0 {
            self.children.push(Child::Flex {
                view: child.into_view().boxed(),
                flex,
            });
        } else {
            self.push_child(child);
        }
    }

    /// Add new fixed size child
    pub fn add_child(mut self, child: impl IntoView + 'a) -> Self {
        self.push_child(child);
        self
    }

    /// Add new flex size child
    pub fn add_flex_child(mut self, flex: f64, child: impl IntoView + 'a) -> Self {
        self.push_flex_child(flex, child);
        self
    }
}

impl<'a> View for Flex<'a>
where
    Self: 'a,
{
    fn render<'b>(
        &self,
        ctx: &ViewContext,
        surf: &'b mut TerminalSurface<'b>,
        layout: &Tree<Layout>,
    ) -> Result<(), Error> {
        let mut surf = layout.apply_to(surf);
        for (index, child) in self.children.iter().enumerate() {
            let child_layout = layout.get(index).ok_or(Error::InvalidLayout)?;
            if child_layout.size.is_empty() {
                continue;
            }
            child.view().render(ctx, &mut surf.as_mut(), child_layout)?;
        }
        Ok(())
    }

    fn layout(&self, ctx: &ViewContext, ct: BoxConstraint) -> Tree<Layout> {
        // allocate children layout
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
                    let child_layout = view.layout(ctx, ct_loosen);
                    major_non_flex += self.direction.major(child_layout.size);
                    minor = max(minor, self.direction.minor(child_layout.size));
                    children_layout[index] = child_layout;
                }
                Child::Flex { flex, .. } => {
                    flex_total += flex;
                }
            }
        }

        // layout flex
        let major_remain = self
            .direction
            .major(ct.max())
            .saturating_sub(major_non_flex);
        let mut major_flex = 0;
        if major_remain > 0 && flex_total > 0.0 {
            let per_flex = (major_remain as f64) / flex_total;
            for (index, child) in self.children.iter().enumerate() {
                if let Child::Flex { view, flex } = child {
                    let child_major = (flex * per_flex).round() as usize;
                    if child_major == 0 {
                        continue;
                    }
                    let child_ct = self.direction.constraint(ct_loosen, 0, child_major);
                    let child_layout = view.layout(ctx, child_ct);

                    major_flex += self.direction.major(child_layout.size);
                    minor = max(minor, self.direction.minor(child_layout.size));
                    children_layout[index] = child_layout;
                }
            }
        }

        // unused space to be filled
        let unused = self
            .direction
            .major(ct.max())
            .saturating_sub(major_non_flex + major_flex);
        let (space_side, space_between) = if unused > 0 {
            match self.justify {
                Justify::Start => (0, 0),
                Justify::Center => (unused / 2, 0),
                Justify::End => (unused, 0),
                Justify::SpaceBetween => {
                    let space_between = if self.children.len() <= 1 {
                        unused
                    } else {
                        unused / (self.children.len() - 1)
                    };
                    (0, space_between)
                }
                Justify::SpaceEvenly => {
                    let space = unused / (self.children.len() + 1);
                    (space, space)
                }
                Justify::SpaceAround => {
                    let space = unused / self.children.len();
                    (space / 2, space)
                }
            }
        } else {
            (0, 0)
        };

        // calculate offsets
        let mut offset = space_side;
        for child_layout in children_layout.iter_mut() {
            *child_layout.pos.major_mut(self.direction) = offset;
            offset += child_layout.size.major(self.direction);
            offset += space_between;
        }

        // create layout tree
        Tree::new(
            Layout::new().with_size(ct.clamp(Size::from_axes(self.direction, offset, minor))),
            children_layout,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{view::Text, Position};

    #[test]
    fn test_flex() -> Result<(), Error> {
        let ctx = ViewContext::dummy();
        let flex = Flex::row()
            .add_flex_child(
                2.0,
                Text::from("some text")
                    .mark("fg=#ff0000".parse()?, ..)
                    .take(),
            )
            .add_flex_child(1.0, "other text");

        let size = Size::new(5, 12);
        print!("[flex] {:?}", flex.debug(size));
        let reference = Tree::new(
            Layout::new().with_size(Size::new(3, 12)),
            vec![
                Tree::leaf(Layout::new().with_size(Size::new(2, 8))),
                Tree::leaf(
                    Layout::new()
                        .with_position(Position::new(0, 8))
                        .with_size(Size::new(3, 4)),
                ),
            ],
        );
        assert_eq!(reference, flex.layout(&ctx, BoxConstraint::loose(size)));

        Ok(())
    }
}
