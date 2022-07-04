use super::{AlongAxis, Axis, BoxConstraint, Layout, Tree, View};
use crate::{Error, Position, Size, SurfaceMut, TerminalSurface};
use std::cmp::{max, min};

#[derive(Debug)]
enum Child<'a> {
    Fixed { view: Box<dyn View + 'a> },
    Flex { view: Box<dyn View + 'a>, flex: f64 },
}

impl<'a> Child<'a> {
    fn view(&self) -> &dyn View {
        match self {
            Self::Fixed { view, .. } => view,
            Self::Flex { view, .. } => view,
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
        let mut surf = layout.apply_to(surf);
        for (index, child) in self.children.iter().enumerate() {
            let child_layout = layout.get(index).ok_or(Error::InvalidLayout)?;
            if child_layout.size.is_empty() {
                continue;
            }
            child.view().render(&mut surf.as_mut(), child_layout)?;
        }
        Ok(())
    }

    fn layout(&self, ct: BoxConstraint) -> Tree<Layout> {
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
                if let Child::Flex { view, flex } = child {
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
            }
        }

        // extra space to be filled
        let _extra = self.direction.major(ct.max()) - (major_non_flex + major_flex);

        // calculate offsets
        let mut offset = 0;
        for child_layout in children_layout.iter_mut() {
            *child_layout.pos.major_mut(self.direction) = offset;
            offset += child_layout.size.major(self.direction);
        }

        // create layout tree
        Tree::new(
            Layout {
                pos: Position::origin(),
                size: ct.clamp(Size::from_axes(self.direction, offset, minor)),
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
        let flex = Flex::row()
            .add_flex_child(2.0, Text::new(&text).with_face("fg=#ff0000".parse()?))
            .add_flex_child(1.0, "other text");

        let size = Size::new(5, 12);
        print!("{:?}", flex.debug(size));
        let reference = Tree::new(
            Layout {
                pos: Position::origin(),
                size: Size::new(3, 12),
            },
            vec![
                Tree::new(
                    Layout {
                        pos: Position::origin(),
                        size: Size::new(2, 8),
                    },
                    Vec::new(),
                ),
                Tree::new(
                    Layout {
                        pos: Position::new(0, 8),
                        size: Size::new(3, 4),
                    },
                    Vec::new(),
                ),
            ],
        );
        assert_eq!(reference, flex.layout(BoxConstraint::loose(size)));

        Ok(())
    }
}
