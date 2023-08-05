//! [Container] view can specify the size and alignment for its child view
use std::ops::Add;

use super::{BoxConstraint, IntoView, Layout, Tree, View, ViewContext};
use crate::{
    Error, Face, FaceAttrs, Position, Size, Surface, SurfaceMut, TerminalSurface,
    TerminalSurfaceExt, RGBA,
};

/// Alignment of a child view
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Align {
    /// Align to the smallest value along the axis
    Start,
    /// Align to the center along the axis
    Center,
    /// Align to the end along the axis
    End,
    /// Take all available space along the axis
    Expand,
    /// Try to shrink container to match the size of the child
    Shrink,
    /// Place  the view at the specified offset, negative means offset is from
    /// the maximum value
    Offset(i32),
}

impl Align {
    pub fn align(&self, small: usize, large: usize) -> usize {
        let small = small.clamp(0, large);
        match self {
            Self::Start | Self::Expand | Self::Shrink => 0,
            Self::Center => (large - small) / 2,
            Self::End => large - small,
            Self::Offset(offset) => {
                if *offset >= 0 {
                    *offset as usize
                } else {
                    (large - small).saturating_sub(offset.unsigned_abs() as usize)
                }
            }
        }
    }
}

impl Default for Align {
    fn default() -> Self {
        Self::Shrink
    }
}

#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub struct Margins {
    pub left: usize,
    pub right: usize,
    pub top: usize,
    pub bottom: usize,
}

/// View that can constrain and align its child view
#[derive(Debug)]
pub struct Container<V> {
    child: V,
    face: Face,
    align_vertical: Align,
    align_horizontal: Align,
    margins: Margins,
    size: Size,
}

impl<V: View> Container<V> {
    /// Create new container view
    pub fn new(child: impl IntoView<View = V>) -> Self {
        Self {
            size: Size::empty(),
            face: Face::default(),
            align_vertical: Align::default(),
            align_horizontal: Align::default(),
            margins: Margins::default(),
            child: child.into_view(),
        }
    }

    /// Set size for the container
    pub fn with_width(self, width: usize) -> Self {
        Self {
            size: Size { width, ..self.size },
            ..self
        }
    }

    /// Set hight for the container
    pub fn with_height(self, height: usize) -> Self {
        Self {
            size: Size {
                height,
                ..self.size
            },
            ..self
        }
    }

    /// Set size for the container
    pub fn with_size(self, size: Size) -> Self {
        Self { size, ..self }
    }

    /// Set horizontal alignment
    pub fn with_horizontal(self, align: Align) -> Self {
        Self {
            align_horizontal: align,
            ..self
        }
    }

    /// Set vertical alignment
    pub fn with_vertical(self, align: Align) -> Self {
        Self {
            align_vertical: align,
            ..self
        }
    }

    /// Fill container with color
    pub fn with_color(self, color: RGBA) -> Self {
        Self {
            face: Face::new(None, Some(color), FaceAttrs::EMPTY),
            ..self
        }
    }

    /// Fill container with face
    pub fn with_face(self, face: Face) -> Self {
        Self { face, ..self }
    }

    /// Set container margins
    pub fn with_margins(self, margins: Margins) -> Self {
        Self { margins, ..self }
    }
}

impl<V: View> View for Container<V> {
    fn render<'a>(
        &self,
        ctx: &ViewContext,
        surf: &'a mut TerminalSurface<'a>,
        layout: &Tree<Layout>,
    ) -> Result<(), Error> {
        let mut surf = layout.apply_to(surf);
        if self.face != Face::default() {
            surf.view_mut(
                self.margins.top..surf.height().saturating_sub(self.margins.bottom),
                self.margins.left..surf.width().saturating_sub(self.margins.right),
            )
            .erase(self.face);
        }
        self.child.render(
            ctx,
            &mut surf.as_mut(),
            layout.get(0).ok_or(Error::InvalidLayout)?,
        )?;
        Ok(())
    }

    fn layout(&self, ctx: &ViewContext, ct: BoxConstraint) -> Tree<Layout> {
        // calculate the size taken by the whole container, it will span
        // all available space if size is not set.
        let mut container_size = Size {
            height: if self.size.height == 0 {
                ct.max().height
            } else {
                self.size.height.clamp(ct.min().height, ct.max().height)
            },
            width: if self.size.width == 0 {
                ct.max().width
            } else {
                self.size.width.clamp(ct.min().width, ct.max().width)
            },
        };

        // calculate view constraints, reserving space for margin
        let child_size_max = Size {
            height: container_size
                .height
                .saturating_sub(self.margins.top)
                .saturating_sub(self.margins.bottom),
            width: container_size
                .width
                .saturating_sub(self.margins.left)
                .saturating_sub(self.margins.right),
        };
        let child_size_min = Size {
            height: if self.align_vertical == Align::Expand {
                child_size_max.height
            } else {
                0
            },
            width: if self.align_horizontal == Align::Expand {
                child_size_max.width
            } else {
                0
            },
        };
        let child_constraint = BoxConstraint::new(child_size_min, child_size_max);

        // calculate child layout
        let mut child_layout = self.child.layout(ctx, child_constraint);
        child_layout.pos = Position {
            row: self
                .align_vertical
                .align(child_layout.size.height, child_size_max.height)
                .add(self.margins.top),
            col: self
                .align_horizontal
                .align(child_layout.size.width, child_size_max.width)
                .add(self.margins.left),
        };

        // try to shrink container if necessary
        if self.align_vertical == Align::Shrink {
            container_size.height = child_layout
                .size
                .height
                .add(self.margins.top)
                .add(self.margins.bottom)
                .clamp(ct.min.height, ct.max.height)
        }
        if self.align_horizontal == Align::Shrink {
            container_size.width = child_layout
                .size
                .width
                .add(self.margins.left)
                .add(self.margins.right)
                .clamp(ct.min.width, ct.max.width)
        }

        // layout tree
        Tree::new(Layout::new().with_size(container_size), vec![child_layout])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RGBA;

    #[derive(Debug)]
    struct Fixed<V> {
        view: V,
        size: Size,
    }

    impl<V> Fixed<V> {
        fn new(size: Size, view: V) -> Self {
            Self { view, size }
        }
    }

    impl<V: View> View for Fixed<V> {
        fn render<'a>(
            &self,
            ctx: &ViewContext,
            surf: &'a mut TerminalSurface<'a>,
            layout: &Tree<Layout>,
        ) -> Result<(), Error> {
            self.view.render(ctx, surf, layout)
        }

        fn layout(&self, ctx: &ViewContext, ct: BoxConstraint) -> Tree<Layout> {
            let size = ct.clamp(self.size);
            self.view.layout(ctx, BoxConstraint::tight(size))
        }
    }

    #[test]
    fn test_container() -> Result<(), Error> {
        let view = Fixed::new(Size::new(1, 4), "#cc241d".parse::<RGBA>()?);
        let ctx = ViewContext::dummy();

        let size = Size::new(5, 10);
        let cont = Container::new(&view)
            // .with_size(size)
            .with_color("#98971a".parse()?)
            .with_vertical(Align::Center)
            .with_horizontal(Align::End);

        println!("{:?}", cont);
        println!("{:?}", cont.debug(size));
        assert_eq!(
            Tree::new(
                Layout::new().with_size(size),
                vec![Tree::leaf(
                    Layout::new()
                        .with_position(Position::new(2, 6))
                        .with_size(Size::new(1, 4))
                )],
            ),
            cont.layout(&ctx, BoxConstraint::loose(size))
        );

        let cont = cont.with_horizontal(Align::Start);
        println!("{:?}", cont);
        println!("{:?}", cont.debug(size));
        assert_eq!(
            Tree::new(
                Layout::new().with_size(size),
                vec![Tree::leaf(
                    Layout::new()
                        .with_position(Position::new(2, 0))
                        .with_size(Size::new(1, 4))
                )]
            ),
            cont.layout(&ctx, BoxConstraint::loose(size))
        );

        let cont = cont.with_horizontal(Align::Center);
        println!("{:?}", cont);
        println!("{:?}", cont.debug(size));
        assert_eq!(
            Tree::new(
                Layout::new().with_size(size),
                vec![Tree::leaf(
                    Layout::new()
                        .with_position(Position::new(2, 3))
                        .with_size(Size::new(1, 4))
                )]
            ),
            cont.layout(&ctx, BoxConstraint::loose(size))
        );

        let cont = cont
            .with_horizontal(Align::Offset(-2))
            .with_vertical(Align::Offset(1));
        println!("{:?}", cont);
        println!("{:?}", cont.debug(size));
        assert_eq!(
            Tree::new(
                Layout::new().with_size(size),
                vec![Tree::leaf(
                    Layout::new()
                        .with_position(Position::new(1, 4))
                        .with_size(Size::new(1, 4))
                )]
            ),
            cont.layout(&ctx, BoxConstraint::loose(size))
        );

        let cont = cont.with_vertical(Align::Expand);
        println!("{:?}", cont);
        println!("{:?}", cont.debug(size));
        assert_eq!(
            Tree::new(
                Layout::new().with_size(size),
                vec![Tree::leaf(
                    Layout::new()
                        .with_position(Position::new(0, 4))
                        .with_size(Size::new(5, 4))
                )]
            ),
            cont.layout(&ctx, BoxConstraint::loose(size))
        );

        let cont = cont.with_margins(Margins {
            top: 1,
            bottom: 0,
            left: 2,
            right: 1,
        });
        println!("{:?}", cont);
        println!("{:?}", cont.debug(size));
        assert_eq!(
            Tree::new(
                Layout::new().with_size(size),
                vec![Tree::leaf(
                    Layout::new()
                        .with_position(Position::new(1, 3))
                        .with_size(Size::new(4, 4))
                )]
            ),
            cont.layout(&ctx, BoxConstraint::loose(size))
        );

        let cont = cont.with_vertical(Align::Shrink);
        println!("{:?}", cont);
        println!("{:?}", cont.debug(size));
        assert_eq!(
            Tree::new(
                Layout::new().with_size(Size::new(2, 10)),
                vec![Tree::leaf(
                    Layout::new()
                        .with_position(Position::new(1, 3))
                        .with_size(Size::new(1, 4))
                )]
            ),
            cont.layout(&ctx, BoxConstraint::loose(size))
        );

        Ok(())
    }
}
