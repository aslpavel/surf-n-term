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
    Start,
    Center,
    End,
    Fill,
    Offset(i32),
}

impl Align {
    pub fn align(&self, small: usize, large: usize) -> usize {
        let small = small.clamp(0, large);
        match self {
            Self::Start | Self::Fill => 0,
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
        Self::Center
    }
}

#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub struct Margins {
    pub left: usize,
    pub right: usize,
    pub top: usize,
    pub bottom: usize,
}

#[derive(Debug)]
pub struct Container<V> {
    view: V,
    color: Option<RGBA>,
    align_vertical: Align,
    align_horizontal: Align,
    margins: Margins,
    size: Size,
}

impl<V: View> Container<V> {
    /// create new container view
    pub fn new(child: impl IntoView<View = V>) -> Self {
        Self {
            size: Size::empty(),
            color: None,
            align_vertical: Align::default(),
            align_horizontal: Align::default(),
            margins: Margins::default(),
            view: child.into_view(),
        }
    }

    /// set size for the container
    pub fn with_width(self, width: usize) -> Self {
        Self {
            size: Size { width, ..self.size },
            ..self
        }
    }

    /// set hight for the container
    pub fn with_height(self, height: usize) -> Self {
        Self {
            size: Size {
                height,
                ..self.size
            },
            ..self
        }
    }

    pub fn with_size(self, size: Size) -> Self {
        Self { size, ..self }
    }

    /// set horizontal alignment
    pub fn with_horizontal(self, align: Align) -> Self {
        Self {
            align_horizontal: align,
            ..self
        }
    }

    /// set vertical alignment
    pub fn with_vertical(self, align: Align) -> Self {
        Self {
            align_vertical: align,
            ..self
        }
    }

    pub fn with_color(self, color: RGBA) -> Self {
        Self {
            color: Some(color),
            ..self
        }
    }

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
        if self.color.is_some() {
            surf.view_mut(
                self.margins.top..surf.height().saturating_sub(self.margins.bottom),
                self.margins.left..surf.width().saturating_sub(self.margins.right),
            )
            .erase(Face::new(None, self.color, FaceAttrs::EMPTY));
        }
        self.view.render(
            ctx,
            &mut surf.as_mut(),
            layout.get(0).ok_or(Error::InvalidLayout)?,
        )?;
        Ok(())
    }

    fn layout(&self, ctx: &ViewContext, ct: BoxConstraint) -> Tree<Layout> {
        // calculate the size taken by the whole container, it will span
        // all available space if size is not set.
        let size = Size {
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
        let view_size_max = Size {
            height: size
                .height
                .saturating_sub(self.margins.top)
                .saturating_sub(self.margins.bottom),
            width: size
                .width
                .saturating_sub(self.margins.left)
                .saturating_sub(self.margins.right),
        };
        let view_size_min = Size {
            height: if self.align_vertical == Align::Fill {
                view_size_max.height
            } else {
                0
            },
            width: if self.align_horizontal == Align::Fill {
                view_size_max.width
            } else {
                0
            },
        };
        // calculate view layout
        let mut view_layout = self
            .view
            .layout(ctx, BoxConstraint::new(view_size_min, view_size_max));
        view_layout.pos = Position {
            row: self
                .align_vertical
                .align(view_layout.size.height, view_size_max.height)
                .add(self.margins.top),
            col: self
                .align_horizontal
                .align(view_layout.size.width, view_size_max.width)
                .add(self.margins.left),
        };
        // layout tree
        Tree::new(Layout::new().with_size(size), vec![view_layout])
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

        let cont = cont.with_vertical(Align::Fill);
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

        Ok(())
    }
}
