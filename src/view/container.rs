//! [Container] view can specify the size and alignment for its child view
use super::{BoxConstraint, IntoView, Layout, Tree, View, ViewContext};
use crate::{Error, Face, FaceAttrs, Position, Size, TerminalSurface, TerminalSurfaceExt, RGBA};

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

    pub fn reserve(&self) -> usize {
        match self {
            Self::Offset(offset) => offset.unsigned_abs() as usize,
            _ => 0,
        }
    }
}

impl Default for Align {
    fn default() -> Self {
        Self::Center
    }
}

#[derive(Debug)]
pub struct Container<V> {
    view: V,
    color: Option<RGBA>,
    align_vertical: Align,
    align_horizontal: Align,
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
}

impl<V: View> View for Container<V> {
    fn render<'a>(
        &self,
        ctx: &ViewContext,
        surf: &'a mut TerminalSurface<'a>,
        layout: &Tree<Layout>,
    ) -> Result<(), Error> {
        let surf = &mut layout.apply_to(surf);
        if self.color.is_some() {
            surf.erase(Face::new(None, self.color, FaceAttrs::EMPTY));
        }
        self.view
            .render(ctx, surf, layout.get(0).ok_or(Error::InvalidLayout)?)?;
        Ok(())
    }

    fn layout(&self, ctx: &ViewContext, ct: BoxConstraint) -> Tree<Layout> {
        // constraint max size if it is specified
        let size_max = Size {
            height: {
                let height = if self.size.height == 0 {
                    ct.max().height
                } else {
                    self.size.height.clamp(ct.min().height, ct.max().height)
                };
                height.saturating_sub(self.align_vertical.reserve())
            },
            width: {
                let width = if self.size.width == 0 {
                    ct.max().width
                } else {
                    self.size.width.clamp(ct.min().width, ct.max().width)
                };
                width.saturating_sub(self.align_horizontal.reserve())
            },
        };
        // make it tight along `Align::Fill` axis
        let size_min = Size {
            height: if self.align_vertical == Align::Fill {
                size_max.height
            } else {
                0
            },
            width: if self.align_horizontal == Align::Fill {
                size_max.width
            } else {
                0
            },
        };
        let mut view_layout = self
            .view
            .layout(ctx, BoxConstraint::new(size_min, size_max));
        let size = Size {
            width: {
                let width = if self.size.width == 0 {
                    view_layout.size.width
                } else {
                    self.size.width
                };
                width + self.align_horizontal.reserve()
            },
            height: {
                let height = if self.size.height == 0 {
                    view_layout.size.height
                } else {
                    self.size.height
                };
                height + self.align_vertical.reserve()
            },
        };
        let size = ct.clamp(size);
        view_layout.pos = Position {
            row: self
                .align_vertical
                .align(view_layout.size.height, size.height),
            col: self
                .align_horizontal
                .align(view_layout.size.width, size.width),
        };
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
            .with_size(size)
            .with_color("#98971a".parse()?)
            .with_horizontal(Align::End);

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

        Ok(())
    }
}
