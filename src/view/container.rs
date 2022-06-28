//! [Container] view can specify the size and alignment for its child view
use super::{BoxConstraint, Layout, Tree, View};
use crate::{Error, Face, FaceAttrs, Position, Size, TerminalSurface, TerminalSurfaceExt, RGBA};
use std::cmp::max;

/// Alignment of a child view
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
    pub fn new(view: V) -> Self {
        Self {
            size: Size::empty(),
            color: None,
            align_vertical: Align::default(),
            align_horizontal: Align::default(),
            view,
        }
    }

    /// set size for the container
    pub fn width(self, width: usize) -> Self {
        Self {
            size: Size { width, ..self.size },
            ..self
        }
    }

    /// set hight for the container
    pub fn height(self, height: usize) -> Self {
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
        surf: &'a mut TerminalSurface<'a>,
        layout: &Tree<Layout>,
    ) -> Result<(), Error> {
        let surf = &mut layout.view(surf);
        if self.color.is_some() {
            surf.erase(Face::new(None, self.color, FaceAttrs::EMPTY));
        }
        let view_layout = layout.get(0).ok_or(Error::InvalidLayout)?;
        self.view.render(&mut view_layout.view(surf), view_layout)?;
        Ok(())
    }

    fn layout(&self, ct: BoxConstraint) -> Tree<Layout> {
        // constraint max size if it is specified
        let size_max = Size {
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
        let view_layout = self.view.layout(BoxConstraint::new(size_min, size_max));
        let size = Size {
            width: view_layout
                .size
                .width
                .clamp(max(self.size.width, ct.min().width), size_max.width),
            height: view_layout
                .size
                .height
                .clamp(max(self.size.height, ct.min().height), size_max.height),
        };
        let pos = Position {
            row: self
                .align_vertical
                .align(size.height.abs_diff(view_layout.size.height)),
            col: self
                .align_horizontal
                .align(size.width.abs_diff(view_layout.size.width)),
        };
        Tree {
            value: Layout { size, pos },
            children: vec![view_layout],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{view::Fixed, RGBA};

    #[test]
    fn test_container() -> Result<(), Error> {
        let view = Fixed::new(Size::new(1, 4), "#ff0000".parse::<RGBA>()?);

        let size = Size::new(5, 10);
        let cont = Container::new(&view)
            .with_size(size)
            .with_horizontal(Align::End);

        println!("{:?}", cont.debug(size));

        Ok(())
    }
}
