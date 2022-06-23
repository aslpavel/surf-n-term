//! [Container] view can specify the size and alignment for its child view
use super::{Align, BoxConstraint, Layout, Tree, View};
use crate::{Error, Face, FaceAttrs, Position, Size, TerminalSurface, TerminalSurfaceExt, RGBA};

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

    /// set horizontal alignment
    pub fn horizontal(self, align: Align) -> Self {
        Self {
            align_horizontal: align,
            ..self
        }
    }

    /// set vertical alignment
    pub fn vertical(self, align: Align) -> Self {
        Self {
            align_vertical: align,
            ..self
        }
    }

    pub fn color(self, color: RGBA) -> Self {
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
                size_max.height
            } else {
                0
            },
        };
        let view_layout = self.view.layout(BoxConstraint::new(size_min, size_max));
        // resulting view size, equal to child view size if not specified
        let size = Size {
            width: if self.size.width == 0 {
                view_layout.size.width
            } else {
                self.size.width
            },
            height: if self.size.height == 0 {
                view_layout.size.height
            } else {
                self.size.height
            },
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
