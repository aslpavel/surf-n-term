//! [Container] view can specify the size and alignment for its child view
use std::ops::Add;

use serde::{de::DeserializeSeed, Deserialize, Serialize};

use super::{
    BoxConstraint, IntoView, Layout, Tree, TreeMut, View, ViewContext, ViewDeserializer,
    ViewLayout, ViewMutLayout,
};
use crate::{Error, Face, FaceAttrs, Position, Size, TerminalSurface, TerminalSurfaceExt, RGBA};

/// Alignment of a child view
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Align {
    /// Align to the smallest value along the axis
    #[serde(rename = "start")]
    Start,
    /// Align to the center along the axis
    #[serde(rename = "center")]
    Center,
    /// Align to the end along the axis
    #[serde(rename = "end")]
    End,
    /// Take all available space along the axis
    #[serde(rename = "expand")]
    Expand,
    /// Try to shrink container to match the size of the child
    #[serde(rename = "shrink")]
    Shrink,
    /// Place  the view at the specified offset, negative means offset is from
    /// the maximum value
    #[serde(rename = "offset")]
    Offset(i32),
}

impl Align {
    pub fn align(&self, size: usize, space: usize) -> usize {
        let size = size.clamp(0, space);
        match self {
            Self::Start | Self::Expand | Self::Shrink => 0,
            Self::Center => (space - size) / 2,
            Self::End => space - size,
            Self::Offset(offset) => {
                if *offset >= 0 {
                    *offset as usize
                } else {
                    (space - size).saturating_sub(offset.unsigned_abs() as usize)
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

#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct Margins {
    #[serde(default)]
    pub left: usize,
    #[serde(default)]
    pub right: usize,
    #[serde(default)]
    pub top: usize,
    #[serde(default)]
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
    fn render(
        &self,
        ctx: &ViewContext,
        surf: TerminalSurface<'_>,
        layout: ViewLayout<'_>,
    ) -> Result<(), Error> {
        let mut surf = layout.apply_to(surf);
        if self.face != Face::default() {
            surf.erase(self.face);
        }
        let child_layout = layout.children().next().ok_or(Error::InvalidLayout)?;
        self.child.render(ctx, surf, child_layout)?;
        Ok(())
    }

    fn layout(
        &self,
        ctx: &ViewContext,
        ct: BoxConstraint,
        mut layout: ViewMutLayout<'_>,
    ) -> Result<(), Error> {
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
        let mut child_layout = layout.push_default();
        self.child
            .layout(ctx, child_constraint, child_layout.view_mut())?;
        let child_size = child_layout.size();
        child_layout.set_pos(Position {
            row: self
                .align_vertical
                .align(child_size.height, child_size_max.height)
                .add(self.margins.top),
            col: self
                .align_horizontal
                .align(child_size.width, child_size_max.width)
                .add(self.margins.left),
        });

        // try to shrink container if necessary
        if self.align_vertical == Align::Shrink {
            container_size.height = child_layout
                .size()
                .height
                .add(self.margins.top)
                .add(self.margins.bottom)
                .clamp(ct.min.height, ct.max.height)
        }
        if self.align_horizontal == Align::Shrink {
            container_size.width = child_layout
                .size()
                .width
                .add(self.margins.left)
                .add(self.margins.right)
                .clamp(ct.min.width, ct.max.width)
        }

        // layout tree
        *layout = Layout::new().with_size(container_size);
        Ok(())
    }
}

/// Construct [Container] object from JSON value
pub(super) fn from_json_value(
    seed: &ViewDeserializer<'_>,
    value: &serde_json::Value,
) -> Result<Container<Box<dyn View>>, Error> {
    let face = value
        .get("face")
        .map(|value| seed.face(value))
        .transpose()?
        .unwrap_or_default();
    let align_vertical = value
        .get("vertical")
        .map(Align::deserialize)
        .transpose()?
        .unwrap_or_default();
    let align_horizontal = value
        .get("horizontal")
        .map(Align::deserialize)
        .transpose()?
        .unwrap_or_default();
    let margins = value
        .get("margins")
        .map(Margins::deserialize)
        .transpose()?
        .unwrap_or_default();
    let size = value
        .get("size")
        .map(Size::deserialize)
        .transpose()?
        .unwrap_or_default();
    let view = value
        .get("child")
        .ok_or_else(|| Error::ParseError("Container", "must include child attribute".to_owned()))?;
    Ok(Container {
        child: seed.deserialize(view)?.boxed(),
        face,
        align_vertical,
        align_horizontal,
        margins,
        size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::ViewLayoutStore;

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
        fn render(
            &self,
            ctx: &ViewContext,
            surf: TerminalSurface<'_>,
            layout: ViewLayout<'_>,
        ) -> Result<(), Error> {
            self.view.render(ctx, surf, layout)
        }

        fn layout(
            &self,
            ctx: &ViewContext,
            ct: BoxConstraint,
            layout: ViewMutLayout<'_>,
        ) -> Result<(), Error> {
            let size = ct.clamp(self.size);
            self.view.layout(ctx, BoxConstraint::tight(size), layout)
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

        let mut layout_store = ViewLayoutStore::new();
        let mut reference_store = ViewLayoutStore::new();

        println!("{:?}", cont);
        println!("{:?}", cont.debug(size));
        let mut reference = ViewMutLayout::new(&mut reference_store, Layout::new().with_size(size));
        reference.push(
            Layout::new()
                .with_position(Position::new(2, 6))
                .with_size(Size::new(1, 4)),
        );
        assert_eq!(
            reference,
            cont.layout_new(&ctx, BoxConstraint::loose(size), &mut layout_store)?
        );

        let cont = cont.with_horizontal(Align::Start);
        println!("{:?}", cont);
        println!("{:?}", cont.debug(size));
        let mut reference = ViewMutLayout::new(&mut reference_store, Layout::new().with_size(size));
        reference.push(
            Layout::new()
                .with_position(Position::new(2, 0))
                .with_size(Size::new(1, 4)),
        );
        assert_eq!(
            reference,
            cont.layout_new(&ctx, BoxConstraint::loose(size), &mut layout_store)?
        );

        let cont = cont.with_horizontal(Align::Center);
        println!("{:?}", cont);
        println!("{:?}", cont.debug(size));
        let mut reference = ViewMutLayout::new(&mut reference_store, Layout::new().with_size(size));
        reference.push(
            Layout::new()
                .with_position(Position::new(2, 3))
                .with_size(Size::new(1, 4)),
        );
        assert_eq!(
            reference,
            cont.layout_new(&ctx, BoxConstraint::loose(size), &mut layout_store)?
        );

        let cont = cont
            .with_horizontal(Align::Offset(-2))
            .with_vertical(Align::Offset(1));
        println!("{:?}", cont);
        println!("{:?}", cont.debug(size));
        let mut reference = ViewMutLayout::new(&mut reference_store, Layout::new().with_size(size));
        reference.push(
            Layout::new()
                .with_position(Position::new(1, 4))
                .with_size(Size::new(1, 4)),
        );
        assert_eq!(
            reference,
            cont.layout_new(&ctx, BoxConstraint::loose(size), &mut layout_store)?
        );

        let cont = cont.with_vertical(Align::Expand);
        println!("{:?}", cont);
        println!("{:?}", cont.debug(size));
        let mut reference = ViewMutLayout::new(&mut reference_store, Layout::new().with_size(size));
        reference.push(
            Layout::new()
                .with_position(Position::new(0, 4))
                .with_size(Size::new(5, 4)),
        );
        assert_eq!(
            reference,
            cont.layout_new(&ctx, BoxConstraint::loose(size), &mut layout_store)?
        );

        let cont = cont.with_margins(Margins {
            top: 1,
            bottom: 0,
            left: 2,
            right: 1,
        });
        println!("{:?}", cont);
        println!("{:?}", cont.debug(size));
        let mut reference = ViewMutLayout::new(&mut reference_store, Layout::new().with_size(size));
        reference.push(
            Layout::new()
                .with_position(Position::new(1, 3))
                .with_size(Size::new(4, 4)),
        );
        assert_eq!(
            reference,
            cont.layout_new(&ctx, BoxConstraint::loose(size), &mut layout_store)?
        );

        let cont = cont.with_vertical(Align::Shrink);
        println!("{:?}", cont);
        println!("{:?}", cont.debug(size));
        let mut reference = ViewMutLayout::new(
            &mut reference_store,
            Layout::new().with_size(Size::new(2, 10)),
        );
        reference.push(
            Layout::new()
                .with_position(Position::new(1, 3))
                .with_size(Size::new(1, 4)),
        );
        assert_eq!(
            reference,
            cont.layout_new(&ctx, BoxConstraint::loose(size), &mut layout_store)?
        );

        Ok(())
    }
}
