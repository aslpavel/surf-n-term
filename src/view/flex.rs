use super::{
    Align, AlongAxis, Axis, BoxConstraint, IntoView, Layout, Tree, TreeMut, View, ViewContext,
    ViewDeserializer, ViewLayout, ViewMutLayout,
};
use crate::{Error, Face, Position, Size, SurfaceMut, TerminalSurface, TerminalSurfaceExt};
use serde::{de::DeserializeSeed, Deserialize, Serialize};
use std::{cmp::max, fmt};

struct Child<'a> {
    view: Box<dyn View + 'a>,
    flex: Option<f64>,
    face: Option<Face>,
    align: Align,
}

impl fmt::Debug for Child<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Child")
            .field("flex", &self.flex)
            .field("face", &self.face)
            .field("align", &self.align)
            .finish()
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub enum Justify {
    #[serde(rename = "start")]
    Start,
    #[serde(rename = "center")]
    Center,
    #[serde(rename = "end")]
    End,
    #[serde(rename = "space-between")]
    SpaceBetween,
    #[serde(rename = "space-around")]
    SpaceAround,
    #[serde(rename = "space-evenly")]
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

    /// How to justify/align children along major axis
    pub fn justify(self, justify: Justify) -> Self {
        Self { justify, ..self }
    }

    /// Create flex with horizontal major axis
    pub fn row() -> Self {
        Self::new(Axis::Horizontal)
    }

    /// Create flex with vertical major axis
    pub fn column() -> Self {
        Self::new(Axis::Vertical)
    }

    /// Push non-flex child
    pub fn push_child(&mut self, child: impl IntoView + 'a) {
        self.push_child_ext(child, None, None, Align::Start)
    }

    /// Push flex child
    pub fn push_flex_child(&mut self, flex: f64, child: impl IntoView + 'a) {
        self.push_child_ext(child, Some(flex), None, Align::Start)
    }

    /// Push child with extended attributes
    pub fn push_child_ext(
        &mut self,
        child: impl IntoView + 'a,
        flex: Option<f64>,
        face: Option<Face>,
        align: Align,
    ) {
        self.children.push(Child {
            view: child.into_view().boxed(),
            flex: flex.and_then(|flex| (flex > 0.0).then_some(flex)),
            face,
            align,
        });
    }

    /// Add new fixed size child
    pub fn add_child(mut self, child: impl IntoView + 'a) -> Self {
        self.push_child(child);
        self
    }

    /// Add new child with extended attributes
    pub fn add_child_ext(
        mut self,
        child: impl IntoView + 'a,
        flex: Option<f64>,
        face: Option<Face>,
        align: Align,
    ) -> Self {
        self.push_child_ext(child, flex, face, align);
        self
    }

    /// Add new flex sized child
    pub fn add_flex_child(mut self, flex: f64, child: impl IntoView + 'a) -> Self {
        self.push_flex_child(flex, child);
        self
    }

    /// Construct [Flex] object from JSON value
    pub(super) fn from_json_value(
        seed: &ViewDeserializer<'_>,
        value: &serde_json::Value,
    ) -> Result<Self, Error> {
        let direction = match value.get("direction") {
            None => Axis::Horizontal,
            Some(axis) => Axis::deserialize(axis)?,
        };
        let justify = match value.get("justify") {
            None => Justify::Start,
            Some(justify) => Justify::deserialize(justify)?,
        };
        let children = match value.get("children") {
            None => Vec::new(),
            Some(serde_json::Value::Array(values)) => {
                let mut children = Vec::with_capacity(values.len());
                for value in values {
                    if value.get("type").is_some() {
                        children.push(Child {
                            view: seed.deserialize(value)?.boxed(),
                            flex: None,
                            face: None,
                            align: Align::default(),
                        })
                    } else {
                        let flex = value.get("flex").map(f64::deserialize).transpose()?;
                        let align = value
                            .get("align")
                            .map(Align::deserialize)
                            .transpose()?
                            .unwrap_or_default();
                        let face = value
                            .get("face")
                            .map(|value| seed.face(value))
                            .transpose()?;
                        let view = value.get("view").ok_or_else(|| {
                            Error::ParseError(
                                "Flex",
                                "child must include view attribute".to_owned(),
                            )
                        })?;
                        children.push(Child {
                            view: seed.deserialize(view)?.boxed(),
                            flex,
                            face,
                            align,
                        })
                    }
                }
                children
            }
            _ => {
                return Err(Error::ParseError(
                    "Flex",
                    "children must be an array".to_owned(),
                ))
            }
        };
        Ok(Flex {
            direction,
            justify,
            children,
        })
    }
}

impl<'a> View for Flex<'a>
where
    Self: 'a,
{
    fn render(
        &self,
        ctx: &ViewContext,
        surf: TerminalSurface<'_>,
        layout: ViewLayout<'_>,
    ) -> Result<(), Error> {
        let mut surf = layout.apply_to(surf);
        for (child, child_layout) in self.children.iter().zip(layout.children()) {
            if child_layout.size().is_empty() {
                continue;
            }

            // fill allocated space
            if let Some(face) = child.face {
                let mut surf = match self.direction {
                    Axis::Horizontal => {
                        let start = child_layout.position().col;
                        let end = start + child_layout.size().width;
                        surf.view_mut(.., start..end)
                    }
                    Axis::Vertical => {
                        let start = child_layout.position().row;
                        let end = start + child_layout.size().height;
                        surf.view_mut(start..end, ..)
                    }
                };
                surf.erase(face);
            }

            child.view.render(ctx, surf.as_mut(), child_layout)?;
        }
        Ok(())
    }

    fn layout(
        &self,
        ctx: &ViewContext,
        ct: BoxConstraint,
        mut layout: ViewMutLayout<'_>,
    ) -> Result<(), Error> {
        let mut flex_total = 0.0;
        let mut major_non_flex = 0;
        let mut minor = self.direction.minor(ct.min());
        let ct_loosen = ct.loosen();

        // layout non-flex
        for child in self.children.iter() {
            let mut child_layout = layout.push_default();
            match child.flex {
                None => {
                    child.view.layout(ctx, ct_loosen, child_layout.view_mut())?;
                    major_non_flex += self.direction.major(child_layout.size());
                    minor = max(minor, self.direction.minor(child_layout.size()));
                }
                Some(flex) => flex_total += flex,
            }
        }

        // layout flex
        let mut major_remain = self
            .direction
            .major(ct.max())
            .saturating_sub(major_non_flex);
        let mut major_flex = 0;
        if major_remain > 0 && flex_total > 0.0 {
            let mut child_layout_opt = layout.child_mut();
            for child in self.children.iter() {
                let mut child_layout =
                    child_layout_opt.expect("not all flex children are allocated");
                if let Some(flex) = child.flex {
                    // compute available flex
                    let child_major_max =
                        ((major_remain as f64) * flex / flex_total).round() as usize;
                    flex_total -= flex;
                    if child_major_max != 0 {
                        // layout child
                        let child_ct = self.direction.constraint(ct_loosen, 0, child_major_max);
                        child.view.layout(ctx, child_ct, child_layout.view_mut())?;
                        let child_major = self.direction.major(child_layout.size());
                        let child_minor = self.direction.minor(child_layout.size());

                        // update counters
                        major_remain -= child_major;
                        major_flex += child_major;
                        minor = max(minor, child_minor);
                    }
                }
                child_layout_opt = child_layout.sibling();
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

        // calculate positions
        let mut major_offset = space_side;
        if !self.children.is_empty() {
            let mut child_layout_opt = layout.child_mut();
            for child in self.children.iter() {
                let mut child_layout =
                    child_layout_opt.expect("not all flex children are allocated");
                let child_size = child_layout.size();
                child_layout.set_position(Position::from_axes(
                    self.direction,
                    major_offset,
                    child.align.align(self.direction.minor(child_size), minor),
                ));

                major_offset += child_size.major(self.direction);
                major_offset += space_between;

                child_layout_opt = child_layout.sibling();
            }
        }

        // set layout
        *layout =
            Layout::new().with_size(ct.clamp(Size::from_axes(self.direction, major_offset, minor)));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::{Text, ViewLayoutStore};

    #[test]
    fn test_flex() -> Result<(), Error> {
        let ctx = ViewContext::dummy();
        let flex = Flex::row()
            .add_child_ext(
                Text::from("some text")
                    .mark("fg=#ebdbb2,bg=#458588".parse()?, ..)
                    .take(),
                Some(2.0),
                Some("bg=#83a598".parse()?),
                Align::End,
            )
            .add_child_ext(
                Text::from("other text")
                    .mark("fg=#ebdbb2,bg=#b16286".parse()?, ..)
                    .take(),
                Some(1.0),
                Some("bg=#d3869b".parse()?),
                Align::Start,
            );
        let mut reference_store = ViewLayoutStore::new();
        let mut result_store = ViewLayoutStore::new();

        let size = Size::new(5, 12);
        print!("[flex] squeeze {:?}", flex.debug(size));
        let mut reference_layout = ViewMutLayout::new(
            &mut reference_store,
            Layout::new().with_size(Size::new(3, 12)),
        );
        reference_layout.push(
            Layout::new()
                .with_position(Position::new(1, 0))
                .with_size(Size::new(2, 8)),
        );
        reference_layout.push(
            Layout::new()
                .with_position(Position::new(0, 8))
                .with_size(Size::new(3, 4)),
        );
        let result_layout = flex.layout_new(&ctx, BoxConstraint::loose(size), &mut result_store)?;
        assert_eq!(reference_layout, result_layout);

        let size = Size::new(5, 40);
        print!("[flex] justify start {:?}", flex.debug(size));
        let mut reference_layout = ViewMutLayout::new(
            &mut reference_store,
            Layout::new().with_size(Size::new(1, 19)),
        );
        reference_layout.push(
            Layout::new()
                .with_position(Position::new(0, 0))
                .with_size(Size::new(1, 9)),
        );
        reference_layout.push(
            Layout::new()
                .with_position(Position::new(0, 9))
                .with_size(Size::new(1, 10)),
        );
        let result_layout = flex.layout_new(&ctx, BoxConstraint::loose(size), &mut result_store)?;
        assert_eq!(reference_layout, result_layout);

        let size = Size::new(5, 40);
        let flex = flex.justify(Justify::SpaceBetween);
        print!("[flex] justify space between {:?}", flex.debug(size));
        let mut reference_layout = ViewMutLayout::new(
            &mut reference_store,
            Layout::new().with_size(Size::new(1, 40)),
        );
        reference_layout.push(
            Layout::new()
                .with_position(Position::new(0, 0))
                .with_size(Size::new(1, 9)),
        );
        reference_layout.push(
            Layout::new()
                .with_position(Position::new(0, 30))
                .with_size(Size::new(1, 10)),
        );
        let result_layout = flex.layout_new(&ctx, BoxConstraint::loose(size), &mut result_store)?;
        assert_eq!(reference_layout, result_layout);

        Ok(())
    }
}
