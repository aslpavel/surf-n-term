use super::{
    Align, AlongAxis, Axis, BoxConstraint, IntoView, Layout, Tree, View, ViewContext,
    ViewDeserializer,
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

impl<'a> fmt::Debug for Child<'a> {
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
        self.push_child_ext(child, None, None, Align::Start)
    }

    pub fn push_flex_child(&mut self, flex: f64, child: impl IntoView + 'a) {
        self.push_child_ext(child, Some(flex), None, Align::Start)
    }

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

    /// Add new flex size child
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
                            view: seed.deserialize(view)?,
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
        layout: &Tree<Layout>,
    ) -> Result<(), Error> {
        let mut surf = layout.apply_to(surf);
        for (index, child) in self.children.iter().enumerate() {
            let child_layout = layout.get(index).ok_or(Error::InvalidLayout)?;
            if child_layout.size.is_empty() {
                continue;
            }

            // fill allocated space
            if let Some(face) = child.face {
                let mut surf = match self.direction {
                    Axis::Horizontal => {
                        let start = child_layout.pos.col;
                        let end = start + child_layout.size.width;
                        surf.view_mut(.., start..end)
                    }
                    Axis::Vertical => {
                        let start = child_layout.pos.row;
                        let end = start + child_layout.size.height;
                        surf.view_mut(start..end, ..)
                    }
                };
                surf.erase(face);
            }

            child.view.render(ctx, surf.as_mut(), child_layout)?;
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
            match child.flex {
                None => {
                    let child_layout = child.view.layout(ctx, ct_loosen);
                    major_non_flex += self.direction.major(child_layout.size);
                    minor = max(minor, self.direction.minor(child_layout.size));
                    children_layout[index] = child_layout;
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
            for (index, child) in self.children.iter().enumerate() {
                if let Some(flex) = child.flex {
                    // compute available flex
                    let child_major_max =
                        ((major_remain as f64) * flex / flex_total).round() as usize;
                    flex_total -= flex;
                    if child_major_max == 0 {
                        continue;
                    }

                    // layout child
                    let child_ct = self.direction.constraint(ct_loosen, 0, child_major_max);
                    let child_layout = child.view.layout(ctx, child_ct);
                    let child_major = self.direction.major(child_layout.size);
                    let child_minor = self.direction.minor(child_layout.size);
                    children_layout[index] = child_layout;

                    // update counters
                    major_remain -= child_major;
                    major_flex += child_major;
                    minor = max(minor, child_minor);
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
        let mut major_offset = space_side;
        for (index, child_layout) in children_layout.iter_mut().enumerate() {
            child_layout.pos = Position::from_axes(
                self.direction,
                major_offset,
                self.children[index]
                    .align
                    .align(self.direction.minor(child_layout.size), minor),
            );

            major_offset += child_layout.size.major(self.direction);
            major_offset += space_between;
        }

        // create layout tree
        Tree::new(
            Layout::new().with_size(ct.clamp(Size::from_axes(self.direction, major_offset, minor))),
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

        let size = Size::new(5, 12);
        print!("[flex] squeeze {:?}", flex.debug(size));
        let reference = Tree::new(
            Layout::new().with_size(Size::new(3, 12)),
            vec![
                Tree::leaf(
                    Layout::new()
                        .with_position(Position::new(1, 0))
                        .with_size(Size::new(2, 8)),
                ),
                Tree::leaf(
                    Layout::new()
                        .with_position(Position::new(0, 8))
                        .with_size(Size::new(3, 4)),
                ),
            ],
        );
        assert_eq!(reference, flex.layout(&ctx, BoxConstraint::loose(size)));

        let size = Size::new(5, 40);
        print!("[flex] justify start {:?}", flex.debug(size));
        let reference = Tree::new(
            Layout::new().with_size(Size::new(1, 19)),
            vec![
                Tree::leaf(
                    Layout::new()
                        .with_position(Position::new(0, 0))
                        .with_size(Size::new(1, 9)),
                ),
                Tree::leaf(
                    Layout::new()
                        .with_position(Position::new(0, 9))
                        .with_size(Size::new(1, 10)),
                ),
            ],
        );
        assert_eq!(reference, flex.layout(&ctx, BoxConstraint::loose(size)));

        let size = Size::new(5, 40);
        let flex = flex.justify(Justify::SpaceBetween);
        print!("[flex] justify space between {:?}", flex.debug(size));
        let reference = Tree::new(
            Layout::new().with_size(Size::new(1, 40)),
            vec![
                Tree::leaf(
                    Layout::new()
                        .with_position(Position::new(0, 0))
                        .with_size(Size::new(1, 9)),
                ),
                Tree::leaf(
                    Layout::new()
                        .with_position(Position::new(0, 30))
                        .with_size(Size::new(1, 10)),
                ),
            ],
        );
        assert_eq!(reference, flex.layout(&ctx, BoxConstraint::loose(size)));

        Ok(())
    }
}
