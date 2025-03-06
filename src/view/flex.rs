use super::{
    Align, AlongAxis, Axis, BoxConstraint, IntoView, Layout, Tree, TreeMut, View, ViewContext,
    ViewDeserializer, ViewLayout, ViewMutLayout,
};
use crate::{Error, Face, Position, Size, SurfaceMut, TerminalSurface, TerminalSurfaceExt};
use either::Either;
use serde::{Deserialize, Serialize, de::DeserializeSeed};
use std::{cmp::max, fmt};

pub struct FlexChild<V> {
    view: V,
    flex: Option<f64>,
    face: Option<Face>,
    align: Align,
}

impl<V> FlexChild<V> {
    pub fn new(view: V) -> Self {
        Self {
            view,
            flex: None,
            face: None,
            align: Align::Start,
        }
    }

    pub fn align(self, align: Align) -> Self {
        Self { align, ..self }
    }

    pub fn flex(self, flex: f64) -> Self {
        Self {
            flex: Some(flex),
            ..self
        }
    }

    pub fn face(self, face: Face) -> Self {
        Self {
            face: Some(face),
            ..self
        }
    }

    pub fn as_dyn(&self) -> FlexChild<&'_ dyn View>
    where
        V: View,
    {
        FlexChild {
            view: &self.view,
            flex: self.flex,
            face: self.face,
            align: self.align,
        }
    }

    pub fn boxed<'a>(self) -> FlexChild<Box<dyn View + 'a>>
    where
        V: View + 'a,
    {
        FlexChild {
            view: Box::new(self.view),
            flex: self.flex,
            face: self.face,
            align: self.align,
        }
    }
}

impl<V> From<V> for FlexChild<V::View>
where
    V: IntoView,
{
    fn from(view: V) -> Self {
        FlexChild::new(view.into_view())
    }
}

impl<V> fmt::Debug for FlexChild<V> {
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

pub struct FlexRef<A> {
    direction: Axis,
    justify: Justify,
    children: A,
}

impl<A> FlexRef<A> {
    pub fn new(children: A) -> Self {
        Self {
            direction: Axis::Horizontal,
            justify: Justify::Start,
            children,
        }
    }

    /// Create flex with horizontal major axis
    pub fn row(children: A) -> Self {
        Self::new(children).direction(Axis::Horizontal)
    }

    /// Create flex with vertical major axis
    pub fn column(children: A) -> Self {
        Self::new(children).direction(Axis::Vertical)
    }

    pub fn direction(self, direction: Axis) -> Self {
        Self { direction, ..self }
    }

    /// How to justify/align children along major axis
    pub fn justify(self, justify: Justify) -> Self {
        Self { justify, ..self }
    }
}

impl<A> View for FlexRef<A>
where
    A: FlexArray + Send + Sync,
{
    fn render(
        &self,
        ctx: &ViewContext,
        surf: TerminalSurface<'_>,
        layout: ViewLayout<'_>,
    ) -> Result<(), Error> {
        flex_render(self.direction, &self.children, ctx, surf, layout)
    }

    fn layout(
        &self,
        ctx: &ViewContext,
        ct: BoxConstraint,
        layout: ViewMutLayout<'_>,
    ) -> Result<(), Error> {
        flex_layout(
            self.direction,
            self.justify,
            &self.children,
            ctx,
            ct,
            layout,
        )
    }
}

#[derive(Debug)]
pub struct Flex<'a> {
    direction: Axis,
    justify: Justify,
    children: Vec<FlexChild<Box<dyn View + 'a>>>,
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
        self.children.push(FlexChild {
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
                        children.push(FlexChild {
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
                        children.push(FlexChild {
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
                ));
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
        flex_render(self.direction, self.children.as_slice(), ctx, surf, layout)
    }

    fn layout(
        &self,
        ctx: &ViewContext,
        ct: BoxConstraint,
        layout: ViewMutLayout<'_>,
    ) -> Result<(), Error> {
        flex_layout(
            self.direction,
            self.justify,
            self.children.as_slice(),
            ctx,
            ct,
            layout,
        )
    }
}

pub fn flex_layout(
    direction: Axis,
    justify: Justify,
    children: impl FlexArray,
    ctx: &ViewContext,
    ct: BoxConstraint,
    mut layout: ViewMutLayout<'_>,
) -> Result<(), Error> {
    let mut flex_total = 0.0;
    let mut major_non_flex = 0;
    let mut minor = direction.minor(ct.min());
    let ct_loosen = ct.loosen();

    // layout non-flex
    for child in children.iter() {
        let mut child_layout = layout.push_default();
        match child.flex {
            None => {
                child.view.layout(ctx, ct_loosen, child_layout.view_mut())?;
                major_non_flex += direction.major(child_layout.size());
                minor = max(minor, direction.minor(child_layout.size()));
            }
            Some(flex) => flex_total += flex,
        }
    }

    // layout flex
    let mut major_remain = direction.major(ct.max()).saturating_sub(major_non_flex);
    let mut major_flex = 0;
    if major_remain > 0 && flex_total > 0.0 {
        let mut child_layout_opt = layout.child_mut();
        for child in children.iter() {
            let mut child_layout = child_layout_opt.expect("not all flex children are allocated");
            if let Some(flex) = child.flex {
                // compute available flex
                let child_major_max = ((major_remain as f64) * flex / flex_total).round() as usize;
                flex_total -= flex;
                if child_major_max != 0 {
                    // layout child
                    let child_ct = direction.constraint(ct_loosen, 0, child_major_max);
                    child.view.layout(ctx, child_ct, child_layout.view_mut())?;
                    let child_major = direction.major(child_layout.size());
                    let child_minor = direction.minor(child_layout.size());

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
    let unused = direction
        .major(ct.max())
        .saturating_sub(major_non_flex + major_flex);
    let (space_side, space_between) = if unused > 0 {
        match justify {
            Justify::Start => (0, 0),
            Justify::Center => (unused / 2, 0),
            Justify::End => (unused, 0),
            Justify::SpaceBetween => {
                let space_between = if children.len() <= 1 {
                    unused
                } else {
                    unused / (children.len() - 1)
                };
                (0, space_between)
            }
            Justify::SpaceEvenly => {
                let space = unused / (children.len() + 1);
                (space, space)
            }
            Justify::SpaceAround => {
                let space = unused / children.len();
                (space / 2, space)
            }
        }
    } else {
        (0, 0)
    };

    // calculate positions
    let mut major_offset = space_side;
    if !children.is_empty() {
        let mut child_layout_opt = layout.child_mut();
        for child in children.iter() {
            let mut child_layout = child_layout_opt.expect("not all flex children are allocated");
            let child_size = child_layout.size();
            child_layout.set_position(Position::from_axes(
                direction,
                major_offset,
                child.align.align(direction.minor(child_size), minor),
            ));

            major_offset += child_size.major(direction);
            major_offset += space_between;

            child_layout_opt = child_layout.sibling();
        }
    }

    // set layout
    *layout = Layout::new().with_size(ct.clamp(Size::from_axes(direction, major_offset, minor)));
    Ok(())
}

pub fn flex_render(
    direction: Axis,
    children: impl FlexArray,
    ctx: &ViewContext,
    surf: TerminalSurface<'_>,
    layout: ViewLayout<'_>,
) -> Result<(), Error> {
    let mut surf = layout.apply_to(surf);
    for (child, child_layout) in children.iter().zip(layout.children()) {
        if child_layout.size().is_empty() {
            continue;
        }

        // fill allocated space
        if let Some(face) = child.face {
            let mut surf = match direction {
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

pub trait FlexArray {
    fn len(&self) -> usize;

    fn get(&self, index: usize) -> Option<FlexChild<&dyn View>>;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn iter(&self) -> FlexArrayIter<'_, Self> {
        FlexArrayIter {
            index: 0,
            array: self,
        }
    }
}

impl<T> FlexArray for &T
where
    T: FlexArray + ?Sized,
{
    fn len(&self) -> usize {
        (**self).len()
    }

    fn get(&self, index: usize) -> Option<FlexChild<&dyn View>> {
        (**self).get(index)
    }
}

impl<L, R> FlexArray for Either<L, R>
where
    L: FlexArray,
    R: FlexArray,
{
    fn len(&self) -> usize {
        match &self {
            Either::Left(left) => left.len(),
            Either::Right(right) => right.len(),
        }
    }

    fn get(&self, index: usize) -> Option<FlexChild<&dyn View>> {
        match &self {
            Either::Left(left) => left.get(index),
            Either::Right(right) => right.get(index),
        }
    }
}

impl<V: View> FlexArray for &[FlexChild<V>] {
    fn len(&self) -> usize {
        <[_]>::len(self)
    }
    fn get(&self, index: usize) -> Option<FlexChild<&dyn View>> {
        <[_]>::get(self, index).map(|child| child.as_dyn())
    }
}

impl<V: View, const N: usize> FlexArray for [FlexChild<V>; N] {
    fn len(&self) -> usize {
        N
    }
    fn get(&self, index: usize) -> Option<FlexChild<&dyn View>> {
        <[_]>::get(self, index).map(|child| child.as_dyn())
    }
}

impl<V: View> FlexArray for Vec<FlexChild<V>> {
    fn len(&self) -> usize {
        <[_]>::len(self.as_slice())
    }

    fn get(&self, index: usize) -> Option<FlexChild<&dyn View>> {
        <[_]>::get(self.as_slice(), index).map(|child| child.as_dyn())
    }
}

macro_rules! impl_flex_array_tuple {
    ($($idx:tt $name:ident),+) => (
        impl<$($name: View),*> FlexArray for ($(FlexChild<$name>,)*) {
            fn len(&self) -> usize {
                [$($idx,)*].len()
            }
            fn get(&self, index: usize) -> Option<FlexChild<&dyn View>> {
                match index {
                    $($idx => Some(self.$idx.as_dyn()),)*
                    _ => None,
                }
            }
        }
    );
}

impl_flex_array_tuple!(0 A);
impl_flex_array_tuple!(0 A, 1 B);
impl_flex_array_tuple!(0 A, 1 B, 2 C);
impl_flex_array_tuple!(0 A, 1 B, 2 C, 3 D);
impl_flex_array_tuple!(0 A, 1 B, 2 C, 3 D, 4 E);
impl_flex_array_tuple!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F);
impl_flex_array_tuple!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G);
impl_flex_array_tuple!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H);
impl_flex_array_tuple!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H, 8 I);
impl_flex_array_tuple!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H, 8 I, 9 J);
impl_flex_array_tuple!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H, 8 I, 9 J, 10 K);
impl_flex_array_tuple!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H, 8 I, 9 J, 10 K, 11 L);
impl_flex_array_tuple!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H, 8 I, 9 J, 10 K, 11 L, 12 M);

pub struct FlexArrayIter<'a, A: ?Sized> {
    index: usize,
    array: &'a A,
}

impl<'a, A: FlexArray> Iterator for FlexArrayIter<'a, A> {
    type Item = FlexChild<&'a dyn View>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.array.len() <= self.index {
            return None;
        }
        self.index += 1;
        self.array.get(self.index - 1)
    }
}

#[cfg(test)]
mod tests {
    use rasterize::RGBA;

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

    #[test]
    fn test_flex_ref() -> Result<(), Error> {
        let _: Box<dyn View> = FlexRef::row((
            "#ff0000".parse::<RGBA>()?.into(),
            FlexChild::new("#00ff00".parse::<RGBA>()?),
        ))
        .boxed();
        Ok(())
    }
}
