use super::{AlongAxis, Axis, BoxConstraint, Layout, View, ViewContext, ViewLayout, ViewMutLayout};
use crate::{
    Cell, CellWrite, Error, Face, FaceAttrs, Position, Size, TerminalSurface, TerminalSurfaceExt,
};
use std::cmp::max;

/// Represent current position
///
/// offset - fraction (0.0..1.0) scrolled from the begging
/// visible - fraction (0.0..1.0) visible on the screen
#[derive(Debug)]
pub struct ScrollBarPosition {
    pub offset: f64,
    pub visible: f64,
}

impl ScrollBarPosition {
    pub fn from_counts(total: usize, offset: usize, visible: usize) -> Self {
        let total = total as f64;
        Self {
            offset: offset as f64 / total,
            visible: visible as f64 / total,
        }
    }
}

#[derive(Debug)]
pub struct ScrollBar {
    direction: Axis,
    face: Face,
    position: ScrollBarPosition,
}

impl ScrollBar {
    /// Create scroll bar along the `axis`
    pub fn new(direction: Axis, face: Face, position: ScrollBarPosition) -> Self {
        Self {
            direction,
            face,
            position,
        }
    }
}

impl View for ScrollBar {
    fn render(
        &self,
        ctx: &ViewContext,
        surf: TerminalSurface<'_>,
        layout: ViewLayout<'_>,
    ) -> Result<(), Error> {
        let major = self.direction.major(layout.size());
        if major == 0 {
            return Ok(());
        }

        let (size, offset) = {
            let major = major as f64;
            let size = (major * self.position.visible).clamp(1.0, major).round();
            let offset = ((major - size) * self.position.offset).round();
            (size as usize, offset as usize)
        };

        let mut surf = layout.apply_to(surf);
        let mut writer = surf.writer(ctx);
        let fg = Face::new(None, self.face.fg, FaceAttrs::EMPTY);
        let bg = Face::new(None, self.face.bg, FaceAttrs::EMPTY);
        for index in 0..major {
            if index < offset || index >= offset + size {
                writer.put_cell(Cell::new_char(bg, ' '));
            } else {
                writer.put_cell(Cell::new_char(fg, ' '));
            }
        }
        Ok(())
    }

    fn layout(
        &self,
        _ctx: &ViewContext,
        ct: BoxConstraint,
        layout: ViewMutLayout<'_>,
    ) -> Result<(), Error> {
        scroll_bar_layout(self.direction, ct, layout)
    }
}

/// Scroll bar that evaluates position during render
pub struct ScrollBarFn<F> {
    direction: Axis,
    face: Face,
    position: F,
}

impl<F> ScrollBarFn<F> {
    pub fn new(direction: Axis, face: Face, position: F) -> Self {
        Self {
            direction,
            face,
            position,
        }
    }
}

impl<F> View for ScrollBarFn<F>
where
    F: Fn() -> ScrollBarPosition + Send + Sync,
{
    fn render(
        &self,
        ctx: &ViewContext,
        surf: TerminalSurface<'_>,
        layout: ViewLayout<'_>,
    ) -> Result<(), Error> {
        ScrollBar::new(self.direction, self.face, (self.position)()).render(ctx, surf, layout)
    }

    fn layout(
        &self,
        _ctx: &ViewContext,
        ct: BoxConstraint,
        layout: ViewMutLayout<'_>,
    ) -> Result<(), Error> {
        scroll_bar_layout(self.direction, ct, layout)
    }
}

fn scroll_bar_layout(
    direction: Axis,
    ct: BoxConstraint,
    mut layout: ViewMutLayout<'_>,
) -> Result<(), Error> {
    let major = direction.major(ct.max());
    let minor = max(direction.minor(ct.min()), 1);
    *layout = Layout::new()
        .with_size(Size::from_axes(direction, major, 1))
        .with_position(Position::from_axes(direction, 0, minor - 1));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Surface,
        view::{ViewLayoutStore, tests::render},
    };

    #[test]
    fn test_scroll_bar() -> Result<(), Error> {
        let bg = Some("#800000".parse()?);
        let fg = Some("#008000".parse()?);
        let bar = ScrollBar::new(
            Axis::Horizontal,
            Face::new(fg, bg, FaceAttrs::EMPTY),
            ScrollBarPosition {
                offset: 0.2,
                visible: 0.3,
            },
        );
        let ctx = &ViewContext::dummy();
        let size = Size::new(5, 20);
        let mut layout_store = ViewLayoutStore::new();
        let layout = bar.layout_new(ctx, BoxConstraint::loose(size), &mut layout_store)?;
        let mut reference_store = ViewLayoutStore::new();
        let reference = ViewLayout::new(
            &mut reference_store,
            Layout::new().with_size(Size::new(1, 20)),
        );
        assert_eq!(reference, layout);
        print!("{:?}", bar.debug(size));

        let surf = render(ctx, &bar, size)?;
        surf.view(0u32, 0..3u32)
            .iter()
            .enumerate()
            .for_each(|(index, cell)| assert_eq!(cell.face().bg, bg, "at {index}"));
        surf.view(0u32, 3..9u32)
            .iter()
            .enumerate()
            .for_each(|(index, cell)| assert_eq!(cell.face().bg, fg, "at {index}"));
        surf.view(0u32, 9..size.width)
            .iter()
            .enumerate()
            .for_each(|(index, cell)| assert_eq!(cell.face().bg, bg, "at {index}"));

        Ok(())
    }
}
