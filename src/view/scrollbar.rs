use super::{AlongAxis, Axis, BoxConstraint, Layout, Tree, View, ViewContext};
use crate::{Error, Face, FaceAttrs, Position, Size, TerminalSurface, TerminalSurfaceExt};
use std::cmp::{max, min, Ordering};

#[derive(Debug)]
pub struct ScrollBar {
    direction: Axis,
    face: Face,
    total: usize,
    offset: usize,
    visible: usize,
}

impl ScrollBar {
    /// Create scroll bar along the `axis`. For something that has `total` the number
    /// of entries with the current `offset`, and the number of `visible` values.
    pub fn new(direction: Axis, face: Face, total: usize, offset: usize, visible: usize) -> Self {
        Self {
            direction,
            face,
            total,
            offset: min(total, offset),
            visible: min(total, visible),
        }
    }
}

impl View for ScrollBar {
    fn render<'a>(
        &self,
        ctx: &ViewContext,
        surf: &'a mut TerminalSurface<'a>,
        layout: &Tree<Layout>,
    ) -> Result<(), Error> {
        let major = self.direction.major(layout.size);
        if major == 0 {
            return Ok(());
        }

        let (size, offset) = match self.total.cmp(&1) {
            Ordering::Greater => {
                let major = major as f64;
                let total = self.total as f64;
                let offset = self.offset as f64;
                let visible = self.visible as f64;
                let size = (major * visible / total).clamp(1.0, major).round();
                let offset = ((major - size) * offset / (total - 1.0)).round();
                (size as usize, offset as usize)
            }
            Ordering::Equal => (major, 0),
            Ordering::Less => (0, 0),
        };

        let mut surf = layout.apply_to(surf);
        let mut writer = surf.writer(ctx);
        let fg = Face::new(None, self.face.fg, FaceAttrs::EMPTY);
        let bg = Face::new(None, self.face.bg, FaceAttrs::EMPTY);
        for index in 0..major {
            if index < offset || index >= offset + size {
                writer.put_char(' ', bg);
            } else {
                writer.put_char(' ', fg);
            }
        }
        Ok(())
    }

    fn layout(&self, _ctx: &ViewContext, ct: BoxConstraint) -> Tree<Layout> {
        let major = self.direction.major(ct.max());
        let minor = max(self.direction.minor(ct.min()), 1);
        Tree::leaf(
            Layout::new()
                .with_size(Size::from_axes(self.direction, major, 1))
                .with_position(Position::from_axes(self.direction, 0, minor - 1)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{view::tests::render, Surface};

    #[test]
    fn test_scroll_bar() -> Result<(), Error> {
        let bg = Some("#800000".parse()?);
        let fg = Some("#008000".parse()?);
        let bar = ScrollBar::new(
            Axis::Horizontal,
            Face::new(fg, bg, FaceAttrs::EMPTY),
            100,
            20,
            30,
        );
        let ctx = &ViewContext::dummy();
        let size = Size::new(5, 20);
        assert_eq!(
            bar.layout(ctx, BoxConstraint::loose(size)),
            Tree::leaf(Layout::new().with_size(Size::new(1, 20)))
        );
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
