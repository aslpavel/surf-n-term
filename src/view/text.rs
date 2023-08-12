use super::{BoxConstraint, Layout, Tree, View, ViewContext};
use crate::{
    surface::ViewBounds, Cell, Error, Face, Glyph, Position, Size, TerminalSurface,
    TerminalSurfaceExt,
};
use std::fmt::Write as _;

impl View for str {
    fn render<'b>(
        &self,
        ctx: &ViewContext,
        surf: &'b mut TerminalSurface<'b>,
        layout: &Tree<Layout>,
    ) -> Result<(), Error> {
        let mut surf = layout.apply_to(surf);
        let mut writer = surf.writer(ctx);
        self.chars().for_each(|c| {
            writer.put_char(c, Face::default());
        });
        Ok(())
    }

    fn layout(&self, ctx: &ViewContext, ct: BoxConstraint) -> Tree<Layout> {
        let mut size = Size::empty();
        let mut cursor = Position::origin();
        let face = Face::default();
        self.chars().for_each(|c| {
            Cell::new_char(face, c).layout(ctx, ct.max.width, &mut size, &mut cursor);
        });
        Tree::leaf(Layout::new().with_size(ct.clamp(size)))
    }
}

impl View for String {
    fn render<'a>(
        &self,
        ctx: &ViewContext,
        surf: &'a mut TerminalSurface<'a>,
        layout: &Tree<Layout>,
    ) -> Result<(), Error> {
        self.as_str().render(ctx, surf, layout)
    }

    fn layout(&self, ctx: &ViewContext, ct: BoxConstraint) -> Tree<Layout> {
        self.as_str().layout(ctx, ct)
    }
}

#[derive(Clone, Default, Debug)]
pub struct Text {
    cells: Vec<Cell>,
    // face used write next symbol (not actual face of the text)
    face: Face,
}

impl Text {
    /// Create new empty text
    pub fn new() -> Self {
        Default::default()
    }

    /// Calculate maximum size of the text
    pub fn max_size(&self) -> Size {
        self.layout(
            &ViewContext::dummy(),
            BoxConstraint::loose(Size::new(usize::MAX, usize::MAX)),
        )
        .size()
    }

    /// Number of cells
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// Check if [Text] is empty
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Remove all cell and reset face
    pub fn clear(&mut self) {
        self.cells.clear();
        self.face = Face::default();
    }

    /// Currently set face
    pub fn face(&self) -> &Face {
        &self.face
    }

    /// Overlay face for the length of the scope
    pub fn with_face(&mut self, face: Face, scope: impl FnOnce(&mut Self)) -> &mut Self {
        let face_old = self.face;
        scope(self.set_face(face_old.overlay(&face)));
        self.set_face(face_old);
        self
    }

    /// Set face that will to add new cells
    pub fn set_face(&mut self, face: Face) -> &mut Self {
        self.face = face;
        self
    }

    /// Append character to the end of the [Text]
    pub fn put_char(&mut self, c: char) -> &mut Self {
        self.cells.push(Cell::new_char(self.face, c));
        self
    }

    /// Append a given string to the end of [Text]
    pub fn push_str(&mut self, str: &str, face: Option<Face>) -> &mut Self {
        self.with_face(face.unwrap_or_default(), |text| {
            str.chars().for_each(|c| {
                text.put_char(c);
            })
        });
        self
    }

    /// Append glyph to the end of the [Text]
    pub fn put_glyph(&mut self, glyph: Glyph) -> &mut Self {
        self.cells.push(Cell::new_glyph(self.face, glyph));
        self
    }

    /// Put cell
    pub fn put_cell(&mut self, cell: Cell) -> &mut Self {
        let face = self.face.overlay(&cell.face());
        self.cells.push(cell.with_face(face));
        self
    }

    /// Append text to the end of the [Text]
    pub fn push_text(&mut self, text: &Text) -> &mut Self {
        text.cells.iter().cloned().for_each(|cell| {
            self.put_cell(cell);
        });
        self
    }

    /// Append format argument to the end of the [Text]
    pub fn push_fmt(&mut self, args: std::fmt::Arguments<'_>) -> &mut Self {
        self.write_fmt(args).expect("in memory write failed");
        self
    }

    /// Mark range of cell with the provided face
    pub fn mark(&mut self, face: Face, bounds: impl ViewBounds) -> &mut Self {
        if let Some((start, end)) = bounds.view_bounds(self.cells.len()) {
            self.cells[start..end].iter_mut().for_each(|cell| {
                *cell = cell.clone().with_face(cell.face().overlay(&face));
            })
        }
        self
    }

    /// Take current value replacing it with the default
    pub fn take(&mut self) -> Self {
        std::mem::replace(self, Text::new())
    }
}

impl std::fmt::Write for Text {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        s.chars().for_each(|c| {
            self.put_char(c);
        });
        Ok(())
    }
}

impl<'a> From<&'a str> for Text {
    fn from(value: &'a str) -> Self {
        let mut span = Text::new();
        span.write_str(value).expect("memory write failed");
        span
    }
}

impl View for Text {
    fn render<'a>(
        &self,
        ctx: &ViewContext,
        surf: &'a mut TerminalSurface<'a>,
        layout: &Tree<Layout>,
    ) -> Result<(), Error> {
        let mut surf = layout.apply_to(surf);
        let mut writer = surf.writer(ctx);
        self.cells.iter().for_each(|cell| {
            writer.put(cell.clone());
        });
        Ok(())
    }

    fn layout(&self, ctx: &ViewContext, ct: BoxConstraint) -> Tree<Layout> {
        let mut size = Size::empty();
        let mut cursor = Position::origin();
        self.cells.iter().for_each(|cell| {
            cell.layout(ctx, ct.max.width, &mut size, &mut cursor);
        });
        Tree::leaf(Layout::new().with_size(ct.clamp(size)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_basic() -> Result<(), Error> {
        let tag = "[text basic]";
        let size = Size::new(5, 10);
        let ctx = ViewContext::dummy();
        let mut text = Text::new().set_face("fg=#ebdbb2".parse()?).take();

        writeln!(&mut text, "11")?;
        text.set_face("fg=#3c3836,bg=#ebdbb2".parse()?);
        writeln!(&mut text, "222")?;

        print!("{tag} first line longest: {:?}", text.debug(size));
        let layout = text.layout(&ctx, BoxConstraint::loose(size));
        assert_eq!(layout.size, Size::new(2, 3));

        text.set_face(text.face().overlay(&"fg=#af3a03,bold".parse()?));
        writeln!(&mut text, "3")?;

        print!("{tag} middle line longest: {:?}", text.debug(size));
        let layout = text.layout(&ctx, BoxConstraint::loose(size));
        assert_eq!(layout.size, Size::new(3, 3));

        Ok(())
    }

    #[test]
    fn test_text_wrap() -> Result<(), Error> {
        let tag = "[text wrap]";
        let size = Size::new(5, 10);
        let ctx = ViewContext::dummy();
        let mut text = Text::new().set_face("fg=#ebdbb2".parse()?).take();

        write!(&mut text, "1 no wrap")?;

        print!("{tag} will not wrap: {:?}", text.debug(size));
        let layout = text.layout(&ctx, BoxConstraint::loose(size));
        assert_eq!(layout.size, Size::new(1, 9));

        writeln!(&mut text, "\n2 this will wrap")?;
        print!("{tag} will wrap: {:?}", text.debug(size));
        let layout = text.layout(&ctx, BoxConstraint::loose(size));
        assert_eq!(layout.size, Size::new(3, 10));

        write!(&mut text, "3 bla")?;
        print!("{tag} one more line: {:?}", text.debug(size));
        let layout = text.layout(&ctx, BoxConstraint::loose(size));
        assert_eq!(layout.size, Size::new(4, 10));

        Ok(())
    }

    #[test]
    fn test_text_tab() -> Result<(), Error> {
        let tag = "[text tab]";
        let size = Size::new(6, 20);
        let ctx = ViewContext::dummy();
        let mut text = Text::new().set_face("fg=#ebdbb2".parse()?).take();

        writeln!(&mut text, ".\t|")?;
        print!("{tag} 1 char: {:?}", text.debug(size));
        let layout = text.layout(&ctx, BoxConstraint::loose(size));
        assert_eq!(layout.size, Size::new(1, 9));

        writeln!(&mut text, ".......\t|")?;
        print!("{tag} 7 chars: {:?}", text.debug(size));
        let layout = text.layout(&ctx, BoxConstraint::loose(size));
        assert_eq!(layout.size, Size::new(2, 9));

        writeln!(&mut text, "........\t|")?;
        print!("{tag} 8 chars: {:?}", text.debug(size));
        let layout = text.layout(&ctx, BoxConstraint::loose(size));
        assert_eq!(layout.size, Size::new(3, 17));

        writeln!(&mut text, "...............\t|")?;
        print!("{tag} 15 chars: {:?}", text.debug(size));
        let layout = text.layout(&ctx, BoxConstraint::loose(size));
        assert_eq!(layout.size, Size::new(4, 17));

        writeln!(&mut text, "................\t|")?;
        print!("{tag} 16 chars: {:?}", text.debug(size));
        let layout = text.layout(&ctx, BoxConstraint::loose(size));
        assert_eq!(layout.size, Size::new(6, 20));

        Ok(())
    }
}
