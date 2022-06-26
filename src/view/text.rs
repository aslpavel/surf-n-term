use super::{BoxConstraint, Layout, Tree, View};
use crate::{
    decoder::{Decoder, Utf8Decoder},
    Cell, Error, Face, Glyph, Position, Size, TerminalSurface, TerminalSurfaceExt, TerminalWriter,
};
use std::{
    borrow::Cow,
    cmp::max,
    io::{Cursor, Write},
};

#[derive(Debug, Clone, Default)]
pub struct Text<'a> {
    text: Cow<'a, str>,
    glyph: Option<Glyph>,
    face: Face,
    children: Vec<Text<'a>>,
}

impl<'a> Text<'a> {
    pub fn text(text: impl Into<Cow<'a, str>>) -> Self {
        Self {
            text: text.into(),
            glyph: None,
            face: Face::default(),
            children: Vec::new(),
        }
    }

    pub fn glyph(glyph: Glyph) -> Self {
        Self {
            text: Cow::Borrowed(""),
            glyph: Some(glyph),
            face: Face::default(),
            children: Vec::new(),
        }
    }

    /// Length of the text
    pub fn len(&self) -> usize {
        let len = self
            .glyph
            .as_ref()
            .map_or_else(|| self.text.len(), |g| g.size().width);
        self.children
            .iter()
            .fold(len, |len, child| len + child.len())
    }

    /// Check if text is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Replace face of the text with the provided one
    pub fn with_face(self, face: Face) -> Self {
        Self { face, ..self }
    }

    /// Assign glyph to the text. It will replace text content when rendered.
    /// TODO: if glyphs are not supported, draw text instead
    pub fn with_glyph(self, glyph: Glyph) -> Self {
        Self {
            glyph: Some(glyph),
            ..self
        }
    }

    /// Extend with text consuming self
    pub fn add_text(mut self, text: impl Into<Text<'a>>) -> Self {
        self.push(text);
        self
    }

    pub fn push(&mut self, text: impl Into<Text<'a>>) -> &mut Self {
        self.children.push(text.into());
        self
    }
}

impl<'a> View for Text<'a> {
    fn render<'b>(
        &self,
        surf: &'b mut TerminalSurface<'b>,
        layout: &Tree<Layout>,
    ) -> Result<(), Error> {
        fn render_rec<'a>(
            writer: &mut TerminalWriter<'_>,
            face: &Face,
            this: &Text<'a>,
        ) -> Result<(), Error> {
            let new_face = face.overlay(&this.face);
            writer.face_set(new_face);
            match &this.glyph {
                Some(glyph) => {
                    writer.put(Cell::new_glyph(new_face, glyph.clone()));
                }
                None => write!(writer, "{}", this.text.as_ref())?,
            };
            for child in this.children.iter() {
                render_rec(writer, &this.face, child)?;
            }
            writer.face_set(*face);
            Ok(())
        }

        let mut surf = layout.view(surf);
        render_rec(&mut surf.writer(), &Face::default(), self)
    }

    fn layout(&self, ct: BoxConstraint) -> Tree<Layout> {
        fn size_rec<'a>(
            ct: BoxConstraint,
            text: &'a Text<'a>,
            size: &mut Size,
            pos: &mut Position,
        ) {
            match &text.glyph {
                Some(glyph) => {
                    let width = glyph.size().width;
                    if pos.col + width < ct.max().width {
                        pos.col += width;
                    } else {
                        size.width = max(size.width, pos.col);
                        size.height += 1;
                        pos.col = width;
                    }
                }
                None => {
                    for chr in text.text.chars() {
                        match chr {
                            '\r' => {}
                            '\n' => {
                                size.width = max(size.width, pos.col);
                                size.height += 1;
                                pos.col = 0;
                            }
                            _ => {
                                if pos.col < ct.max().width {
                                    pos.col += 1;
                                } else {
                                    size.width = max(size.width, pos.col);
                                    size.height += 1;
                                    pos.col = 1;
                                }
                            }
                        }
                    }
                }
            }
            for child in &text.children {
                size_rec(ct, child, size, pos)
            }
        }
        let mut size = Size::empty();
        let mut pos = Position::origin();
        size_rec(ct, self, &mut size, &mut pos);
        size.width = max(size.width, pos.col);
        if size.height != 0 || size.width != 0 {
            size.height += 1;
        }
        Tree::new(
            Layout {
                pos: Position::origin(),
                size: ct.clamp(size),
            },
            Vec::new(),
        )
    }
}

impl<'a, T: Into<Text<'a>>> Extend<T> for Text<'a> {
    fn extend<TS: IntoIterator<Item = T>>(&mut self, iter: TS) {
        for item in iter {
            self.push(item);
        }
    }
}

impl<'a> Write for Text<'a> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let out = if self.children.is_empty() {
            self.text.to_mut()
        } else {
            self.children.push(Text::text(String::new()));
            self.children.last_mut().unwrap().text.to_mut()
        };
        let mut decoder = Utf8Decoder::new();
        let mut cursor = Cursor::new(buf);
        while let Some(chr) = decoder.decode(&mut cursor)? {
            out.push(chr);
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a, 'b: 'a> From<&'b str> for Text<'a> {
    fn from(string: &'b str) -> Self {
        Text::text(string)
    }
}

impl<'a> From<String> for Text<'a> {
    fn from(string: String) -> Self {
        Text::text(string)
    }
}

impl<'a> From<Glyph> for Text<'a> {
    fn from(glyph: Glyph) -> Self {
        Text::glyph(glyph)
    }
}

impl<'a> View for &'a str {
    fn render<'b>(
        &self,
        surf: &'b mut TerminalSurface<'b>,
        layout: &Tree<Layout>,
    ) -> Result<(), Error> {
        Text::text(*self).render(surf, layout)
    }

    fn layout(&self, ct: BoxConstraint) -> Tree<Layout> {
        Text::text(*self).layout(ct)
    }
}

impl View for String {
    fn render<'a>(
        &self,
        surf: &'a mut TerminalSurface<'a>,
        layout: &Tree<Layout>,
    ) -> Result<(), Error> {
        self.as_str().render(surf, layout)
    }

    fn layout(&self, ct: BoxConstraint) -> Tree<Layout> {
        self.as_str().layout(ct)
    }
}

#[cfg(test)]
mod tests {
    use crate::{BBox, FillRule, Glyph, Path};

    use super::*;

    #[test]
    fn test_text() -> Result<(), Error> {
        let two = "two".to_string();
        let mut text = Text::text("one ")
            .with_face("fg=#3c3836,bg=#ebdbb2".parse()?)
            .add_text(Text::text(two.as_str()).with_face("fg=#af3a03,bold".parse()?))
            .add_text(" three".to_string())
            .add_text("\nfour")
            .add_text(" ");
        write!(text, "and more")?;
        assert_eq!(text.len(), 27);

        let size = Size::new(5, 10);
        print!("{:?}", text.debug(size));

        let layout = text.layout(BoxConstraint::loose(size));
        assert_eq!(layout.size, Size::new(4, 10));

        Ok(())
    }

    #[test]
    fn test_text_glyph() -> Result<(), Error> {
        let path: Path = "
        M13.5 2a5.5 5.5 0 0 0-4.905 3.008 6.995 6.995 0 0 1 5.49 3.125C14.545 8.046
        15.018 8 15.5 8h3.478c.014-.165.022-.331.022-.5V3.44A1.44 1.44 0 0 0 17.56 2
        H13.5ZM8.426 17.997A6 6 0 0 1 2.25 12V7.514C2.25 6.678 2.928 6 3.764 6H8.25
        c1.966 0 3.712.946 4.806 2.407a7.522 7.522 0 0 0-3.938 3.15L7.53 9.97
        a.75.75 0 0 0-1.06 1.06l1.96 1.96A7.488 7.488 0 0 0 8 15.5c0 .876.15 1.716.426 2.497Z
        M9 15.5A6.5 6.5 0 0 1 15.5 9h4.914C21.29 9 22 9.71 22 10.586V15.5
        a6.5 6.5 0 0 1-10.535 5.096L10.28 21.78a.75.75 0 1 1-1.06-1.06l1.184-1.185
        A6.473 6.473 0 0 1 9 15.5Zm3.177 4.383 4.603-4.603a.75.75 0 1 0-1.06-1.06
        l-4.603 4.603c.303.4.66.757 1.06 1.06Z
        "
        .parse()
        .unwrap();
        let glyph = Glyph::new(
            path,
            FillRule::NonZero,
            Some(BBox::new((1.0, 1.0), (23.0, 23.0))),
            Size::new(1, 2),
        );

        let text = Text::text("before ->")
            .with_face("fg=#3c3836,bg=#ebdbb2".parse()?)
            .add_text(Text::glyph(glyph).with_face("fg=#79740e".parse()?))
            .add_text("<- after ");

        let size = Size::new(5, 11);
        assert_eq!(
            Tree::new(
                Layout {
                    pos: Position::origin(),
                    size: Size::new(2, 11)
                },
                Vec::new()
            ),
            text.layout(BoxConstraint::loose(size))
        );
        print!("{:?}", text.debug(size));

        let size = Size::new(5, 10);
        assert_eq!(
            Tree::new(
                Layout {
                    pos: Position::origin(),
                    size: Size::new(3, 10)
                },
                Vec::new()
            ),
            text.layout(BoxConstraint::loose(size))
        );
        print!("{:?}", text.debug(size));

        // check tight constraint
        let size = Size::new(10, 12);
        assert_eq!(
            Tree::new(
                Layout {
                    pos: Position::origin(),
                    size,
                },
                Vec::new(),
            ),
            text.layout(BoxConstraint::tight(size))
        );

        Ok(())
    }
}
