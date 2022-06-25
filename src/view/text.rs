use super::{BoxConstraint, Layout, Tree, View};
use crate::{
    decoder::{Decoder, Utf8Decoder},
    Error, Face, Position, Size, TerminalSurface, TerminalSurfaceExt, TerminalWriter,
};
use std::{
    borrow::Cow,
    cmp::max,
    io::{Cursor, Write},
};

#[derive(Debug)]
pub struct Text<'a> {
    text: Cow<'a, str>,
    face: Face,
    children: Vec<Text<'a>>,
}

impl<'a> Text<'a> {
    pub fn new(text: impl Into<Cow<'a, str>>) -> Self {
        Self {
            text: text.into(),
            face: Face::default(),
            children: Vec::new(),
        }
    }

    /// Length of the text
    pub fn len(&self) -> usize {
        self.children
            .iter()
            .fold(self.text.len(), |len, child| len + child.len())
    }

    /// Check if text is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn with_face(self, face: Face) -> Self {
        Self { face, ..self }
    }

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
            writer.face_set(face.overlay(&this.face));
            write!(writer, "{}", this.text.as_ref())?;
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
        fn chars<'a>(this: &'a Text<'a>) -> Box<dyn Iterator<Item = char> + 'a> {
            // TODO: implement FIFO based depth first iterator to reduce allocations?
            let iter = this
                .text
                .as_ref()
                .chars()
                .chain(this.children.iter().flat_map(chars));
            Box::new(iter)
        }
        Tree::new(layout_chars(ct, chars(self)), Vec::new())
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
            self.children.push(Text::new(String::new()));
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
        Text::new(string)
    }
}

impl<'a> From<String> for Text<'a> {
    fn from(string: String) -> Self {
        Text::new(string)
    }
}

/// Layout iterator of chars
pub fn layout_chars(ct: BoxConstraint, chrs: impl IntoIterator<Item = char>) -> Layout {
    let mut size = Size::empty();
    let mut col = 0;
    for chr in chrs {
        match chr {
            '\r' => {}
            '\n' => {
                size.width = max(size.width, col);
                size.height += 1;
                col = 0;
            }
            _ => {
                if col < ct.max().width {
                    col += 1;
                } else {
                    size.width = max(size.width, col);
                    size.height += 1;
                    col = 1;
                }
            }
        }
    }
    size.width = max(size.width, col);
    if size.height != 0 || size.width != 0 {
        size.height += 1;
    }
    Layout {
        pos: Position::origin(),
        size,
    }
}

impl<'a> View for &'a str {
    fn render<'b>(
        &self,
        surf: &'b mut TerminalSurface<'b>,
        layout: &Tree<Layout>,
    ) -> Result<(), Error> {
        Text::new(*self).render(surf, layout)
    }

    fn layout(&self, ct: BoxConstraint) -> Tree<Layout> {
        Text::new(*self).layout(ct)
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
    use super::*;

    #[test]
    fn test_text() -> Result<(), Error> {
        let two = "two".to_string();
        let mut text = Text::new("one ")
            .with_face("fg=#3c3836,bg=#ebdbb2".parse()?)
            .add_text(Text::new(two.as_str()).with_face("fg=#af3a03,bold".parse()?))
            .add_text(" three".to_string())
            .add_text("\nfour")
            .add_text(" ");
        write!(text, "{}", "and more")?;

        let size = Size::new(5, 10);
        println!("{:?}", text.debug(size));

        let layout = text.layout(BoxConstraint::loose(size));
        assert_eq!(layout.size, Size::new(4, 10));

        Ok(())
    }
}
