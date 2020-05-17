use crate::{
    decoder::Decoder, error::Error, Face, Position, Shape, Surface, Terminal, TerminalCommand,
    View, ViewExt, ViewMut, ViewMutExt,
};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Cell {
    face: Face,
    glyph: Option<char>,
    damaged: bool,
}

impl Cell {
    pub fn new(face: Face, glyph: Option<char>) -> Self {
        Self {
            face,
            glyph,
            damaged: false,
        }
    }

    fn new_damnaged() -> Self {
        Self {
            face: Default::default(),
            glyph: None,
            damaged: true,
        }
    }
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            face: Default::default(),
            glyph: None,
            damaged: false,
        }
    }
}

pub struct TerminalRenderer {
    face: Face,
    cursor: Position,
    front: Surface<Cell>,
    back: Surface<Cell>,
}

impl TerminalRenderer {
    pub fn new<T: Terminal + ?Sized>(term: &mut T, clear: bool) -> Result<Self, Error> {
        let size = term.size()?;
        term.execute(TerminalCommand::Face(Default::default()))?;
        term.execute(TerminalCommand::CursorTo(Position::new(0, 0)))?;
        let mut back = Surface::new(size.height, size.width);
        if clear {
            back.fill(Cell::new_damnaged());
        }
        Ok(Self {
            face: Default::default(),
            cursor: Position::new(0, 0),
            front: Surface::new(size.height, size.width),
            back,
        })
    }

    /// Render the current frame
    pub fn frame<T: Terminal + ?Sized>(&mut self, term: &mut T) -> Result<(), Error> {
        for row in 0..self.back.height() {
            let mut col = 0;
            while col < self.back.width() {
                let (src, dst) = match (self.front.get(row, col), self.back.get(row, col)) {
                    (Some(src), Some(dst)) => (src, dst),
                    _ => break,
                };
                if src == dst {
                    col += 1;
                    continue;
                }
                // update face
                if src.face != self.face {
                    term.execute(TerminalCommand::Face(src.face))?;
                    self.face = src.face;
                }
                // update position
                if self.cursor.row != row || self.cursor.col != col {
                    self.cursor.row = row;
                    self.cursor.col = col;
                    term.execute(TerminalCommand::CursorTo(self.cursor))?;
                }
                // identify glyph
                let glyph = match src.glyph {
                    None => ' ',
                    Some(glyph) => glyph,
                };
                // find if it is possible to erase instead of using ' '
                if glyph == ' ' {
                    let repeats = self.find_repeats(row, col);
                    // only use erase command when it is more efficient
                    if repeats > 4 {
                        term.execute(TerminalCommand::EraseChars(repeats))?;
                        col += repeats;
                        continue;
                    }
                }
                term.execute(TerminalCommand::Char(glyph))?;
                self.cursor.col += 1;
                col += 1;
            }
        }
        // swap buffers
        std::mem::swap(&mut self.front, &mut self.back);
        self.front.clear();
        Ok(())
    }

    /// View associated with the current frame
    pub fn view(&mut self) -> TerminalView<'_> {
        TerminalView {
            surf: self.front.view_mut(.., ..),
        }
    }

    fn find_repeats(&self, row: usize, col: usize) -> usize {
        let cell = self.front.get(row, col);
        if cell.is_none() {
            return 0;
        }
        let mut repeats = 1;
        while cell == self.front.get(row, col + repeats) {
            repeats += 1;
        }
        repeats
    }
}

pub struct TerminalView<'a> {
    surf: crate::surface::SurfaceViewMut<'a, Cell>,
}

impl<'a> View for TerminalView<'a> {
    type Item = Cell;

    fn shape(&self) -> Shape {
        self.surf.shape()
    }

    fn data(&self) -> &[Self::Item] {
        self.surf.data()
    }
}

impl<'a> ViewMut for TerminalView<'a> {
    fn data_mut(&mut self) -> &mut [Cell] {
        self.surf.data_mut()
    }
}

impl<'a> TerminalView<'a> {
    pub fn draw_box(&mut self, face: Option<Face>) {
        let shape = self.shape();
        if shape.width < 2 || shape.height < 2 {
            return;
        }
        let face = face.unwrap_or_default();

        let h = Cell::new(face, Some('─'));
        let v = Cell::new(face, Some('│'));
        self.view_mut(..1, 1..-1).fill(h.clone());
        self.view_mut(-1.., 1..-1).fill(h.clone());
        self.view_mut(1..-1, ..1).fill(v.clone());
        self.view_mut(1..-1, -1..).fill(v.clone());

        self.view_mut(..1, ..1).fill(Cell::new(face, Some('┌')));
        self.view_mut(..1, -1..).fill(Cell::new(face, Some('┐')));
        self.view_mut(-1.., -1..).fill(Cell::new(face, Some('┘')));
        self.view_mut(-1.., ..1).fill(Cell::new(face, Some('└')));
    }

    pub fn writer(&mut self, row: usize, col: usize, face: Option<Face>) -> TerminalViewWriter<'_> {
        let offset = self.shape().width * row + col;
        let mut iter = self.iter_mut();
        if offset > 0 {
            iter.nth(offset - 1);
        }
        TerminalViewWriter {
            face: face.unwrap_or_default(),
            iter,
            decoder: crate::decoder::Utf8Decoder::new(),
        }
    }
}

pub struct TerminalViewWriter<'a> {
    face: Face,
    iter: crate::surface::ViewMutIter<'a, Cell>,
    decoder: crate::decoder::Utf8Decoder,
}

impl<'a> std::io::Write for TerminalViewWriter<'a> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut cur = std::io::Cursor::new(buf);
        while let Some(glyph) = self.decoder.decode(&mut cur)? {
            let glyph = if glyph == ' ' { None } else { Some(glyph) };
            match self.iter.next() {
                Some(cell) => *cell = Cell::new(self.face, glyph),
                None => return Ok(buf.len()),
            }
        }
        Ok(cur.position() as usize)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        encoder::{Encoder, TTYEncoder},
        terminal::{TerminalEvent, TerminalSize},
    };
    use std::io::Write;

    fn debug<V: View<Item = Cell>>(view: V) -> Result<String, Error> {
        let mut encoder = TTYEncoder::new();
        let mut out = Vec::new();
        write!(&mut out, "\n┌")?;
        for _ in 0..view.width() {
            write!(&mut out, "─")?;
        }
        writeln!(&mut out, "┐")?;
        for row in 0..view.height() {
            write!(&mut out, "│")?;
            for col in 0..view.width() {
                match view.get(row, col) {
                    None => break,
                    Some(cell) => {
                        encoder.encode(&mut out, TerminalCommand::Face(cell.face))?;
                        write!(&mut out, "{}", cell.glyph.unwrap_or(' '))?;
                    }
                }
            }
            writeln!(&mut out, "\x1b[m│")?;
        }
        write!(&mut out, "└")?;
        for _ in 0..view.width() {
            write!(&mut out, "─")?;
        }
        write!(&mut out, "┘")?;
        Ok(String::from_utf8_lossy(&out).to_string())
    }

    struct DummyTerminal {
        size: TerminalSize,
        cmds: Vec<TerminalCommand>,
        buffer: Vec<u8>,
    }

    impl DummyTerminal {
        fn new(height: usize, width: usize) -> Self {
            Self {
                size: TerminalSize {
                    height,
                    width,
                    height_pixels: 0,
                    width_pixels: 0,
                },
                cmds: Default::default(),
                buffer: Default::default(),
            }
        }

        fn clear(&mut self) {
            self.cmds.clear();
            self.buffer.clear();
        }
    }

    impl Write for DummyTerminal {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.buffer.write(buf)
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl Terminal for DummyTerminal {
        fn execute(&mut self, cmd: TerminalCommand) -> Result<(), Error> {
            self.cmds.push(cmd);
            Ok(())
        }

        fn poll(
            &mut self,
            _timeout: Option<std::time::Duration>,
        ) -> Result<Option<TerminalEvent>, Error> {
            Ok(None)
        }

        fn size(&self) -> Result<TerminalSize, Error> {
            Ok(self.size)
        }
    }

    #[test]
    fn test_render() -> Result<(), Error> {
        use TerminalCommand::*;

        let purple = "bg=#d3869b".parse()?;
        // let red = "bg=#fb4934".parse()?;

        let mut term = DummyTerminal::new(3, 7);
        let mut render = TerminalRenderer::new(&mut term, false)?;

        let mut view = render.view();
        let mut writer = view.writer(0, 4, Some(purple));
        write!(writer, "TEST")?;

        println!("{}", debug(view)?);
        render.frame(&mut term)?;

        assert_eq!(
            term.cmds,
            vec![
                Face(Default::default()),
                CursorTo(Position::new(0, 0)),
                Face(purple),
                CursorTo(Position::new(0, 4)),
                Char('T'),
                Char('E'),
                Char('S'),
                CursorTo(Position::new(1, 0)),
                Char('T'),
            ]
        );
        term.clear();

        Ok(())
    }
}
