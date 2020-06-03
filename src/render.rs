use crate::{
    decoder::Decoder, error::Error, Face, FaceAttrs, ImageHandle, Position, Surface, SurfaceMut,
    SurfaceMutIter, SurfaceMutView, SurfaceOwned, Terminal, TerminalCommand, RGBA,
};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Cell {
    face: Face,
    glyph: Option<char>,
    image: Option<ImageHandle>,
    damaged: bool,
}

impl Cell {
    pub fn new(face: Face, glyph: Option<char>) -> Self {
        Self {
            face,
            glyph,
            image: None,
            damaged: false,
        }
    }

    pub fn new_image(image: ImageHandle) -> Self {
        Self {
            face: Default::default(),
            glyph: None,
            image: Some(image),
            damaged: false,
        }
    }

    fn new_damnaged() -> Self {
        Self {
            face: Default::default(),
            glyph: None,
            image: None,
            damaged: true,
        }
    }
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            face: Default::default(),
            glyph: None,
            image: None,
            damaged: false,
        }
    }
}

pub type TerminalSurface<'a> = SurfaceMutView<'a, Cell>;

pub struct TerminalRenderer {
    face: Face,
    cursor: Position,
    front: SurfaceOwned<Cell>,
    back: SurfaceOwned<Cell>,
}

impl TerminalRenderer {
    pub fn new<T: Terminal + ?Sized>(term: &mut T, clear: bool) -> Result<Self, Error> {
        let size = term.size()?;
        term.execute(TerminalCommand::Face(Default::default()))?;
        term.execute(TerminalCommand::CursorTo(Position::new(0, 0)))?;
        let mut back = SurfaceOwned::new(size.height, size.width);
        if clear {
            back.fill(Cell::new_damnaged());
        }
        Ok(Self {
            face: Default::default(),
            cursor: Position::new(0, 0),
            front: SurfaceOwned::new(size.height, size.width),
            back,
        })
    }

    /// Render the current frame
    pub fn frame<T: Terminal + ?Sized>(&mut self, term: &mut T) -> Result<(), Error> {
        // we have to issue erase commands first since images can overlap and
        // newly rendered image might be erased by erase command.
        for row in 0..self.back.height() {
            for col in 0..self.back.width() {
                let (src, dst) = match (self.front.get(row, col), self.back.get(row, col)) {
                    (Some(src), Some(dst)) => (src, dst),
                    _ => break,
                };
                if src.image != dst.image {
                    term.execute(TerminalCommand::ImageErase(Position::new(row, col)))?;
                }
            }
        }

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
                // handle image
                if src.image != dst.image {
                    if let Some(image) = src.image.clone() {
                        term.execute(TerminalCommand::Image(image))?;
                    }
                }
                // identify glyph
                let glyph = match src.glyph {
                    None => ' ',
                    Some(glyph) => glyph,
                };
                // find if it is possible to erase instead of using ' '
                if glyph == ' ' {
                    let repeats = self.find_repeats(row, col);
                    col += repeats;
                    if repeats > 4 {
                        // NOTE:
                        //   - only use erase command when it is more efficient
                        //     this value choosen arbirtraraly
                        //   - erase is not moving cursor
                        term.execute(TerminalCommand::EraseChars(repeats))?;
                    } else {
                        self.cursor.col += repeats;
                        for _ in 0..repeats {
                            term.execute(TerminalCommand::Char(glyph))?;
                        }
                    }
                } else {
                    term.execute(TerminalCommand::Char(glyph))?;
                    self.cursor.col += 1;
                    col += 1;
                }
            }
        }
        // swap buffers
        std::mem::swap(&mut self.front, &mut self.back);
        self.front.clear();
        Ok(())
    }

    /// View associated with the current frame
    pub fn view(&mut self) -> TerminalSurface<'_> {
        self.front.view_mut(.., ..)
    }

    fn find_repeats(&self, row: usize, col: usize) -> usize {
        let first = self.front.get(row, col);
        if first.is_none() {
            return 0;
        }
        let mut repeats = 1;
        loop {
            let src = self.front.get(row, col + repeats);
            let dst = self.back.get(row, col + repeats);
            if first == src && src != dst {
                repeats += 1;
            } else {
                break;
            }
        }
        repeats
    }
}

pub trait TerminalSurfaceExt {
    fn draw_box(&mut self, face: Option<Face>);
    fn draw_image_ascii(&mut self, img: impl Surface<Item = RGBA>);
    fn draw_image(&mut self, img: ImageHandle);
    fn erase(&mut self, color: RGBA);
    fn writer(&mut self) -> TerminalWriter<'_>;
}

impl<S> TerminalSurfaceExt for S
where
    S: SurfaceMut<Item = Cell>,
{
    fn draw_box(&mut self, face: Option<Face>) {
        if self.width() < 2 || self.height() < 2 {
            return;
        }
        let face = face.unwrap_or_default();

        let h = Cell::new(face, Some('─'));
        let v = Cell::new(face, Some('│'));
        self.view_mut(0, 1..-1).fill(h.clone());
        self.view_mut(-1, 1..-1).fill(h.clone());
        self.view_mut(1..-1, 0).fill(v.clone());
        self.view_mut(1..-1, -1).fill(v.clone());

        self.view_mut(0, 0).fill(Cell::new(face, Some('┌')));
        self.view_mut(0, -1).fill(Cell::new(face, Some('┐')));
        self.view_mut(-1, -1).fill(Cell::new(face, Some('┘')));
        self.view_mut(-1, 0).fill(Cell::new(face, Some('└')));
    }

    // draw image using unicode uppper half block symbol \u{2580}
    fn draw_image_ascii(&mut self, img: impl Surface<Item = RGBA>) {
        let height = (img.height() / 2 + img.height() % 2) as i32;
        let width = img.width() as i32;
        self.view_mut(..height, ..width).fill_with(|row, col, _| {
            let fg = img.get(row * 2, col).copied();
            let bg = img.get(row * 2 + 1, col).copied();
            let face = Face::new(fg, bg, FaceAttrs::EMPTY);
            Cell::new(face, Some('\u{2580}'))
        });
    }

    fn draw_image(&mut self, img: ImageHandle) {
        if let Some(cell) = self.get_mut(0, 0) {
            std::mem::replace(cell, Cell::new_image(img));
        }
    }

    fn erase(&mut self, color: RGBA) {
        let face = Face::default().with_bg(Some(color));
        self.fill_with(|_, _, _| Cell::new(face, None));
    }

    fn writer(&mut self) -> TerminalWriter<'_> {
        TerminalWriter::new(self)
    }
}

pub struct TerminalWriter<'a> {
    face: Face,
    iter: SurfaceMutIter<'a, Cell>,
    decoder: crate::decoder::Utf8Decoder,
}

impl<'a> TerminalWriter<'a> {
    pub fn new<S>(surf: &'a mut S) -> Self
    where
        S: SurfaceMut<Item = Cell> + ?Sized,
    {
        Self {
            face: Default::default(),
            iter: surf.iter_mut(),
            decoder: crate::decoder::Utf8Decoder::new(),
        }
    }

    pub fn face(self, face: Face) -> Self {
        Self { face, ..self }
    }

    pub fn skip(mut self, offset: usize) -> Self {
        if offset > 0 {
            self.iter.nth(offset - 1);
        }
        self
    }
}

impl<'a> std::io::Write for TerminalWriter<'a> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut cur = std::io::Cursor::new(buf);
        while let Some(glyph) = self.decoder.decode(&mut cur)? {
            let glyph = if glyph == ' ' { None } else { Some(glyph) };
            match self.iter.next() {
                Some(cell) => {
                    let face = cell.face.overlay(&self.face);
                    *cell = Cell::new(face, glyph)
                }
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
        ImageHandle,
    };
    use std::io::Write;

    fn debug(view: impl Surface<Item = Cell>) -> Result<String, Error> {
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

        fn image_register(
            &mut self,
            _img: impl Surface<Item = RGBA>,
        ) -> Result<ImageHandle, Error> {
            unimplemented!()
        }
    }

    #[test]
    fn test_render() -> Result<(), Error> {
        use TerminalCommand::*;

        let purple = "bg=#d3869b".parse()?;
        let red = "bg=#fb4934".parse()?;

        let mut term = DummyTerminal::new(3, 7);
        let mut render = TerminalRenderer::new(&mut term, false)?;

        let mut view = render.view().view_owned(.., 1..);
        let mut writer = view.writer().skip(4).face(purple);
        write!(writer, "TEST")?;
        println!("writer with offset: {}", debug(render.view())?);
        render.frame(&mut term)?;
        assert_eq!(
            term.cmds,
            vec![
                Face(Default::default()),
                CursorTo(Position::new(0, 0)),
                Face(purple),
                CursorTo(Position::new(0, 5)),
                Char('T'),
                Char('E'),
                CursorTo(Position::new(1, 1)),
                Char('S'),
                Char('T'),
            ]
        );
        term.clear();

        render
            .view()
            .view_owned(1..2, 1..-1)
            .fill(Cell::new(red, None));
        println!("erase:{}", debug(render.view())?);
        render.frame(&mut term)?;
        assert_eq!(
            term.cmds,
            vec![
                Face(Default::default()),
                // erase is not used as we only need to remove two spaces
                CursorTo(Position { row: 0, col: 5 }),
                Char(' '),
                Char(' '),
                // erase is used
                Face(red),
                CursorTo(Position { row: 1, col: 1 }),
                EraseChars(5)
            ]
        );
        term.clear();

        let mut img = SurfaceOwned::new(3, 3);
        let purple = "#d3869b".parse()?;
        let green = "#b8bb26".parse()?;
        img.fill_with(|r, c, _| if (r + c) % 2 == 0 { purple } else { green });
        render.view().view_mut(1.., 2..).draw_image_ascii(&img);
        println!("ascii image: {}", debug(render.view())?);
        render.frame(&mut term)?;
        assert_eq!(
            term.cmds,
            vec![
                Face(Default::default()),
                Char(' '),
                Face("fg=#d3869b, bg=#b8bb26".parse()?),
                Char('▀'),
                Face("fg=#b8bb26, bg=#d3869b".parse()?),
                Char('▀'),
                Face("fg=#d3869b, bg=#b8bb26".parse()?),
                Char('▀'),
                Face(Default::default()),
                Char(' '),
                Face("fg=#d3869b".parse()?),
                CursorTo(Position { row: 2, col: 2 }),
                Char('▀'),
                Face("fg=#b8bb26".parse()?),
                Char('▀'),
                Face("fg=#d3869b".parse()?),
                Char('▀')
            ]
        );
        term.clear();

        Ok(())
    }
}
