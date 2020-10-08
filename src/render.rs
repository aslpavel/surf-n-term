use crate::{
    decoder::Decoder, error::Error, Face, FaceAttrs, Image, Position, Surface, SurfaceMut,
    SurfaceMutIter, SurfaceMutView, SurfaceOwned, Terminal, TerminalCommand, TerminalSize, RGBA,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cell {
    face: Face,
    glyph: Option<char>,
    image: Option<Image>,
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

    pub fn new_image(image: Image) -> Self {
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
    size: TerminalSize,
}

impl TerminalRenderer {
    pub fn new<T: Terminal + ?Sized>(term: &mut T, clear: bool) -> Result<Self, Error> {
        let size = term.size()?;
        term.execute(TerminalCommand::Face(Default::default()))?;
        term.execute(TerminalCommand::CursorTo(Position::new(0, 0)))?;
        let mut back = SurfaceOwned::new(size.cells.height, size.cells.width);
        if clear {
            back.fill(Cell::new_damnaged());
        }
        Ok(Self {
            face: Default::default(),
            cursor: Position::new(0, 0),
            front: SurfaceOwned::new(size.cells.height, size.cells.width),
            back,
            size,
        })
    }

    pub fn clear(&mut self) {
        self.face = Face::default().with_fg(Some(RGBA::new(254, 0, 253, 252)));
        self.cursor = Position::new(100_000, 100_000);
        self.front.fill(Cell::new_damnaged());
        self.back.fill(Cell::new_damnaged());
    }

    /// View associated with the current frame
    pub fn view(&mut self) -> TerminalSurface<'_> {
        self.front.view_mut(.., ..)
    }

    /// Render the current frame
    pub fn frame<T: Terminal + ?Sized>(&mut self, term: &mut T) -> Result<(), Error> {
        // Images can overlap and newly rendered image might be erased by erase command
        // addressed to images of the previous frame. That is why we are erasing all images
        // of the previous frame before rendering new images.
        for row in 0..self.back.height() {
            for col in 0..self.back.width() {
                let (src, dst) = match (self.front.get(row, col), self.back.get(row, col)) {
                    (Some(src), Some(dst)) => (src, dst),
                    _ => break,
                };
                if src.image != dst.image && dst.image.is_some() {
                    term.execute(TerminalCommand::ImageErase(Position::new(row, col)))?;
                }
                // mark all cells effected by the image as dameged
                if let Some(img) = src.image.clone() {
                    let cell_size = self.size.cell_size();
                    let heigth = img.height() / cell_size.height;
                    let width = img.width() / cell_size.width;
                    let mut view = self.front.view_mut(row..row + heigth, col..col + width);
                    for cell in view.iter_mut() {
                        cell.damaged = true;
                    }
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
                if let Some(image) = src.image.clone() {
                    if src.image != dst.image {
                        // issure render command
                        term.execute(TerminalCommand::Image(image))?;
                        // set position large enough so it would tirgger position update
                        self.cursor = Position::new(100000, 1000000);
                    }
                    col += 1;
                    continue;
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

pub trait TerminalSurfaceExt: SurfaceMut<Item = Cell> {
    fn draw_box(&mut self, face: Option<Face>);
    fn draw_image_ascii(&mut self, img: impl Surface<Item = RGBA>);
    fn draw_image(&mut self, img: Image);
    fn erase(&mut self, color: Option<RGBA>);
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
        self.view_mut(-1, 1..-1).fill(h);
        self.view_mut(1..-1, 0).fill(v.clone());
        self.view_mut(1..-1, -1).fill(v);

        self.view_mut(0, 0).fill(Cell::new(face, Some('┌')));
        self.view_mut(0, -1).fill(Cell::new(face, Some('┐')));
        self.view_mut(-1, -1).fill(Cell::new(face, Some('┘')));
        self.view_mut(-1, 0).fill(Cell::new(face, Some('└')));
    }

    // draw image using unicode uppper half block symbol \u{2580}
    fn draw_image_ascii(&mut self, img: impl Surface<Item = RGBA>) {
        let height = img.height() / 2 + img.height() % 2;
        let width = img.width();
        self.view_mut(..height, ..width).fill_with(|row, col, _| {
            let fg = img.get(row * 2, col).copied();
            let bg = img.get(row * 2 + 1, col).copied();
            let face = Face::new(fg, bg, FaceAttrs::EMPTY);
            Cell::new(face, Some('\u{2580}'))
        });
    }

    fn draw_image(&mut self, img: Image) {
        if let Some(cell) = self.get_mut(0, 0) {
            *cell = Cell::new_image(img);
        }
    }

    fn erase(&mut self, color: Option<RGBA>) {
        let face = Face::default().with_bg(color);
        self.fill_with(|_, _, _| Cell::new(face, None));
    }

    fn writer(&mut self) -> TerminalWriter<'_> {
        TerminalWriter::new(self)
    }
}

pub trait TerminalWritable {
    fn fmt(&self, writer: &mut TerminalWriter<'_>) -> std::io::Result<()>;

    /// Estimate height occupied given with of the available surface
    fn height_hint(&self, _width: usize) -> Option<usize> {
        None
    }
}

impl<'a, T> TerminalWritable for &'a T
where
    T: TerminalWritable + ?Sized,
{
    fn fmt(&self, writer: &mut TerminalWriter<'_>) -> std::io::Result<()> {
        (*self).fmt(writer)
    }

    fn height_hint(&self, width: usize) -> Option<usize> {
        (*self).height_hint(width)
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

    pub fn face_set(&mut self, face: Face) {
        self.face = face;
    }

    pub fn skip(mut self, offset: usize) -> Self {
        if offset > 0 {
            self.iter.nth(offset - 1);
        }
        self
    }

    pub fn position(&self) -> (usize, usize) {
        self.iter.position()
    }

    pub fn display(&mut self, value: impl TerminalWritable) -> std::io::Result<()> {
        value.fmt(self)
    }

    pub fn put(&mut self, cell: Cell) -> bool {
        match self.iter.next() {
            Some(cell_ref) => {
                *cell_ref = cell;
                true
            }
            None => false,
        }
    }

    pub fn put_char(&mut self, c: char, face: Face) -> bool {
        match c {
            '\r' => true,
            '\n' => {
                let index = self.iter.index();
                let shape = self.iter.shape();
                let (row, col) = self.position();
                if col != 0 {
                    let offset = shape.index(row, shape.width - 1) - index;
                    self.iter.nth(offset);
                }
                true
            }
            glyph => match self.iter.next() {
                Some(cell) => {
                    let face = cell.face.overlay(&face);
                    *cell = Cell::new(face, Some(glyph));
                    true
                }
                None => false,
            },
        }
    }
}

impl<'a> std::io::Write for TerminalWriter<'a> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut cur = std::io::Cursor::new(buf);
        while let Some(glyph) = self.decoder.decode(&mut cur)? {
            if !self.put_char(glyph, self.face) {
                return Ok(buf.len());
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
        terminal::{Size, TerminalEvent, TerminalSize, TerminalWaker},
    };
    use std::io::Write;

    fn debug(view: impl Surface<Item = Cell>) -> Result<String, Error> {
        let mut encoder = TTYEncoder::default();
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
                    cells: Size { height, width },
                    pixels: Size {
                        height: 0,
                        width: 0,
                    },
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

        fn waker(&self) -> TerminalWaker {
            TerminalWaker::new(|| Ok(()))
        }

        fn frames_pending(&self) -> usize {
            0
        }

        fn frames_drop(&mut self) {}
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
                CursorTo(Position::new(0, 5)),
                Char(' '),
                Char(' '),
                // erase is used
                Face(red),
                CursorTo(Position::new(1, 1)),
                EraseChars(5)
            ]
        );
        term.clear();

        let mut img = SurfaceOwned::new(3, 3);
        let purple_color = "#d3869b".parse()?;
        let green_color = "#b8bb26".parse()?;
        img.fill_with(|r, c, _| {
            if (r + c) % 2 == 0 {
                purple_color
            } else {
                green_color
            }
        });
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

        let mut render = TerminalRenderer::new(&mut term, false)?;
        let mut view = render.view().view_owned(.., 1..-1);
        let mut writer = view.writer().face(purple);
        write!(&mut writer, "one\ntwo")?;
        println!("writer new line:{}", debug(render.view())?);
        render.frame(&mut term)?;
        assert_eq!(
            term.cmds,
            vec![
                Face(Default::default()),
                CursorTo(Position::new(0, 0)),
                Face(purple),
                CursorTo(Position::new(0, 1)),
                Char('o'),
                Char('n'),
                Char('e'),
                CursorTo(Position::new(1, 1)),
                Char('t'),
                Char('w'),
                Char('o')
            ]
        );
        term.clear();

        let mut render = TerminalRenderer::new(&mut term, false)?;
        let mut view = render.view().view_owned(.., 1..-1);
        let mut writer = view.writer().face(purple);
        write!(&mut writer, "  one\nx")?;
        println!("writer new line:{}", debug(render.view())?);
        render.frame(&mut term)?;
        assert_eq!(
            term.cmds,
            vec![
                Face(Default::default()),
                CursorTo(Position::new(0, 0)),
                Face(purple),
                CursorTo(Position::new(0, 1)),
                Char(' '),
                Char(' '),
                Char('o'),
                Char('n'),
                Char('e'),
                CursorTo(Position::new(1, 1)),
                Char('x'),
            ]
        );
        term.clear();

        Ok(())
    }
}
