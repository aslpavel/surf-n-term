//! Terminal rendering logic
use crate::{
    decoder::Decoder, error::Error, Face, FaceAttrs, Glyph, Image, Position, Size, Surface,
    SurfaceMut, SurfaceMutIter, SurfaceMutView, SurfaceOwned, Terminal, TerminalCommand,
    TerminalSize, RGBA,
};
use std::{cmp::max, collections::HashMap, num::NonZeroUsize};

/// Terminal cell kind
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CellKind {
    /// Contains useful content
    Content,
    /// Must be skipped during rendering
    Ignore,
    /// Must be re-rendered
    Damaged,
}

/// Terminal cell
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Cell {
    face: Face,
    character: Option<char>,
    image: Option<Image>,
    glyph: Option<Glyph>,
    kind: CellKind,
}

impl Cell {
    /// Create new cell from face and char
    pub fn new(face: Face, character: Option<char>) -> Self {
        Self {
            face,
            character,
            image: None,
            glyph: None,
            kind: CellKind::Content,
        }
    }

    /// Create new cell from image
    pub fn new_image(image: Image) -> Self {
        Self {
            face: Default::default(),
            character: None,
            image: Some(image),
            glyph: None,
            kind: CellKind::Content,
        }
    }

    /// Create new cell from glyph
    pub fn new_glyph(face: Face, glyph: Glyph) -> Self {
        Self {
            face,
            character: None,
            image: None,
            glyph: Some(glyph),
            kind: CellKind::Content,
        }
    }

    /// Width occupied by cell (can be != 1 for Glyph)
    pub fn width(&self) -> NonZeroUsize {
        let width = self
            .glyph
            .as_ref()
            .map_or_else(|| 1, |g| max(1, g.size().width));
        NonZeroUsize::new(width).expect("zero cell width")
    }

    /// Create damaged cell
    fn new_damaged() -> Self {
        Self {
            face: Default::default(),
            character: None,
            image: None,
            glyph: None,
            kind: CellKind::Damaged,
        }
    }
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            face: Default::default(),
            character: None,
            image: None,
            glyph: None,
            kind: CellKind::Content,
        }
    }
}

pub type TerminalSurface<'a> = SurfaceMutView<'a, Cell>;

/// Terminal renderer
///
/// This object keeps two surfaces (front and back) and on each call to frame
/// generates necessary terminal commands to reconcile them.
pub struct TerminalRenderer {
    /// Current face
    face: Face,
    /// Current cursor position
    cursor: Position,
    /// Front surface (modified)
    front: SurfaceOwned<Cell>,
    /// Back surface (keeping previous state)
    back: SurfaceOwned<Cell>,
    /// Current terminal size
    size: TerminalSize,
    /// Cache of rendered glyphs
    glyph_cache: HashMap<Cell, Image>,
}

impl TerminalRenderer {
    /// Create new terminal renderer
    pub fn new<T: Terminal + ?Sized>(term: &mut T, clear: bool) -> Result<Self, Error> {
        let size = term.size()?;
        term.execute(TerminalCommand::Face(Default::default()))?;
        term.execute(TerminalCommand::CursorTo(Position::new(0, 0)))?;
        let mut back = SurfaceOwned::new(size.cells.height, size.cells.width);
        if clear {
            back.fill(Cell::new_damaged());
        }
        Ok(Self {
            face: Default::default(),
            cursor: Position::new(0, 0),
            front: SurfaceOwned::new(size.cells.height, size.cells.width),
            back,
            size,
            glyph_cache: HashMap::new(),
        })
    }

    /// Clear terminal
    pub fn clear<T: Terminal + ?Sized>(&mut self, term: &mut T) -> Result<(), Error> {
        // erase all images
        for cell in self.back.iter() {
            if let Some(img) = &cell.image {
                term.execute(TerminalCommand::ImageErase(img.clone(), None))?;
            }
        }

        self.face = Face::default().with_fg(Some(RGBA::new(254, 0, 253, 252)));
        self.cursor = Position::new(100_000, 100_000);
        self.front.fill(Cell::new_damaged());
        self.back.fill(Cell::new_damaged());

        Ok(())
    }

    /// View associated with the current frame
    pub fn view(&mut self) -> TerminalSurface<'_> {
        self.front.view_mut(.., ..)
    }

    /// Render the current frame
    pub fn frame<T: Terminal + ?Sized>(&mut self, term: &mut T) -> Result<(), Error> {
        // Rasterize all glyphs
        self.glyphs_reasterize(term.size()?);

        // Images can overlap and newly rendered image might be erased by erase command
        // addressed to images of the previous frame. That is why we are erasing all images
        // of the previous frame before rendering new images.
        for row in 0..self.back.height() {
            for col in 0..self.back.width() {
                let (front, back) = match (self.front.get(row, col), self.back.get(row, col)) {
                    (Some(front), Some(back)) => (front, back),
                    _ => break,
                };
                if front.image != back.image {
                    if let Some(img) = &back.image {
                        term.execute(TerminalCommand::ImageErase(
                            img.clone(),
                            Some(Position::new(row, col)),
                        ))?;
                    }
                }
                // mark all cells effected by the image as damaged
                if let Some(size) = front.image.as_ref().map(|img| img.size_cells(self.size)) {
                    let mut view = self
                        .front
                        .view_mut(row..row + size.height, col..col + size.width);
                    for cell in view.iter_mut() {
                        cell.kind = CellKind::Damaged;
                    }
                }
            }
        }

        for row in 0..self.back.height() {
            let mut col = 0;
            while col < self.back.width() {
                let (front, back) = match (self.front.get(row, col), self.back.get(row, col)) {
                    (Some(front), Some(back)) => (front, back),
                    _ => break,
                };
                if front.kind == CellKind::Ignore || front == back {
                    col += 1;
                    continue;
                }
                // update face
                if front.face != self.face {
                    term.execute(TerminalCommand::Face(front.face))?;
                    self.face = front.face;
                }
                // update position
                if self.cursor.row != row || self.cursor.col != col {
                    self.cursor.row = row;
                    self.cursor.col = col;
                    term.execute(TerminalCommand::CursorTo(self.cursor))?;
                }
                // handle image
                if let Some(image) = front.image.clone() {
                    let image_changed = front.image != back.image;
                    // make sure surface under image is not changed
                    let size = image.size_cells(self.size);
                    let mut view = self
                        .front
                        .view_mut(row..row + size.height, col..col + size.width);
                    for cell in view.iter_mut() {
                        cell.kind = CellKind::Ignore;
                    }
                    // render image if changed
                    if image_changed {
                        // issue render command
                        term.execute(TerminalCommand::Image(image, Position::new(row, col)))?;
                        // set position large enough so it would trigger position update
                        self.cursor = Position::new(100000, 1000000);
                    }
                    col += 1;
                    continue;
                }
                // identify character
                let chr = front.character.unwrap_or(' ');
                // find if it is possible to erase instead of using ' '
                if chr == ' ' {
                    let repeats = self.find_repeats(row, col);
                    col += repeats;
                    if repeats > 4 {
                        // NOTE:
                        //   - only use erase command when it is more efficient
                        //     this value chosen arbitrarily
                        //   - erase is not moving cursor
                        term.execute(TerminalCommand::EraseChars(repeats))?;
                    } else {
                        self.cursor.col += repeats;
                        for _ in 0..repeats {
                            term.execute(TerminalCommand::Char(chr))?;
                        }
                    }
                } else {
                    term.execute(TerminalCommand::Char(chr))?;
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

    /// Rasterize all glyphs in the front surface
    ///
    /// All glyphs are replaced with rasterized image
    fn glyphs_reasterize(&mut self, term_size: TerminalSize) {
        for cell in self.front.iter_mut() {
            if let Some(glyph) = &cell.glyph {
                let image = match self.glyph_cache.get(cell) {
                    Some(image) => image.clone(),
                    None => {
                        let image = glyph.rasterize(cell.face, term_size);
                        self.glyph_cache.insert(cell.clone(), image.clone());
                        image
                    }
                };
                cell.image = Some(image);
            }
        }
    }

    /// Find how many identical cells is located starting from provided coordinate
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

/// Terminal surface extension trait
pub trait TerminalSurfaceExt: SurfaceMut<Item = Cell> {
    /// Draw box
    fn draw_box(&mut self, face: Option<Face>);
    /// Draw image encoded as ascii blocks
    fn draw_image_ascii(&mut self, img: impl Surface<Item = RGBA>);
    /// Draw image
    fn draw_image(&mut self, img: Image);
    /// Erase surface with provided color
    fn erase(&mut self, color: Option<RGBA>);
    /// Write object that can be used to add text to the surface
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

    // draw image using unicode upper half block symbol \u{2580}
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

    /// Create new surface with updated face
    pub fn face(self, face: Face) -> Self {
        Self { face, ..self }
    }

    /// Set current face
    pub fn face_set(&mut self, face: Face) {
        self.face = face;
    }

    /// Skip offset amount of cells (row major order)
    pub fn skip(mut self, offset: usize) -> Self {
        if offset > 0 {
            self.iter.nth(offset - 1);
        }
        self
    }

    /// Get current position inside allocated view
    pub fn position(&self) -> (usize, usize) {
        self.iter.position()
    }

    /// Get size of the view backing this writer
    pub fn size(&self) -> Size {
        let shape = self.iter.shape();
        Size {
            height: shape.height,
            width: shape.width,
        }
    }

    /// Render terminal writable value
    pub fn display(&mut self, value: impl TerminalWritable) -> std::io::Result<()> {
        value.fmt(self)
    }

    /// Put cell
    pub fn put(&mut self, mut cell: Cell) -> bool {
        let blank = cell.width().get() - 1;
        // compose cell face with the current face
        let face = self.face.overlay(&cell.face);
        let result = match self.iter.next() {
            Some(cell_ref) => {
                cell.face = cell_ref.face.overlay(&face);
                *cell_ref = cell;
                true
            }
            None => false,
        };
        // fill the rest of the width with empty spaces
        for (_, cell) in (0..blank).zip(&mut self.iter) {
            *cell = Cell::new(face, Some(' '));
        }
        result
    }

    /// Put char
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
            chr => match self.iter.next() {
                Some(cell) => {
                    let face = cell.face.overlay(&face);
                    *cell = Cell::new(face, Some(chr));
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
        TerminalCaps,
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
                        write!(&mut out, "{}", cell.character.unwrap_or(' '))?;
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
        capabiliets: TerminalCaps,
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
                capabiliets: TerminalCaps::default(),
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

        fn dyn_ref(&mut self) -> &mut dyn Terminal {
            self
        }

        fn capabilities(&self) -> &TerminalCaps {
            &self.capabiliets
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
