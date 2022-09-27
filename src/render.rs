//! Terminal rendering logic
use crate::{
    decoder::Decoder,
    encoder::{Encoder, TTYEncoder},
    error::Error,
    view::{BoxConstraint, IntoView, View, ViewContext},
    Face, FaceAttrs, Glyph, Image, ImageHandler, KittyImageHandler, Position, Size, Surface,
    SurfaceMut, SurfaceMutIter, SurfaceMutView, SurfaceOwned, SurfaceView, Terminal, TerminalCaps,
    TerminalCommand, TerminalEvent, TerminalSize, TerminalWaker, RGBA,
};
use std::{
    cmp::max,
    collections::HashMap,
    fmt::Debug,
    io::{BufWriter, Write},
    num::NonZeroUsize,
};

/// Terminal cell kind
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum CellKind {
    /// Contains character
    Char(char),
    /// Contains image
    Image(Image),
    /// Contains glyph
    Glyph(Glyph),
    /// Must be re-rendered
    Damaged,
}

/// Terminal cell
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Cell {
    /// Cell's face
    face: Face,
    /// Cell's kind
    kind: CellKind,
}

impl Cell {
    /// Create new cell from face and char
    pub fn new_char(face: Face, character: Option<char>) -> Self {
        Self {
            face,
            kind: CellKind::Char(character.unwrap_or(' ')),
        }
    }

    /// Create new cell from image
    pub fn new_image(image: Image) -> Self {
        Self {
            face: Default::default(),
            kind: CellKind::Image(image),
        }
    }

    /// Create new cell from glyph
    pub fn new_glyph(face: Face, glyph: Glyph) -> Self {
        Self {
            face,
            kind: CellKind::Glyph(glyph),
        }
    }

    /// Create damaged cell
    fn new_damaged() -> Self {
        Self {
            face: Default::default(),
            kind: CellKind::Damaged,
        }
    }

    /// Width occupied by cell (can be != 1 for Glyph)
    pub fn width(&self) -> NonZeroUsize {
        let width = match &self.kind {
            CellKind::Glyph(glyph) => max(1, glyph.size().width),
            _ => 1,
        };
        NonZeroUsize::new(width).expect("zero cell width")
    }

    /// Cell face
    pub fn face(&self) -> Face {
        self.face
    }

    pub fn with_face(self, face: Face) -> Self {
        Cell { face, ..self }
    }
}

impl Default for Cell {
    fn default() -> Self {
        Self::new_char(Face::default(), None)
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
enum CellMark {
    /// Not marked
    #[default]
    Empty,
    /// Needs to be ignored
    Ignored,
    /// Needs to be returned during diffing
    Damaged,
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
    /// Marked cell that are treaded specially during diffing
    marks: SurfaceOwned<CellMark>,
    /// Current terminal size
    size: TerminalSize,
    /// Cache of rendered glyphs
    glyph_cache: HashMap<Cell, Image>,
    /// Frame counter
    frame_count: usize,
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
            marks: SurfaceOwned::new_with(size.cells.height, size.cells.width, |_, _| {
                CellMark::Empty
            }),
            size,
            glyph_cache: HashMap::new(),
            frame_count: 0,
        })
    }

    /// Clear terminal
    pub fn clear<T: Terminal + ?Sized>(&mut self, term: &mut T) -> Result<(), Error> {
        // erase all images
        for (pos, cell) in self.back.iter().with_position() {
            if let CellKind::Image(img) = &cell.kind {
                term.execute(TerminalCommand::ImageErase(img.clone(), Some(pos)))?;
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
        self.front.as_mut()
    }

    /// Iterator over changed cells between back and front buffers
    pub fn diff(&self) -> impl Iterator<Item = (Position, &'_ Cell, &'_ Cell)> {
        TerminalRendererDiff {
            renderer: self,
            pos: Position::new(0, 0),
        }
    }

    /// Render the current frame
    #[tracing::instrument(level="trace", skip_all, fields(frame_count = %self.frame_count))]
    pub fn frame<T: Terminal + ?Sized>(&mut self, term: &mut T) -> Result<(), Error> {
        self.frame_count += 1;
        self.marks.fill(CellMark::Empty);

        // replace all glyphs with rendered images
        self.glyphs_reasterize(term.size()?);

        // Erase changed images
        //
        // - Images can overlap and newly rendered image might be erased by erase
        //   command addressed to images of the previous frame. That is why we
        //   are erasing all changed images of the previous frame before rendering
        //   new images.
        // - Mark erased areas as damaged, as they will need to be redrawn
        for ((pos, old), new) in self.back.iter().with_position().zip(self.front.iter()) {
            if old == new {
                continue;
            }
            if let CellKind::Image(image) = &old.kind {
                term.execute(TerminalCommand::ImageErase(
                    image.clone(),
                    Some(Position::new(pos.row, pos.col)),
                ))?;
                let size = image.size_cells(self.size.pixels_per_cell());
                self.marks
                    .view_mut(
                        pos.row..pos.row + size.height,
                        pos.col..pos.col + size.width,
                    )
                    .fill_with(|row, col, _| {
                        if row != 0 || col != 0 {
                            CellMark::Damaged
                        } else {
                            CellMark::Empty
                        }
                    });
            }
        }

        // Ignores cell covered by new images
        //
        // We need to ignore cells covered by images but not containing image itself
        // otherwise they might show up in the diff and overwrite part of the image
        // (in case of sixel)
        for (pos, cell) in self.front.iter().with_position() {
            if let CellKind::Image(image) = &cell.kind {
                let size = image.size_cells(self.size.pixels_per_cell());
                self.marks
                    .view_mut(
                        pos.row..pos.row + size.height,
                        pos.col..pos.col + size.width,
                    )
                    .fill_with(|row, col, _| {
                        if row != 0 || col != 0 {
                            CellMark::Ignored
                        } else {
                            CellMark::Empty
                        }
                    });
            }
        }

        for row in 0..self.back.height() {
            let mut col = 0;
            while col < self.back.width() {
                let (front, back) = match (self.front.get(row, col), self.back.get(row, col)) {
                    (Some(front), Some(back)) => (front, back),
                    _ => break,
                };
                let mark = self.marks.get(row, col).copied().unwrap_or(CellMark::Empty);
                if mark != CellMark::Damaged && (mark == CellMark::Ignored || front == back) {
                    col += 1;
                    continue;
                }

                // update face
                if front.face != self.face {
                    term.execute(TerminalCommand::Face(front.face))?;
                    self.face = front.face;
                }

                match front.kind.clone() {
                    CellKind::Image(image) => {
                        // erase area under image, needed for images with transparency
                        let size = image.size_cells(self.size.pixels_per_cell());
                        self.term_set_cursor(term.dyn_ref(), Position::new(row, col))?;
                        for row in 0..size.height {
                            self.term_set_cursor(
                                term.dyn_ref(),
                                Position {
                                    row: self.cursor.row + row,
                                    col: self.cursor.col,
                                },
                            )?;
                            self.term_erase(term.dyn_ref(), size.width)?;
                        }

                        // issue render command
                        self.term_set_cursor(term.dyn_ref(), Position::new(row, col))?;
                        term.execute(TerminalCommand::Image(image, Position::new(row, col)))?;

                        // set position large enough so it would trigger position update
                        self.cursor = Position::new(1_000_000, 1_000_000);

                        col += 1;
                    }
                    CellKind::Char(chr) => {
                        self.term_set_cursor(term.dyn_ref(), Position::new(row, col))?;
                        if chr == ' ' {
                            let repeats = self.find_repeats(row, col);
                            col += repeats;
                            self.term_erase(term.dyn_ref(), repeats)?;
                        } else {
                            term.execute(TerminalCommand::Char(chr))?;
                            self.cursor.col += 1;
                            col += 1;
                        }
                    }
                    _ => col += 1,
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
            if let CellKind::Glyph(glyph) = &cell.kind {
                let image = match self.glyph_cache.get(cell) {
                    Some(image) => image.clone(),
                    None => {
                        let image = glyph.rasterize(cell.face, term_size);
                        self.glyph_cache.insert(cell.clone(), image.clone());
                        image
                    }
                };
                cell.kind = CellKind::Image(image);
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

    fn term_set_cursor(&mut self, term: &mut dyn Terminal, pos: Position) -> Result<(), Error> {
        if self.cursor.row != pos.row || self.cursor.col != pos.col {
            self.cursor = pos;
            term.execute(TerminalCommand::CursorTo(self.cursor))?;
        }
        Ok(())
    }

    fn term_erase(&mut self, term: &mut dyn Terminal, count: usize) -> Result<(), Error> {
        if count > 4 {
            // NOTE:
            //   - only use erase command when it is more efficient
            //   - erase is not moving cursor
            term.execute(TerminalCommand::EraseChars(count))?;
        } else {
            self.cursor.col += count;
            for _ in 0..count {
                term.execute(TerminalCommand::Char(' '))?;
            }
        }
        Ok(())
    }
}

pub struct TerminalRendererDiff<'a> {
    renderer: &'a TerminalRenderer,
    pos: Position,
}

impl<'a> Iterator for TerminalRendererDiff<'a> {
    type Item = (Position, &'a Cell, &'a Cell);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let pos = self.pos;
            if self.pos.col + 1 < self.renderer.back.width() {
                self.pos.col += 1;
            } else if self.pos.row + 1 < self.renderer.back.height() {
                self.pos.col = 0;
                self.pos.row += 1;
            } else {
                return None;
            }
            let new = self.renderer.front.get(pos.row, pos.col)?;
            let old = self.renderer.back.get(pos.row, pos.col)?;
            let mark = self.renderer.marks.get(pos.row, pos.col)?;
            if *mark != CellMark::Damaged && (*mark == CellMark::Ignored || new == old) {
                continue;
            }
            return Some((pos, new, old));
        }
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

    /// Draw view on the surface
    fn draw_view(&mut self, ctx: &ViewContext, view: impl IntoView) -> Result<(), Error>;

    /// Erase surface with face
    fn erase(&mut self, face: Face);

    /// Write object that can be used to add text to the surface
    fn writer(&mut self) -> TerminalWriter<'_>;

    /// Wrapper around terminal surface that implements [std::fmt::Debug]
    /// which renders a surface to the terminal.
    fn preview(&self) -> TerminalSurfacePreview<'_>;
}

impl<S> TerminalSurfaceExt for S
where
    S: SurfaceMut<Item = Cell>,
{
    // Draw square box on the surface
    fn draw_box(&mut self, face: Option<Face>) {
        if self.width() < 2 || self.height() < 2 {
            return;
        }
        let face = face.unwrap_or_default();

        let h = Cell::new_char(face, Some('─'));
        let v = Cell::new_char(face, Some('│'));
        self.view_mut(0, 1..-1).fill(h.clone());
        self.view_mut(-1, 1..-1).fill(h);
        self.view_mut(1..-1, 0).fill(v.clone());
        self.view_mut(1..-1, -1).fill(v);

        self.view_mut(0, 0).fill(Cell::new_char(face, Some('┌')));
        self.view_mut(0, -1).fill(Cell::new_char(face, Some('┐')));
        self.view_mut(-1, -1).fill(Cell::new_char(face, Some('┘')));
        self.view_mut(-1, 0).fill(Cell::new_char(face, Some('└')));
    }

    // Draw image using unicode upper half block symbol \u{2580}
    fn draw_image_ascii(&mut self, img: impl Surface<Item = RGBA>) {
        let height = img.height() / 2 + img.height() % 2;
        let width = img.width();
        self.view_mut(..height, ..width).fill_with(|row, col, _| {
            let fg = img.get(row * 2, col).copied();
            let bg = img.get(row * 2 + 1, col).copied();
            let face = Face::new(fg, bg, FaceAttrs::EMPTY);
            Cell::new_char(face, Some('\u{2580}'))
        });
    }

    /// Draw image on the surface
    fn draw_image(&mut self, img: Image) {
        if let Some(cell) = self.get_mut(0, 0) {
            *cell = Cell::new_image(img).with_face(cell.face);
        }
    }

    /// Draw view on the surface
    fn draw_view(&mut self, ctx: &ViewContext, view: impl IntoView) -> Result<(), Error> {
        let view = view.into_view();
        let layout = view.layout(ctx, BoxConstraint::loose(self.size()));
        view.render(ctx, &mut self.view_mut(.., ..), &layout)?;
        Ok(())
    }

    /// Replace all cells with empty character and provided face
    fn erase(&mut self, face: Face) {
        self.fill_with(|_row, _col, cell| Cell::new_char(cell.face.overlay(&face), None));
    }

    /// Crete writable wrapper around the terminal surface
    fn writer(&mut self) -> TerminalWriter<'_> {
        TerminalWriter::new(self)
    }

    /// Create preview object that implements [Debug], only useful in tests
    /// and for debugging
    fn preview(&self) -> TerminalSurfacePreview<'_> {
        TerminalSurfacePreview {
            surf: self.as_ref(),
        }
    }
}

/// Writable (implements `Write`) object for `TerminalSurface`
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
    pub fn face_set(&mut self, face: Face) -> &mut Self {
        self.face = face;
        self
    }

    /// Skip offset amount of cells (row major order)
    pub fn skip(mut self, offset: usize) -> Self {
        if offset > 0 {
            self.iter.nth(offset - 1);
        }
        self
    }

    /// Get current position inside allocated view
    pub fn position(&self) -> Position {
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

    pub fn put_newline(&mut self) {
        let index = self.iter.index();
        let shape = self.iter.shape();
        let pos = self.position();
        if pos.col != 0 {
            let offset = shape.index(pos.row, shape.width - 1) - index;
            self.iter.nth(offset);
        }
    }

    /// Put cell
    pub fn put(&mut self, mut cell: Cell) -> bool {
        // compose cell face with the current face
        let face = self.face.overlay(&cell.face);
        let blank = cell.width().get() - 1;
        // create newline if cell is too wide, but only once
        let width = self.iter.shape().width;
        if self.position().col + cell.width().get() > width {
            self.put_newline()
        }
        // put cell
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
            *cell = Cell::new_char(face, Some(' '));
        }
        result
    }

    /// Put char
    pub fn put_char(&mut self, c: char, face: Face) -> bool {
        match c {
            '\r' => true,
            '\n' => {
                self.put_newline();
                true
            }
            chr => match self.iter.next() {
                Some(cell) => {
                    let face = cell.face.overlay(&face);
                    *cell = Cell::new_char(face, Some(chr));
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

pub struct TerminalSurfacePreview<'a> {
    surf: SurfaceView<'a, Cell>,
}

impl<'a> TerminalSurfacePreview<'a> {
    fn as_bytes(&self) -> Result<Vec<u8>, Error> {
        // expect true color and kitty image support
        let capabilities = TerminalCaps {
            depth: crate::encoder::ColorDepth::TrueColor,
            glyphs: true,
            kitty_keyboard: false,
        };
        // space for a box around the provided surface
        let size = Size {
            width: self.surf.width() + 2,
            height: self.surf.height() + 2,
        };

        // debug terminal
        let pixels_per_cell = ViewContext::dummy().pixels_per_cell();
        let mut term = DebugTerminal {
            size: TerminalSize {
                cells: size,
                // TOOD: allow to specify cell size?
                pixels: Size {
                    height: pixels_per_cell.height * size.height,
                    width: pixels_per_cell.width * size.width,
                },
            },
            encoder: TTYEncoder::new(capabilities.clone()),
            image_handler: KittyImageHandler::new().quiet(),
            output: Vec::new(),
            capabilities,
            face: Default::default(),
        };

        // render single frame
        for _ in 0..size.height {
            writeln!(term)?;
        }
        term.execute(TerminalCommand::CursorMove {
            row: -(size.height as i32),
            col: 0,
        })?;
        term.execute(TerminalCommand::CursorSave)?;
        let mut renderer = TerminalRenderer::new(&mut term, true)?;
        let mut view = renderer.view();
        view.draw_box(None);
        write!(
            view.view_mut(0, 2..-1).writer(),
            "{}x{}",
            size.height - 2,
            size.width - 2,
        )?;
        view.view_mut(1..-1, 1..-1)
            .iter_mut()
            .zip(self.surf.iter())
            .for_each(|(dst, src)| *dst = src.clone());
        renderer.frame(&mut term)?;

        Ok(term.output)
    }

    /// Write rendered surface to a file
    pub fn dump(&self, path: impl AsRef<std::path::Path>) -> Result<(), Error> {
        let mut file = BufWriter::new(std::fs::File::create(path)?);
        file.write_all(self.as_bytes()?.as_slice())?;
        writeln!(file)?;
        Ok(())
    }
}

impl<'a> Debug for TerminalSurfacePreview<'a> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut cur = std::io::Cursor::new(self.as_bytes().map_err(|_| std::fmt::Error)?);
        let mut decoder = crate::decoder::Utf8Decoder::new();
        writeln!(fmt)?;
        while let Some(chr) = decoder.decode(&mut cur).map_err(|_| std::fmt::Error)? {
            write!(fmt, "{}", chr)?;
        }
        writeln!(fmt)?;
        Ok(())
    }
}

struct DebugTerminal {
    size: TerminalSize,
    encoder: TTYEncoder,
    image_handler: KittyImageHandler,
    capabilities: TerminalCaps,
    output: Vec<u8>,
    face: Face,
}

impl Terminal for DebugTerminal {
    fn execute(&mut self, cmd: TerminalCommand) -> Result<(), Error> {
        use TerminalCommand::*;
        match cmd {
            Image(img, pos) => self.image_handler.draw(&mut self.output, &img, pos)?,
            ImageErase(img, pos) => self.image_handler.erase(&mut self.output, &img, pos)?,
            Face(face) => {
                self.face = face;
                self.encoder.encode(&mut self.output, cmd)?;
            }
            CursorTo(pos) => {
                // convert absolute position move to relative moves
                self.encoder.encode(&mut self.output, CursorRestore)?;
                self.encoder.encode(&mut self.output, Face(self.face))?;
                self.encoder.encode(
                    &mut self.output,
                    CursorMove {
                        row: pos.row as i32,
                        col: pos.col as i32,
                    },
                )?
            }
            cmd => self.encoder.encode(&mut self.output, cmd)?,
        }
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

    fn position(&mut self) -> Result<Position, Error> {
        Ok(Position::new(0, 0))
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
        &self.capabilities
    }
}

impl Write for DebugTerminal {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.output.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        terminal::{Size, TerminalEvent, TerminalSize, TerminalWaker},
        TerminalCaps,
    };

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

        fn position(&mut self) -> Result<Position, Error> {
            Ok(Position::new(0, 0))
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
        print!("writer with offset: {:?}", render.view().preview());
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
            .fill(Cell::new_char(red, None));
        print!("erase: {:?}", render.view().preview());
        render.view().preview().dump("/tmp/frame.txt")?;
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
        print!("ascii image: {:?}", render.view().preview());
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
        print!("writer new line: {:?}", render.view().preview());
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
        print!("writer new line: {:?}", render.view().preview());
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
