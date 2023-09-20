//! Terminal rendering logic
use crate::{
    decoder::Decoder,
    encoder::{Encoder, TTYEncoder},
    error::Error,
    view::{BoxConstraint, IntoView, View, ViewContext},
    Face, Glyph, Image, ImageHandler, KittyImageHandler, Position, Size, Surface, SurfaceMut,
    SurfaceMutView, SurfaceOwned, SurfaceView, Terminal, TerminalCaps, TerminalCommand,
    TerminalEvent, TerminalSize, TerminalWaker,
};
use rasterize::RGBA;
use std::{
    cmp::{max, min},
    collections::HashMap,
    fmt::Debug,
    io::{BufWriter, Write},
};
use unicode_width::UnicodeWidthChar;

/// Terminal cell kind
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CellKind {
    /// Contains character
    Char(char),
    /// Contains image
    Image(Image),
    /// Contains glyph
    Glyph(Glyph),
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
    pub fn new_char(face: Face, character: char) -> Self {
        Self {
            face,
            kind: CellKind::Char(character),
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

    /// Get size of the cell
    pub fn size(&self, ctx: &ViewContext) -> Size {
        match &self.kind {
            CellKind::Char(ch) => Size::new(1, ch.width().unwrap_or(0)),
            CellKind::Glyph(glyph) => {
                if ctx.has_glyphs() {
                    glyph.size()
                } else {
                    Size {
                        height: 1,
                        width: glyph
                            .fallback_str()
                            .chars()
                            .map(|c| c.width().unwrap_or(0))
                            .sum(),
                    }
                }
            }
            CellKind::Image(image) => image.size_cells(ctx.pixels_per_cell()),
        }
    }

    /// Cell face
    pub fn face(&self) -> Face {
        self.face
    }

    /// Replace cell face
    pub fn with_face(self, face: Face) -> Self {
        Cell { face, ..self }
    }

    /// Return cell kind
    pub fn kind(&self) -> &CellKind {
        &self.kind
    }

    pub fn overlay(&mut self, other: Cell) -> &mut Self {
        self.face = self.face.overlay(&other.face);
        self.kind = other.kind;
        self
    }

    /// Layout cell
    ///
    /// Arguments:
    ///   - `max_width`       - maximum available width
    ///   - `pixels_per_cell` - number of pixels in a cell
    ///   - `size`            - tracked total size
    ///   - `cursor`          - tracked current cursor position
    ///
    /// Returns optional position where cell needs to be placed.
    pub fn layout(
        &self,
        ctx: &ViewContext,
        max_width: usize,
        size: &mut Size,
        cursor: &mut Position,
    ) -> Option<Position> {
        // special characters
        if let CellKind::Char(character) = &self.kind {
            match character {
                '\n' => {
                    size.width = max(size.width, cursor.col);
                    size.height = max(size.height, cursor.row + 1);
                    cursor.col = 0;
                    cursor.row += 1;
                    return None;
                }
                '\t' => {
                    cursor.col += (8 - cursor.col % 8).min(max_width.saturating_sub(cursor.col));
                    size.width = max(size.width, cursor.col);
                    return None;
                }
                _ => {}
            }
        }

        // skip empty cells
        let cell_size = self.size(ctx);
        if cell_size.height == 0 || cell_size.width == 0 {
            return None;
        }

        if cursor.col + cell_size.width <= max_width {
            // enough space to put cell
            let pos = *cursor;
            cursor.col += cell_size.width;
            size.width = max(size.width, cursor.col);
            size.height = max(size.height, cursor.row + cell_size.height);
            Some(pos)
        } else {
            // put new line
            cursor.row += 1;
            cursor.col = 0;
            size.height = max(size.height, cursor.row);
            // put cell unconditionally
            let pos = *cursor;
            cursor.col = min(cell_size.width, max_width);
            size.width = max(size.width, cursor.col);
            size.height = max(size.height, cursor.row + cell_size.height);
            Some(pos)
        }
    }
}

impl Default for Cell {
    fn default() -> Self {
        Self::new_char(Face::default(), ' ')
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
/// This object keeps two surfaces front (new) and back (old) and on each call
/// to frame function it generates necessary terminal commands to reconcile them.
pub struct TerminalRenderer {
    /// Current terminal size (not changes for the lifetime of the object)
    size: TerminalSize,
    /// Front surface (new)
    front: SurfaceOwned<Cell>,
    /// Back surface (old)
    back: SurfaceOwned<Cell>,

    /// Marked cell that are treaded specially during diffing
    marks: SurfaceOwned<CellMark>,
    /// Images to be rendered (frame function local kept here to avoid allocation)
    images: Vec<(Position, Face, Image)>,

    /// Cache of rendered glyphs
    glyph_cache: HashMap<Cell, Image>,
    /// Frame counter
    frame_count: usize,
}

impl TerminalRenderer {
    /// Create new terminal renderer
    pub fn new<T: Terminal + ?Sized>(term: &mut T, clear: bool) -> Result<Self, Error> {
        let size = term.size()?;
        let mark = if clear {
            CellMark::Damaged
        } else {
            CellMark::Empty
        };
        Ok(Self {
            size,
            front: SurfaceOwned::new(size.cells),
            back: SurfaceOwned::new(size.cells),
            marks: SurfaceOwned::new_with(size.cells, |_| mark),
            images: Vec::new(),
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

        self.marks.fill(CellMark::Damaged);
        self.front.fill(Cell::default());
        self.back.fill(Cell::default());

        Ok(())
    }

    /// Terminal surface for the new frame
    pub fn surface(&mut self) -> TerminalSurface<'_> {
        self.front.as_mut()
    }

    /// Generate frame, that is issue terminal command to reconcile
    /// back (old) and front (new) buffers.
    #[tracing::instrument(name = "[TerminalRenderer.frame]", level="trace", skip_all, fields(frame_count = %self.frame_count))]
    pub fn frame<T: Terminal + ?Sized>(&mut self, term: &mut T) -> Result<(), Error> {
        // First pass
        //
        // - Replace glyphs with images in the front buffer
        // - Erase changed images
        // - Record images that we need to render
        for ((pos, old), new) in self.back.iter().with_position().zip(self.front.iter_mut()) {
            // replace glyphs with images
            if let CellKind::Glyph(glyph) = &new.kind {
                let image = match self.glyph_cache.get(new) {
                    Some(image) => image.clone(),
                    None => {
                        let image = glyph.rasterize(new.face, self.size);
                        self.glyph_cache.insert(new.clone(), image.clone());
                        image
                    }
                };
                new.kind = CellKind::Image(image);
            }

            // skip cells that have not changed, go over ignored items too as they
            // might remove old images.
            if old == new && self.marks.get(pos) != Some(&CellMark::Damaged) {
                continue;
            }

            // erase and damage area under old image
            if let CellKind::Image(image) = &old.kind {
                term.execute(TerminalCommand::ImageErase(image.clone(), Some(pos)))?;
                let size = image.size_cells(self.size.pixels_per_cell());
                self.marks
                    .view_mut(
                        pos.row..pos.row + size.height,
                        pos.col..pos.col + size.width,
                    )
                    .fill(CellMark::Damaged);
            }

            // record image to be rendered, and mark area under the image to be ignored
            if let CellKind::Image(image) = &new.kind {
                self.images.push((pos, new.face, image.clone()));
                let size = image.size_cells(self.size.pixels_per_cell());
                self.marks
                    .view_mut(
                        pos.row..pos.row + size.height,
                        pos.col..pos.col + size.width,
                    )
                    .fill(CellMark::Ignored);
            }
        }

        // Second pass
        //
        // Render or characters
        let mut face = Face::default().with_bg(Some(RGBA::new(1, 2, 3, 255)));
        let mut cursor = Position::new(123_456, 654_123);

        let mut pos = Position::origin();
        while pos.row < self.front.height() {
            while pos.col < self.front.width() {
                // fetch buffers
                let offset = self.front.shape().offset(pos);
                let new = &self.front.data()[offset];
                let old = &self.back.data()[offset];
                let mark = self.marks.data()[offset];

                // skip conditions
                if mark != CellMark::Damaged && (mark == CellMark::Ignored || old == new) {
                    pos.col += 1;
                    continue;
                }
                let CellKind::Char(character) = &new.kind else {
                    pos.col += 1;
                    continue;
                };
                let character_width = character.width().unwrap_or(0);
                if character_width == 0 {
                    pos.col += 1;
                    continue;
                }

                // update face and cursor
                if face != new.face {
                    face = new.face;
                    term.execute(TerminalCommand::Face(face))?;
                }
                if cursor != pos {
                    cursor = pos;
                    term.execute(TerminalCommand::CursorTo(cursor))?;
                }

                if matches!(character, ' ') {
                    // find repeated empty cells
                    let mut repeats = 1;
                    for col in pos.col + 1..self.front.width() {
                        let pos = Position::new(pos.row, col);
                        let Some(next) = self.front.get(pos) else {
                            break;
                        };
                        let next_mark = self.marks.get(pos).copied().unwrap_or_default();
                        if next == new && next_mark != CellMark::Ignored {
                            repeats += 1;
                        } else {
                            break;
                        }
                    }
                    pos.col += repeats;
                    // erase if it is more efficient
                    if repeats > 4 {
                        // NOTE: erase is not moving cursor
                        term.execute(TerminalCommand::EraseChars(repeats))?;
                    } else {
                        cursor.col += repeats;
                        for _ in 0..repeats {
                            term.execute(TerminalCommand::Char(' '))?;
                        }
                    }
                } else {
                    term.execute(TerminalCommand::Char(*character))?;
                    cursor.col += character_width;
                    pos.col += character_width;
                }
            }
            pos.col = 0;
            pos.row += 1;
        }

        // Render images
        for (pos, face, image) in self.images.drain(..) {
            // Erase area under image, which makes sure are under image
            // has the same face as image cells.
            term.execute(TerminalCommand::Face(face))?;
            let size = image.size_cells(self.size.pixels_per_cell());
            for row in pos.row..pos.row + size.height {
                term.execute(TerminalCommand::CursorTo(Position::new(row, pos.col)))?;
                term.execute(TerminalCommand::EraseChars(size.width))?;
            }

            // draw image
            term.execute(TerminalCommand::CursorTo(pos))?;
            term.execute(TerminalCommand::Image(image, pos))?;
        }

        // Flip and clear buffers
        self.frame_count += 1;
        std::mem::swap(&mut self.front, &mut self.back);
        self.front.clear();
        self.images.clear();
        self.marks.fill(CellMark::Empty);

        Ok(())
    }
}

/// Terminal surface extension trait
pub trait TerminalSurfaceExt: SurfaceMut<Item = Cell> {
    /// Draw box
    fn draw_box(&mut self, face: Option<Face>);

    /// Fill surface with check pattern
    fn draw_check_pattern(&mut self, face: Face);

    /// Draw view on the surface
    fn draw_view(&mut self, ctx: &ViewContext, view: impl IntoView) -> Result<(), Error>;

    /// Erase surface with face
    fn erase(&mut self, face: Face);

    /// Write object that can be used to add text to the surface
    fn writer(&mut self, ctx: &ViewContext) -> TerminalWriter<'_>;

    /// Wrapper around terminal surface that implements [std::fmt::Debug]
    /// which renders a surface to the terminal.
    fn debug(&self) -> TerminalSurfaceDebug<'_>;
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

        let h = Cell::new_char(face, '─');
        let v = Cell::new_char(face, '│');
        self.view_mut(0, 1..-1).fill(h.clone());
        self.view_mut(-1, 1..-1).fill(h);
        self.view_mut(1..-1, 0).fill(v.clone());
        self.view_mut(1..-1, -1).fill(v);

        self.view_mut(0, 0).fill(Cell::new_char(face, '┌'));
        self.view_mut(0, -1).fill(Cell::new_char(face, '┐'));
        self.view_mut(-1, -1).fill(Cell::new_char(face, '┘'));
        self.view_mut(-1, 0).fill(Cell::new_char(face, '└'));
    }

    /// Fill surface with check pattern
    fn draw_check_pattern(&mut self, face: Face) {
        self.fill_with(|pos, _| {
            Cell::new_char(
                face,
                if pos.col & 1 == 1 {
                    '\u{2580}' // upper half block
                } else {
                    '\u{2584}' // lower half block
                },
            )
        })
    }

    /// Draw view on the surface
    fn draw_view(&mut self, ctx: &ViewContext, view: impl IntoView) -> Result<(), Error> {
        let view = view.into_view();
        let layout = view.layout(ctx, BoxConstraint::loose(self.size()));
        view.render(ctx, &mut self.as_mut(), &layout)?;
        Ok(())
    }

    /// Replace all cells with empty character and provided face
    fn erase(&mut self, face: Face) {
        self.fill_with(|_, cell| Cell::new_char(cell.face.overlay(&face), ' '));
    }

    /// Crete writable wrapper around the terminal surface
    fn writer(&mut self, ctx: &ViewContext) -> TerminalWriter<'_> {
        TerminalWriter::new(ctx.clone(), self)
    }

    /// Create object that implements [Debug], only useful in tests
    /// and for debugging
    fn debug(&self) -> TerminalSurfaceDebug<'_> {
        TerminalSurfaceDebug {
            surf: self.as_ref(),
        }
    }
}

/// Writable (implements `Write`) object for `TerminalSurface`
pub struct TerminalWriter<'a> {
    ctx: ViewContext,
    face: Face,       // face underlay-ed over all cells
    cursor: Position, // cursor position (next insert will happen at this position)
    size: Size,       // actual used size
    surf: SurfaceMutView<'a, Cell>,
    decoder: crate::decoder::Utf8Decoder,
}

impl<'a> TerminalWriter<'a> {
    pub fn new<S>(ctx: ViewContext, surf: &'a mut S) -> Self
    where
        S: SurfaceMut<Item = Cell> + ?Sized,
    {
        Self {
            ctx,
            face: Default::default(),
            cursor: Position::origin(),
            size: Size::empty(),
            surf: surf.as_mut(),
            decoder: crate::decoder::Utf8Decoder::new(),
        }
    }

    /// Get currently set face
    pub fn face(&self) -> Face {
        self.face
    }

    /// Create new surface with updated face
    pub fn with_face(self, face: Face) -> Self {
        Self { face, ..self }
    }

    /// Set current face
    pub fn set_face(&mut self, face: Face) -> &mut Self {
        self.face = face;
        self
    }

    /// Get current cursor position
    pub fn cursor(&self) -> Position {
        self.cursor
    }

    /// Move cursor to specified position
    pub fn set_cursor(&mut self, pos: Position) -> &mut Self {
        self.cursor = Position {
            col: min(pos.col, self.size().width),
            row: min(pos.row, self.size().height),
        };
        self
    }

    /// Get size of the view backing this writer
    pub fn size(&self) -> Size {
        self.surf.size()
    }

    /// Put cell
    pub fn put(&mut self, cell: Cell) -> bool {
        if !self.ctx.has_glyphs() {
            // glyph string fallback
            if let CellKind::Glyph(glyph) = &cell.kind {
                return glyph
                    .fallback_str()
                    .chars()
                    .all(|c| self.put_char(c, cell.face));
            }
        }

        // layout cell
        let cursor_start = self.cursor;
        let pos = cell.layout(
            &self.ctx,
            self.size().width,
            &mut self.size,
            &mut self.cursor,
        );

        let face = self.face.overlay(&cell.face);
        if let Some(pos) = pos {
            // normal cell
            if let Some(cell_ref) = self.surf.get_mut(pos) {
                cell_ref.overlay(cell.with_face(face));
                true
            } else {
                false
            }
        } else {
            if cursor_start != self.cursor {
                // cursor advanced: '\t', '\n'
                let shape = self.surf.shape();
                let data = self.surf.data_mut();

                let start = shape.offset(cursor_start);
                let end = shape.offset(self.cursor);

                let blank = Cell::new_char(face, ' ');
                for row in cursor_start.row..min(self.cursor.row + 1, shape.height) {
                    for col in 0..shape.width {
                        let offset = shape.offset(Position::new(row, col));
                        if (start..end).contains(&offset) {
                            data[offset].overlay(blank.clone());
                        }
                    }
                }
            }
            true
        }
    }

    /// Put character
    pub fn put_char(&mut self, character: char, face: Face) -> bool {
        self.put(Cell::new_char(face, character))
    }

    /// Put glyph
    pub fn put_glyph(&mut self, glyph: Glyph, face: Face) -> bool {
        self.put(Cell::new_glyph(face, glyph))
    }

    /// Put image
    pub fn put_image(&mut self, image: Image) -> bool {
        self.put(Cell::new_image(image))
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

pub struct TerminalSurfaceDebug<'a> {
    surf: SurfaceView<'a, Cell>,
}

impl<'a> TerminalSurfaceDebug<'a> {
    /// Write rendered surface to the output
    fn save<W: Write + Send>(&self, output: W) -> Result<W, Error> {
        // init capabilities
        let capabilities = TerminalCaps {
            depth: crate::encoder::ColorDepth::TrueColor,
            glyphs: true,
            kitty_keyboard: false,
        };

        // init size
        let size = Size {
            width: self.surf.width() + 2,
            height: self.surf.height() + 2,
        };

        // init debug terminal
        let ctx = ViewContext::dummy();
        let pixels_per_cell = ctx.pixels_per_cell();
        let mut term = TerminalDebug {
            size: TerminalSize {
                cells: size,
                pixels: Size {
                    height: pixels_per_cell.height * size.height,
                    width: pixels_per_cell.width * size.width,
                },
            },
            encoder: TTYEncoder::new(capabilities.clone()),
            image_handler: KittyImageHandler::new().quiet(),
            output,
            capabilities,
            face: Default::default(),
        };

        // move to origin an save cursor position
        let scroll = size.height as i32;
        term.execute_many([
            TerminalCommand::Scroll(scroll),
            TerminalCommand::CursorMove {
                row: -scroll,
                col: 0,
            },
            TerminalCommand::CursorSave,
        ])?;

        // create renderer
        let mut renderer = TerminalRenderer::new(&mut term, true)?;
        let mut surf = renderer.surface();

        // draw frame
        surf.draw_box(Some("fg=#665c54".parse()?));
        write!(
            surf.view_mut(0, 2..-1).writer(&ctx),
            "{}x{}",
            size.height - 2,
            size.width - 2,
        )?;

        // render frame
        surf.view_mut(1..-1, 1..-1)
            .iter_mut()
            .zip(self.surf.iter())
            .for_each(|(dst, src)| *dst = src.clone());
        renderer.frame(&mut term)?;

        // move to the end
        term.execute_many([
            TerminalCommand::Face(Face::default()),
            TerminalCommand::CursorRestore,
            TerminalCommand::CursorMove {
                row: size.height as i32,
                col: 0,
            },
        ])?;

        Ok(term.output)
    }

    /// Write rendered surface to a file
    pub fn save_to_file(&self, path: impl AsRef<std::path::Path>) -> Result<(), Error> {
        let file = BufWriter::new(std::fs::File::create(path)?);
        self.save(file)?;
        Ok(())
    }
}

impl<'a> Debug for TerminalSurfaceDebug<'a> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut cur = std::io::Cursor::new(self.save(Vec::new()).map_err(|_| std::fmt::Error)?);
        let mut decoder = crate::decoder::Utf8Decoder::new();
        writeln!(fmt)?;
        while let Some(chr) = decoder.decode(&mut cur).map_err(|_| std::fmt::Error)? {
            write!(fmt, "{}", chr)?;
        }
        Ok(())
    }
}

struct TerminalDebug<W> {
    size: TerminalSize,
    encoder: TTYEncoder,
    image_handler: KittyImageHandler,
    capabilities: TerminalCaps,
    output: W,
    face: Face,
}

impl<W: Write + Send> Terminal for TerminalDebug<W> {
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

impl<W: Write> Write for TerminalDebug<W> {
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
        encoder::ColorDepth::TrueColor,
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
        fn new(size: Size) -> Self {
            Self {
                size: TerminalSize {
                    cells: size,
                    pixels: Size {
                        height: size.height * 20,
                        width: size.width * 10,
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
    fn test_render_commands() -> Result<(), Error> {
        use TerminalCommand::*;

        let purple = "bg=#d3869b".parse()?;
        let red = "bg=#fb4934".parse()?;

        let mut term = DummyTerminal::new(Size::new(3, 7));
        let mut render = TerminalRenderer::new(&mut term, false)?;
        let ctx = ViewContext::new(&term)?;

        let mut view = render.surface().view_owned(.., 1..);
        let mut writer = view.writer(&ctx).with_face(purple);
        writer.set_cursor(Position::new(0, 4));

        // write with offset
        assert_eq!(writer.cursor(), Position::new(0, 4));
        write!(writer, "TEST")?;
        assert_eq!(writer.cursor(), Position::new(1, 2));
        print!(
            "[render] writer with offset: {:?}",
            render.surface().debug()
        );
        render.frame(&mut term)?;
        assert_eq!(
            term.cmds,
            vec![
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

        // erase
        render
            .surface()
            .view_owned(1..2, 1..-1)
            .fill(Cell::new_char(red, ' '));
        print!("[render] erase: {:?}", render.surface().debug());
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

        // ascii image
        let mut image_surf = SurfaceOwned::new(Size::new(3, 3));
        let purple_color = "#d3869b".parse()?;
        let green_color = "#b8bb26".parse()?;
        image_surf.fill_with(|pos, _| {
            if (pos.row + pos.col) % 2 == 0 {
                purple_color
            } else {
                green_color
            }
        });
        let image_ascii = crate::Image::from(image_surf).ascii_view();
        render
            .surface()
            .view_mut(1.., 2..)
            .draw_view(&ctx, image_ascii)?;
        print!("[render] ascii image: {:?}", render.surface().debug());
        render.frame(&mut term)?;
        assert_eq!(
            term.cmds,
            vec![
                Face(Default::default()),
                CursorTo(Position { row: 1, col: 1 }),
                Char(' '),
                Face("fg=#d3869b, bg=#b8bb26".parse()?),
                Char('▀'),
                Face("fg=#b8bb26, bg=#d3869b".parse()?),
                Char('▀'),
                Face("fg=#d3869b, bg=#b8bb26".parse()?),
                Char('▀'),
                Face(Default::default()),
                Char(' '),
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

        // new line
        let mut render = TerminalRenderer::new(&mut term, false)?;
        let mut view = render.surface().view_owned(.., 1..-1);
        let mut writer = view.writer(&ctx).with_face(purple);
        assert_eq!(writer.cursor(), Position::new(0, 0));
        write!(&mut writer, "one\ntwo")?;
        assert_eq!(writer.cursor(), Position::new(1, 3));
        print!("[render] writer new line: {:?}", render.surface().debug());
        render.frame(&mut term)?;
        assert_eq!(
            term.cmds,
            vec![
                Face(purple),
                CursorTo(Position::new(0, 1)),
                Char('o'),
                Char('n'),
                Char('e'),
                Char(' '),
                Char(' '),
                CursorTo(Position::new(1, 1)),
                Char('t'),
                Char('w'),
                Char('o')
            ]
        );
        term.clear();

        // new line at the end of line
        let mut render = TerminalRenderer::new(&mut term, false)?;
        let mut view = render.surface().view_owned(.., 1..-1);
        let mut writer = view.writer(&ctx).with_face(purple);
        write!(&mut writer, "  one\ntwo")?;
        print!(
            "[render] writer new line at the end of line: {:?}",
            render.surface().debug()
        );
        render.frame(&mut term)?;
        assert_eq!(
            term.cmds,
            vec![
                Face(purple),
                CursorTo(Position::new(0, 1)),
                Char(' '),
                Char(' '),
                Char('o'),
                Char('n'),
                Char('e'),
                CursorTo(Position::new(1, 1)),
                Char('t'),
                Char('w'),
                Char('o'),
            ]
        );
        term.clear();

        // double with characters
        let gray = "bg=#504945".parse()?;
        let mut render = TerminalRenderer::new(&mut term, false)?;
        let mut view = render.surface().view_owned(.., 1..);
        let mut writer = view.writer(&ctx).with_face(gray);
        write!(&mut writer, "🤩 awesome 😻|")?;
        print!(
            "[render] double with characters: {:?}",
            render.surface().debug()
        );
        render.frame(&mut term)?;
        assert_eq!(
            term.cmds,
            vec![
                Face(gray),
                CursorTo(Position::new(0, 1)),
                Char('🤩'),
                Char(' '),
                Char('a'),
                Char('w'),
                Char('e'),
                CursorTo(Position::new(1, 1)),
                Char('s'),
                Char('o'),
                Char('m'),
                Char('e'),
                Char(' '),
                CursorTo(Position::new(2, 1)),
                Char('😻'),
                Char('|')
            ]
        );
        term.clear();

        Ok(())
    }

    #[test]
    fn test_render_image() -> Result<(), Box<dyn std::error::Error>> {
        use TerminalCommand::*;

        const ICON: &str = r#"
        {
            "view_box": [0, 0, 24, 24],
            "size": [2, 5],
            "path": "M10,17L5,12L6.41,10.58L10,14.17L17.59,6.58L19,8M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2Z"
        }
        "#;
        let face = "bg=#665c54,fg=#b8bb26".parse()?;
        let icon = Cell::new_glyph(face, serde_json::from_str(ICON)?);

        let mut term = DummyTerminal::new(Size::new(5, 10));
        let mut render = TerminalRenderer::new(&mut term, false)?;
        term.clear();

        let mut view = render.surface();
        view.set(Position::new(1, 2), icon.clone());
        print!("[render] image: {:?}", render.surface().debug());
        render.frame(&mut term)?;
        let img = render
            .glyph_cache
            .get(&icon)
            .expect("image is not rendered")
            .clone();
        assert_eq!(
            term.cmds,
            vec![
                Face(face),
                CursorTo(Position::new(1, 2)),
                EraseChars(5),
                CursorTo(Position::new(2, 2)),
                EraseChars(5),
                CursorTo(Position::new(1, 2)),
                Image(img.clone(), Position::new(1, 2))
            ]
        );
        term.clear();

        let mut view = render.surface();
        view.set(Position::new(2, 3), icon.clone());
        print!("[render] image move: {:?}", render.surface().debug());
        render.frame(&mut term)?;
        assert_eq!(
            term.cmds,
            vec![
                // erase old image
                ImageErase(img.clone(), Some(Position::new(1, 2))),
                Face(crate::Face::default()),
                CursorTo(Position::new(1, 2)),
                EraseChars(8),
                CursorTo(Position::new(2, 2)),
                Char(' '),
                // draw new image
                Face(face),
                CursorTo(Position::new(2, 3)),
                EraseChars(5),
                CursorTo(Position::new(3, 3)),
                EraseChars(5),
                CursorTo(Position::new(2, 3)),
                Image(img, Position::new(2, 3)),
            ]
        );
        term.clear();

        Ok(())
    }

    #[test]
    fn test_cell_layout() {
        let max_width = 10;
        let ctx = &ViewContext {
            pixels_per_cell: Size::new(10, 10),
            has_glyphs: true,
            color_depth: TrueColor,
        };
        let face = Face::default();

        let mut size = Size::default();
        let mut cursor = Position::default();

        // empty new line at the start
        let pos = Cell::new_char(face, '\n').layout(ctx, max_width, &mut size, &mut cursor);
        assert!(pos.is_none());
        assert_eq!(cursor, Position::new(1, 0));
        assert_eq!(size, Size::new(1, 0));

        // simple text line
        for c in "test".chars() {
            Cell::new_char(face, c).layout(ctx, max_width, &mut size, &mut cursor);
        }
        assert_eq!(cursor, Position::new(1, 4));
        assert_eq!(size, Size::new(2, 4));

        // new line
        let pos = Cell::new_char(face, '\n').layout(ctx, max_width, &mut size, &mut cursor);
        assert!(pos.is_none());
        assert_eq!(cursor, Position::new(2, 0));
        assert_eq!(size, Size::new(2, 4));

        // single width character
        let pos = Cell::new_char(face, ' ').layout(ctx, max_width, &mut size, &mut cursor);
        assert_eq!(pos, Some(Position::new(2, 0)));
        assert_eq!(cursor, Position::new(2, 1));
        assert_eq!(size, Size::new(3, 4));

        // double width character
        let pos = Cell::new_char(face, '🤩').layout(ctx, max_width, &mut size, &mut cursor);
        assert_eq!(pos, Some(Position::new(2, 1)));
        assert_eq!(cursor, Position::new(2, 3));
        assert_eq!(size, Size::new(3, 4));

        // tabulation
        let pos = Cell::new_char(face, '\t').layout(ctx, max_width, &mut size, &mut cursor);
        assert!(pos.is_none());
        assert_eq!(cursor, Position::new(2, 8));
        assert_eq!(size, Size::new(3, 8));

        // zero-width character
        let pos = Cell::new_char(face, '\0').layout(ctx, max_width, &mut size, &mut cursor);
        assert!(pos.is_none());
        assert_eq!(cursor, Position::new(2, 8));
        assert_eq!(size, Size::new(3, 8));

        // single width character close to the end of line
        let pos = Cell::new_char(face, 'P').layout(ctx, max_width, &mut size, &mut cursor);
        assert_eq!(pos, Some(Position::new(2, 8)));
        assert_eq!(cursor, Position::new(2, 9));
        assert_eq!(size, Size::new(3, 9));

        // double width character wraps
        let pos = Cell::new_char(face, '🥳').layout(ctx, max_width, &mut size, &mut cursor);
        assert_eq!(pos, Some(Position::new(3, 0)));
        assert_eq!(cursor, Position::new(3, 2));
        assert_eq!(size, Size::new(4, 9));

        // glyph
        let glyph = Glyph::new(
            rasterize::Path::empty(),
            Default::default(),
            None,
            Size::new(1, 3),
            " ".to_owned(),
        );
        let pos = Cell::new_glyph(face, glyph).layout(ctx, max_width, &mut size, &mut cursor);
        assert_eq!(pos, Some(Position::new(3, 2)));
        assert_eq!(cursor, Position::new(3, 5));
        assert_eq!(size, Size::new(4, 9));

        // image
        let image = Image::from(SurfaceOwned::new(Size::new(20, 30)));
        let image_cell = Cell::new_image(image);
        assert_eq!(image_cell.size(ctx), Size::new(2, 3));
        let pos = image_cell.layout(ctx, max_width, &mut size, &mut cursor);
        assert_eq!(pos, Some(Position::new(3, 5)));
        assert_eq!(cursor, Position::new(3, 8));
        assert_eq!(size, Size::new(5, 9));

        // image wrap
        let pos = image_cell.layout(ctx, max_width, &mut size, &mut cursor);
        assert_eq!(pos, Some(Position::new(4, 0)));
        assert_eq!(cursor, Position::new(4, 3));
        assert_eq!(size, Size::new(6, 9));
    }

    #[test]
    fn test_writer_tab() -> Result<(), Error> {
        let guide_face = "bg=#cc241d,fg=#fbf1c7".parse()?;
        let underline_face = "bg=#b8bb26,fg=#fbf1c7".parse()?;
        let tab_face = "bg=#458588,fg=#fbf1c7".parse()?;
        let mark_face = "bg=#b16286,fg=#fbf1c7".parse()?;

        let mut term = DummyTerminal::new(Size::new(30, 21));
        let mut render = TerminalRenderer::new(&mut term, false)?;
        let ctx = ViewContext::new(&term)?;

        {
            let mut view = render.surface().view_owned(1.., 0).transpose();
            let mut writer = view.writer(&ctx).with_face(guide_face);
            write!(writer, "012345678901234567890123456789")?;
        }

        let mut view = render.surface().view_owned(.., 1..);
        let mut writer = view.writer(&ctx).with_face(guide_face);
        write!(writer, "01234567890123456789\n")?;
        writer.set_face(Face::default());

        let mut cols = Vec::new();
        for index in 0..22 {
            (0..index).for_each(|_| {
                writer.put_char('_', underline_face);
            });
            writer.put_char('\t', tab_face);
            writer.put_char('X', mark_face);
            cols.push(writer.cursor().col);
            writer.put_char('\n', Face::default());
        }
        print!("[render] tab writer: {:?}", render.surface().debug());
        for (line, col) in cols.into_iter().enumerate() {
            assert_eq!((col - 1) % 8, 0, "column {} % 8 != 0 at line {}", col, line);
        }
        Ok(())
    }
}
