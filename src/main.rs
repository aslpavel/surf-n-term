mod cell;
pub use crate::cell::{Color, Face};

mod surface;
pub use crate::surface::{Surface, View};

use std::{fmt, io::Write};

pub trait Renderer {
    type Error;
    fn render(&mut self, surface: &Surface) -> Result<(), Self::Error>;
}

// -----------------------------------------------------------------------------
// TTY
// -----------------------------------------------------------------------------
#[derive(Debug)]
pub enum TTYError {
    IOError(std::io::Error),
}

impl fmt::Display for TTYError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for TTYError {}

impl From<std::io::Error> for TTYError {
    fn from(error: std::io::Error) -> Self {
        Self::IOError(error)
    }
}

pub struct TTYRenderer {
    face: Face,
    queue: Vec<u8>,
    tty: std::io::Stdout, // FIXME: use `/dev/tty` instead
}

impl TTYRenderer {
    pub fn new() -> Result<Self, TTYError> {
        Ok(Self {
            face: Default::default(),
            queue: Vec::new(),
            tty: std::io::stdout(),
        })
    }

    // FIXME: support for different color depth and attributes
    fn set_face(&mut self, face: Face) -> Result<(), TTYError> {
        if self.face == face {
            return Ok(());
        }
        write!(&mut self.queue, "\x1b[00")?;
        if self.face.bg != face.bg {
            face.bg
                .map(|c| write!(self.queue, ";48;2;{};{};{}", c.red, c.green, c.blue))
                .transpose()?;
        }
        if self.face.fg != face.fg {
            face.fg
                .map(|c| write!(self.queue, ";38;2;{};{};{}", c.red, c.green, c.blue))
                .transpose()?;
        }
        if self.face.attrs != face.attrs {
            unimplemented!()
        }
        write!(self.queue, "m")?;
        self.face = face;
        Ok(())
    }

    fn set_cursor(&mut self, row: usize, col: usize) -> Result<(), TTYError> {
        write!(self.queue, "\x1b[{};{}H", row + 1, col + 1)?;
        Ok(())
    }
}

impl Renderer for TTYRenderer {
    type Error = TTYError;

    fn render(&mut self, surface: &Surface) -> Result<(), Self::Error> {
        let shape = surface.shape();
        let data = surface.data();
        for row in 0..shape.height {
            self.set_cursor(row, 0)?;
            for col in 0..shape.width {
                let cell = &data[shape.index(row, col)];
                self.set_face(cell.face)?;
                match cell.glyph {
                    Some(glyph) => self.queue.write(&glyph)?,
                    None => self.queue.write(&[b' '])?,
                };
            }
        }
        self.set_face(Face::default())?;

        self.tty.write_all(&self.queue)?;
        self.tty.flush()?;
        self.queue.clear();

        Ok(())
    }
}

fn main() -> Result<(), std::boxed::Box<dyn std::error::Error>> {
    let bg = Face::default().with_bg(Some("#3c3836".parse()?));
    let one = Face::default().with_bg(Some("#d3869b".parse()?));
    let two = Face::default().with_bg(Some("#b8bb26".parse()?));
    let three = Face::default().with_bg(Some("#fb4934".parse()?));

    let mut surface = Surface::new(10, 20);
    surface.fill(|_, _, cell| cell.face = bg);
    surface.view(..2, ..2).fill(|_, _, cell| cell.face = one);
    surface.view(-2.., -2..).fill(|_, _, cell| cell.face = two);
    surface.view(.., 3..4).fill(|_, _, cell| cell.face = three);
    surface
        .view(3..4, ..-1)
        .fill(|_, _, cell| cell.face = three);

    let mut renderer = TTYRenderer::new()?;
    renderer.render(&surface)?;
    println!("");
    Ok(())
}
