use crate::{Cell, Face, Position, Surface, TerminalCommand, ViewExt};

struct TerminalRenderer {
    face: Face,
    cursor: Position,
    front: Surface<Cell>,
    back: Surface<Cell>,
    height: usize,
    width: usize,
}

impl TerminalRenderer {
    fn new(height: usize, width: usize) -> Self {
        Self {
            face: Default::default(),
            cursor: Position::new(0, 0),
            front: Surface::new(height, width),
            back: Surface::new(height, width),
            height,
            width,
        }
    }

    pub fn render(&mut self) -> Vec<TerminalCommand> {
        let mut cmds = Vec::new();
        for row in 0..self.height {
            for col in 0..self.width {
                let (src, dst) = match (self.front.get(row, col), self.back.get(row, col)) {
                    (Some(src), Some(dst)) => (src, dst),
                    _ => break,
                };
                if src == dst {
                    continue;
                }
                // update face
                if src.face != self.face {
                    cmds.push(TerminalCommand::Face(src.face));
                    self.face = src.face;
                }
                // update position
                if self.cursor.row != row || self.cursor.col != col {
                    self.cursor.row = row;
                    self.cursor.col = col;
                    cmds.push(TerminalCommand::CursorTo(self.cursor));
                }
                // TOOD: use `TerminalErase` command to clean consequent spaces
                // render glyph
                let glyph = match src.glyph {
                    None => ' ',
                    Some(glyph) => glyph,
                };
                cmds.push(TerminalCommand::Char(glyph));
                self.cursor.col += 1;
            }
        }
        cmds
    }
}

use crate::{error::Error, Terminal, TerminalEvent, ViewMut, ViewMutExt};

pub enum RenderAction {
    Quit,
    Continue,
    Sleep(std::time::Duration),
}

pub fn run<T, R, E>(term: &mut T, mut render: R) -> Result<(), E>
where
    T: Terminal,
    R: FnMut(Option<TerminalEvent>, &mut dyn ViewMut<Item = Cell>) -> Result<RenderAction, E>,
    E: From<Error>,
{
    let size = term.size()?;
    term.execute(TerminalCommand::Face(Default::default()))?;
    term.execute(TerminalCommand::CursorTo(Position::new(0, 0)))?;
    let mut renderer = TerminalRenderer::new(size.height, size.width);
    let mut event = None;
    loop {
        let timeout = match render(event, &mut renderer.front)? {
            RenderAction::Quit => return Ok(()),
            RenderAction::Sleep(timeout) => Some(timeout),
            RenderAction::Continue => None,
        };

        for cmd in renderer.render() {
            term.execute(cmd)?;
        }
        std::mem::swap(&mut renderer.front, &mut renderer.back);
        renderer.front.clear();

        event = term.poll(timeout)?;
        if let Some(TerminalEvent::Resize(size)) = event {
            renderer = TerminalRenderer::new(size.height, size.width);
            term.execute(TerminalCommand::Face(Default::default()))?;
            term.execute(TerminalCommand::CursorTo(Position::new(0, 0)))?;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TerminalCommand, TerminalView, View, ViewMutExt};
    use std::io::Write;

    #[allow(unused)]
    fn debug<V: View<Item = Cell>>(view: V) -> Result<(), std::boxed::Box<dyn std::error::Error>> {
        use crate::encoder::{Encoder, TTYEncoder};

        let mut encoder = TTYEncoder::new();
        let stdout_locked = std::io::stdout();
        let mut stdout = stdout_locked.lock();
        write!(&mut stdout, "\n┌")?;
        for _ in 0..view.width() {
            write!(&mut stdout, "─")?;
        }
        writeln!(&mut stdout, "┐")?;
        for row in 0..view.height() {
            write!(&mut stdout, "│")?;
            for col in 0..view.width() {
                match view.get(row, col) {
                    None => break,
                    Some(cell) => {
                        encoder.encode(&mut stdout, TerminalCommand::Face(cell.face))?;
                        write!(&mut stdout, "{}", cell.glyph.unwrap_or(' '))?;
                    }
                }
            }
            writeln!(&mut stdout, "\x1b[m│")?;
        }
        write!(&mut stdout, "└")?;
        for _ in 0..view.width() {
            write!(&mut stdout, "─")?;
        }
        writeln!(&mut stdout, "┘")?;
        stdout.flush()?;
        Ok(())
    }

    #[test]
    fn test_diff() -> Result<(), std::boxed::Box<dyn std::error::Error>> {
        let dark = Some("#3c3836".parse()?);
        let bg = Face::default().with_bg(dark);
        let _green = Some(
            Face::default()
                .with_fg(dark)
                .with_bg(Some("#b8bb26".parse()?)),
        );
        let purple = Some(
            Face::default()
                .with_fg(dark)
                .with_bg(Some("#d3869b".parse()?)),
        );

        let mut render = TerminalRenderer::new(3, 7);
        render.front.fill(Cell::new(bg, None));
        render.back.fill(Cell::new(bg, None));

        let mut view = render.front.view_mut(.., 1..);
        let mut writer = view.writer(Position::new(0, 4), purple);
        write!(&mut writer, "TEST")?;
        // debug(&render.front)?;

        for cmd in render.render() {
            println!("{:?}", cmd);
        }

        Ok(())
    }
}
