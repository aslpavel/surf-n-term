use std::{boxed::Box, error::Error, io::Write, time::Duration};
use surf_n_term::{Face, Position, SystemTerminal, Terminal, TerminalCommand};

fn main() -> Result<(), Box<dyn Error>> {
    // let bg = Face::default().with_bg(Some("#3c3836".parse()?));
    let purple = Face::default().with_bg(Some("#d3869b".parse()?));
    let green = Face::default().with_bg(Some("#b8bb26".parse()?));
    let red = Face::default().with_bg(Some("#fb4934".parse()?));

    let mut term = SystemTerminal::new()?;
    {
        use TerminalCommand::*;
        term.execute(CursorSave)?;

        term.execute(CursorTo(Position::new(20, 0)))?;
        term.execute(Face(purple))?;
        write!(&mut term, "\x1b[1J")?;

        term.execute(CursorTo(Position::new(0, 0)))?;
        write!(&mut term, "Erase chars")?;
        term.execute(CursorTo(Position::new(1, 20)))?;
        term.execute(Face(green))?;
        write!(&mut term, "\x1b[10X")?;

        term.execute(CursorTo(Position::new(3, 0)))?;
        write!(&mut term, "Erase right")?;
        term.execute(CursorTo(Position::new(4, 10)))?;
        term.execute(Face(green))?;
        term.execute(EraseLineRight)?;

        // Erase rect area
        term.execute(Face(red))?;
        write!(&mut term, "\x1b[5;5;10;10$z")?;

        term.execute(CursorRestore)?;
    }

    term.poll(Some(Duration::from_secs(0)))?;

    Ok(())
}
