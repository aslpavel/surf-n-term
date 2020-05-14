use std::{boxed::Box, error::Error, time::Duration};
use surf_n_term::{
    DecMode, Key, KeyName, SystemTerminal, Terminal, TerminalCommand, TerminalEvent,
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut term = SystemTerminal::new()?;

    // get size
    println!("{:?}", term.size()?);

    // query DEC modes
    use TerminalCommand::*;
    term.execute(DecModeReport(DecMode::VisibleCursor))?;
    term.execute(DecModeReport(DecMode::AutoWrap))?;
    term.execute(DecModeReport(DecMode::MouseReport))?;
    term.execute(DecModeReport(DecMode::MouseSGR))?;
    term.execute(DecModeReport(DecMode::MouseMotions))?;
    term.execute(DecModeReport(DecMode::AltScreen))?;
    term.execute(DecModeReport(DecMode::KittyKeyboard))?;
    term.execute(CursorReport)?;

    // enable mouse
    term.execute(TerminalCommand::DecModeSet {
        enable: true,
        mode: DecMode::MouseReport,
    })?;
    term.execute(TerminalCommand::DecModeSet {
        enable: true,
        mode: DecMode::MouseMotions,
    })?;
    term.execute(TerminalCommand::DecModeSet {
        enable: true,
        mode: DecMode::MouseSGR,
    })?;

    use std::io::Write;
    // term.write_all(b"\x1bP+q62656c\x1b\\")?;
    term.write_all(b"\x1b[14t")?;

    // read terminal events
    let timeout = Duration::from_secs(10);
    println!("Program will exit after {:?} of idling ...", timeout);
    let q = Key::from(KeyName::Char('q'));
    while let Some(event) = term.poll(Some(timeout))? {
        match event {
            TerminalEvent::Key(key) if key == q => break,
            _ => println!("{:?}", event),
        }
    }

    Ok(())
}
