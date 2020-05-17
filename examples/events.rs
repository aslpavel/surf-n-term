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
    term.execute(DecModeGet(DecMode::VisibleCursor))?;
    term.execute(DecModeGet(DecMode::AutoWrap))?;
    term.execute(DecModeGet(DecMode::MouseReport))?;
    term.execute(DecModeGet(DecMode::MouseSGR))?;
    term.execute(DecModeGet(DecMode::MouseMotions))?;
    term.execute(DecModeGet(DecMode::AltScreen))?;
    term.execute(DecModeGet(DecMode::KittyKeyboard))?;
    term.execute(CursorGet)?;

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

    // run programm
    term.run(|_term, event| -> Result<_, Box<dyn Error>> {
        use surf_n_term::terminal::TerminalAction::*;
        match event {
            None => Ok(Quit),
            Some(TerminalEvent::Key(key)) if key == q => Ok(Quit),
            Some(event) => {
                println!("{:?}", event);
                Ok(Sleep(timeout))
            }
        }
    })?;

    Ok(())
}
