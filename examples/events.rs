use std::{boxed::Box, error::Error, io::Write, time::Duration};
use surf_n_term::{
    DecMode, Key, KeyName, SystemTerminal, Terminal, TerminalColor, TerminalCommand, TerminalEvent,
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut term = SystemTerminal::new()?;

    // get size
    let size = term.size()?;
    write!(&mut term, "{:?}\r\n", size)?;

    // query DEC modes
    use TerminalCommand::*;
    term.execute(DecModeGet(DecMode::VisibleCursor))?;
    term.execute(DecModeGet(DecMode::AutoWrap))?;
    term.execute(DecModeGet(DecMode::MouseReport))?;
    term.execute(DecModeGet(DecMode::MouseSGR))?;
    term.execute(DecModeGet(DecMode::MouseMotions))?;
    term.execute(DecModeGet(DecMode::AltScreen))?;
    term.execute(DecModeGet(DecMode::KittyKeyboard))?;
    term.execute(Color {
        name: TerminalColor::Palette(1),
        color: None,
    })?;
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

    term.execute(TerminalCommand::Termcap(vec![
        "bel".to_string(),
        "smcup".to_string(),
        "TN".to_string(),
        "Co".to_string(),
    ]))?;
    term.write_all(b"\x1b[14t")?;

    // read terminal events
    let timeout = Duration::from_secs(10);
    write!(
        &mut term,
        "Program will exit after {:?} of idling ...\r\n",
        timeout
    )?;
    let q = Key::from(KeyName::Char('q'));

    let waker = term.waker();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(2));
        waker.wake().expect("wake failed");
    });

    // run programm
    term.run(None, |mut term, event| -> Result<_, Box<dyn Error>> {
        use surf_n_term::terminal::TerminalAction::*;
        match event {
            None => Ok(Quit(())),
            Some(TerminalEvent::Key(key)) if key == q => Ok(Quit(())),
            Some(event) => {
                write!(&mut term, "{:?}\r\n", event)?;
                Ok(Sleep(timeout))
            }
        }
    })?;

    Ok(())
}
