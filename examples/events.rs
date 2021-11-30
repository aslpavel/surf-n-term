use std::{boxed::Box, error::Error, io::Write, time::Duration};
use surf_n_term::{
    DecMode, SystemTerminal, Terminal, TerminalColor, TerminalCommand, TerminalEvent,
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut term = SystemTerminal::new()?;

    // query DEC modes
    use TerminalCommand::*;
    term.execute(DecModeGet(DecMode::VisibleCursor))?;
    term.execute(DecModeGet(DecMode::AutoWrap))?;
    term.execute(DecModeGet(DecMode::MouseReport))?;
    term.execute(DecModeGet(DecMode::MouseSGR))?;
    term.execute(DecModeGet(DecMode::MouseMotions))?;
    term.execute(DecModeGet(DecMode::AltScreen))?;
    term.execute(DecModeGet(DecMode::SynchronizedOutput))?;
    term.execute(Color {
        name: TerminalColor::Palette(1),
        color: None,
    })?;
    term.execute(Color {
        name: TerminalColor::Foreground,
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
    term.execute(TerminalCommand::Face("bg=#8f3f71".parse()?))?;
    term.execute(TerminalCommand::FaceGet)?;
    term.execute(TerminalCommand::Face("".parse()?))?;

    term.execute(TerminalCommand::Termcap(vec![
        "bel".to_string(),
        "smcup".to_string(),
        "TN".to_string(),
        "Co".to_string(),
    ]))?;
    term.write_all(b"\x1b[18t\x1b[14t")?;
    term.execute(TerminalCommand::Title("events test title".to_string()))?;

    let caps = term.capabilities().clone();
    write!(&mut term, "Terminal::capabilities(): {:?}\r\n", caps)?;
    let size = term.size()?;
    write!(&mut term, "Terminal::size(): {:?}\r\n", size)?;
    let image_handler = term.image_handler().kind();
    write!(
        &mut term,
        "Terminal::image_handler(): {:?}\r\n",
        image_handler
    )?;

    // read terminal events
    let timeout = Duration::from_secs(10);
    write!(
        &mut term,
        "\x1b[91mProgram will exit after {:?} of idling or if 'q' is pressed ...\x1b[m\r\n",
        timeout
    )?;

    let q_key = "q".parse()?;

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
            Some(TerminalEvent::Key(key)) if key == q_key => Ok(Quit(())),
            Some(event) => {
                write!(&mut term, "{:?}\r\n", event)?;
                Ok(Sleep(timeout))
            }
        }
    })?;

    Ok(())
}
