use std::{error::Error, io::Write, time::Duration};
use surf_n_term::{
    DecMode, SystemTerminal, Terminal, TerminalColor, TerminalCommand, TerminalEvent,
};

fn header(term: &mut dyn Terminal, content: impl std::fmt::Display) -> Result<(), Box<dyn Error>> {
    term.execute(TerminalCommand::Face("fg=#b8bb26,bg=#3c3836,bold".parse()?))?;
    term.execute(TerminalCommand::EraseLine)?;
    write!(term, " {}", content)?;
    term.execute(TerminalCommand::Face(Default::default()))?;
    write!(term, "\r\n")?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    use TerminalCommand::*;
    let mut term = SystemTerminal::new()?;

    let caps = term.capabilities().clone();
    let size = term.size()?;
    let image_handler = term.image_handler().kind();

    // show terminal info
    header(&mut term, "Terminal info")?;
    write!(&mut term, "Capabilities  : {:?}\r\n", caps)?;
    write!(&mut term, "Terminal Size : {:?}\r\n", size)?;
    write!(
        &mut term,
        "Cell Size     : {:?}\r\n",
        size.pixels_per_cell()
    )?;
    write!(&mut term, "Image Handler : {:?}\r\n", image_handler)?;

    // message
    let timeout = Duration::from_secs(10);
    header(
        &mut term,
        format!(
            "Program will exit after {:?} of idling or if 'q' is pressed ...",
            timeout
        ),
    )?;

    // trigger some events
    term.execute_many(TerminalCommand::mouse_events_set(true, true))?;
    term.execute_many([
        DecModeGet(DecMode::VisibleCursor),
        DecModeGet(DecMode::AutoWrap),
        DecModeGet(DecMode::MouseReport),
        DecModeGet(DecMode::MouseSGR),
        DecModeGet(DecMode::MouseMotions),
        DecModeGet(DecMode::AltScreen),
        DecModeGet(DecMode::SynchronizedOutput),
        Color {
            name: TerminalColor::Palette(1),
            color: None,
        },
        Color {
            name: TerminalColor::Foreground,
            color: None,
        },
        CursorGet,
        Face("bg=#8f3f71".parse()?),
        FaceGet,
        Face(Default::default()),
        Termcap(
            ["bel", "smcup", "TN", "Co"]
                .into_iter()
                .map(|s| s.to_owned())
                .collect(),
        ),
    ])?;
    term.write_all(b"\x1b[18t\x1b[14t")?;
    term.execute(Title("events test title".to_string()))?;

    let q_key = "q".parse()?;
    let waker = term.waker();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(2));
        waker.wake().expect("wake failed");
    });

    // run program
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
