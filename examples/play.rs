use surf_n_term::{
    DecMode, Error, Key, KeyName, SystemTerminal, Terminal, TerminalAction, TerminalCommand,
    TerminalEvent,
};

fn main() -> Result<(), Error> {
    let q = TerminalEvent::Key(Key::from(KeyName::Char('q')));
    let mut term = SystemTerminal::new()?;

    // init
    term.execute(TerminalCommand::CursorSave)?;
    term.execute(TerminalCommand::DecModeSet {
        enable: false,
        mode: DecMode::VisibleCursor,
    })?;

    // loop
    term.run_render(|_term, event, mut _view| -> Result<_, Error> {
        // quit
        let _event = match event {
            None => return Ok(TerminalAction::Wait),
            Some(event) if &event == &q => return Ok(TerminalAction::Quit),
            Some(event) => event,
        };

        Ok(TerminalAction::Wait)
    })?;

    // clean up
    term.execute(TerminalCommand::CursorRestore)?;

    Ok(())
}
