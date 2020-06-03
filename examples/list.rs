use surf_n_term::{
    Error, Key, KeyName, SurfaceMut, SystemTerminal, Terminal, TerminalAction, TerminalEvent,
    TerminalSurfaceExt, RGBA,
};

fn main() -> Result<(), Error> {
    let q = TerminalEvent::Key(Key::from(KeyName::Char('q')));
    let up = TerminalEvent::Key(Key::from(KeyName::Up));
    let down = TerminalEvent::Key(Key::from(KeyName::Down));
    let left = TerminalEvent::Key(Key::from(KeyName::Left));
    let right = TerminalEvent::Key(Key::from(KeyName::Right));

    let bg: RGBA = "#fbf1c7".parse()?;

    let mut term = SystemTerminal::new()?;
    term.duplicate_output("/tmp/list_example.log")?;
    term.run_render(|term, event, mut view| -> Result<_, Error> {
        view.view_mut(..10, ..).erase(bg);
        if let Some(event) = event {
            if &event == &q {
                return Ok(TerminalAction::Quit);
            }
        }

        Ok(TerminalAction::Wait)
    })?;

    Ok(())
}
