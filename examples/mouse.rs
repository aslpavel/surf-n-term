use surf_n_term::render::{run, RenderAction};
use surf_n_term::{
    error::Error, Cell, DecMode, Face, Key, KeyName, Position, SystemTerminal, Terminal,
    TerminalCommand, TerminalEvent, TerminalView,
};

fn main() -> Result<(), Error> {
    let mut term = SystemTerminal::new()?;

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
    term.execute(TerminalCommand::DecModeSet {
        enable: true,
        mode: DecMode::AltScreen,
    })?;
    term.execute(TerminalCommand::DecModeSet {
        enable: false,
        mode: DecMode::VisibleCursor,
    })?;

    let q = TerminalEvent::Key(Key::from(KeyName::Char('q')));
    let red = Face::default().with_bg(Some("#fb4934".parse().unwrap()));
    run(&mut term, |event, view| -> Result<_, Error> {
        view.draw_box(None);

        let event = match event {
            None => return Ok(RenderAction::Continue),
            Some(event) if &event == &q => return Ok(RenderAction::Quit),
            Some(event) => event,
        };

        let label = format!("┤ Event: {:?} ├", event);
        view.draw_text(Position::new(0, 3), None, &label);
        match event {
            TerminalEvent::Mouse(mouse) if mouse.name == KeyName::MouseMove => {
                if let Some(cell) = view.get_mut(mouse.row, mouse.col) {
                    *cell = Cell::new(red, None);
                }
            }
            _ => (),
        }

        Ok(RenderAction::Continue)
    })?;

    // disable mouse
    term.execute(TerminalCommand::DecModeSet {
        enable: false,
        mode: DecMode::AltScreen,
    })?;
    term.poll(Some(std::time::Duration::new(0, 0)))?;

    Ok(())
}
