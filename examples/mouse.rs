use std::io::Write;
use surf_n_term::{
    error::Error, Cell, DecMode, Key, KeyName, SurfaceMut, SystemTerminal, Terminal,
    TerminalAction, TerminalCommand, TerminalEvent, TerminalSurfaceExt,
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
    let red = "bg=#fb4935".parse()?;
    let mut count = 0;
    term.run_render(|term, event, mut view| -> Result<_, Error> {
        count += 1;

        // render box
        view.draw_box(None);

        // quit
        let event = match event {
            None => return Ok(TerminalAction::Wait),
            Some(event) if &event == &q => return Ok(TerminalAction::Quit(())),
            Some(event) => event,
        };

        // render label with event
        let mut label = view.view_mut(0, 3..-3);
        let mut writer = label.writer();
        write!(
            &mut writer,
            "┤ Stats: {:?} Count: {} Event: {:?} ├",
            term.stats(),
            count,
            event
        )?;

        // render mouse cursor
        match event {
            TerminalEvent::Mouse(mouse) if mouse.name == KeyName::MouseMove => {
                if let Some(cell) = view.get_mut(mouse.row, mouse.col) {
                    *cell = Cell::new(red, None);
                }
            }
            _ => (),
        }

        Ok(TerminalAction::Wait)
    })?;

    // switch off alt screen
    term.execute(TerminalCommand::DecModeSet {
        enable: false,
        mode: DecMode::AltScreen,
    })?;
    term.poll(Some(std::time::Duration::new(0, 0)))?;

    Ok(())
}
