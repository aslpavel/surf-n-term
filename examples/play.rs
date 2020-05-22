use std::ops::Range;
use surf_n_term::{
    Color, DecMode, Error, Key, KeyName, StorageMut, Surface, SystemTerminal, Terminal,
    TerminalAction, TerminalCommand, TerminalEvent,
};

fn mandelbrot_at(x0: f64, y0: f64, count: usize) -> usize {
    (0..count)
        .try_fold([0.0, 0.0], |[x, y], i| {
            if x * x + y * y >= 4.0 {
                Err(i)
            } else {
                Ok([x * x - y * y + x0, 2.0 * x * y + y0])
            }
        })
        .err()
        .unwrap_or(count)
}

fn mandelbrot<S>(xs: Range<f64>, ys: Range<f64>, colors: Range<Color>, img: &mut Surface<S>)
where
    S: StorageMut<Item = Color>,
{
    todo!()
}

fn lerp(vs: Range<f64>, t: f64) -> f64 {
    todo!()
}

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

        // view.draw_image_ascii

        Ok(TerminalAction::Wait)
    })?;

    // clean up
    term.execute(TerminalCommand::CursorRestore)?;

    Ok(())
}
