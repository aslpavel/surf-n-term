use std::{ops::Range, time::Duration};
use surf_n_term::{
    Color, ColorExt, DecMode, Error, Key, KeyMod, KeyName, SurfaceMut, SurfaceOwned,
    SystemTerminal, Terminal, TerminalAction, TerminalCommand, TerminalEvent, TerminalSurfaceExt,
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

fn mandelbrot(
    xs: Range<f64>,
    ys: Range<f64>,
    colors: &Range<Color>,
    count: usize,
    mut img: impl SurfaceMut<Item = Color>,
) {
    if img.height() < 2 || img.width() < 2 {
        return;
    }
    let height = (img.height() - 1) as f64;
    let width = (img.width() - 1) as f64;
    img.fill_with(|row, col, _| {
        let x = lerp(&xs, (col as f64 + 0.5) / width);
        let y = lerp(&ys, (row as f64 + 0.5) / height);
        let ratio = mandelbrot_at(x, y, count) as f32 / (count as f32);
        colors.start.lerp(colors.end, ratio)
    })
}

fn lerp(vs: &Range<f64>, t: f64) -> f64 {
    vs.start * (1.0 - t) + vs.end * t
}

fn main() -> Result<(), Error> {
    let q = TerminalEvent::Key(Key::from(KeyName::Char('q')));
    let ctrl_c = TerminalEvent::Key(Key::new(KeyName::Char('c'), KeyMod::CTRL));
    let mut term = SystemTerminal::new()?;
    term.duplicate_output("/tmp/surf_n_term.log")?;

    // init
    term.execute(TerminalCommand::CursorSave)?;
    term.execute(TerminalCommand::DecModeSet {
        enable: false,
        mode: DecMode::VisibleCursor,
    })?;
    term.execute(TerminalCommand::DecModeSet {
        enable: true,
        mode: DecMode::AltScreen,
    })?;

    // loop
    let delay = Duration::from_millis(20);
    let mut count = 1;
    let mut img = SurfaceOwned::new(58, 100);
    let color_start = "#000000".parse()?;
    let color_end = "#ffffff".parse()?;
    let colors = color_start..color_end;
    term.run_render(|_term, event, mut view| -> Result<_, Error> {
        mandelbrot(-2.5..1.0, -1.0..1.0, &colors, 1 + (count % 60), &mut img);
        view.draw_image_ascii(&img);
        count += 1;

        // quit
        match event {
            Some(event) if &event == &q || &event == &ctrl_c => Ok(TerminalAction::Quit),
            _ => Ok(TerminalAction::Sleep(delay)),
        }
    })?;

    // clean up
    term.execute(TerminalCommand::DecModeSet {
        enable: false,
        mode: DecMode::AltScreen,
    })?;
    term.execute(TerminalCommand::CursorRestore)?;

    Ok(())
}
