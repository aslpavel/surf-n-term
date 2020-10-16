use std::{ops::Range, time::Duration};
use surf_n_term::{
    Color, ColorLinear, DecMode, Error, Image, Surface, SurfaceMut, SurfaceOwned, SystemTerminal,
    Terminal, TerminalAction, TerminalCommand, TerminalEvent, TerminalSurfaceExt, RGBA,
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
    colors: &Range<ColorLinear>,
    count: usize,
    mut img: impl SurfaceMut<Item = RGBA>,
) {
    if img.height() < 2 || img.width() < 2 {
        return;
    }
    let height = (img.height() - 1) as f64;
    let width = (img.width() - 1) as f64;
    img.fill_with(|row, col, _| {
        let x = lerp(&xs, (col as f64 + 0.5) / width);
        let y = lerp(&ys, (row as f64 + 0.5) / height);
        let ratio = mandelbrot_at(x, y, count) as f64 / (count as f64);
        colors.start.lerp(colors.end, ratio).into()
    })
}

fn lerp(vs: &Range<f64>, t: f64) -> f64 {
    vs.start * (1.0 - t) + vs.end * t
}

fn main() -> Result<(), Error> {
    let f = TerminalEvent::Key("f".parse()?);
    let q = TerminalEvent::Key("q".parse()?);
    let ctrl_c = TerminalEvent::Key("ctrl+c".parse()?);
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
    let mut img_ascii = SurfaceOwned::new(58, 100);
    let mut img = SurfaceOwned::new(232, 400);
    let color_start = "#000000".parse()?;
    let color_end = "#ffffff".parse()?;
    let colors = color_start..color_end;
    let mut ascii = true;
    term.run_render(|_term, event, mut view| -> Result<_, Error> {
        if ascii {
            mandelbrot(
                -2.5..1.0,
                -1.0..1.0,
                &colors,
                1 + (count % 60),
                &mut img_ascii,
            );
            view.draw_image_ascii(&img_ascii);
        } else {
            mandelbrot(-2.5..1.0, -1.0..1.0, &colors, 1 + (count % 60), &mut img);
            view.draw_image(Image::new(img.to_owned_surf()));
        }
        count += 1;

        // process event
        if event.as_ref() == Some(&q) || event.as_ref() == Some(&ctrl_c) {
            Ok(TerminalAction::Quit(()))
        } else {
            if event.as_ref() == Some(&f) {
                ascii = !ascii;
            }
            Ok(TerminalAction::Sleep(delay))
        }
    })?;

    // clean up
    term.execute(TerminalCommand::DecModeSet {
        enable: false,
        mode: DecMode::AltScreen,
    })?;
    term.execute(TerminalCommand::CursorRestore)?;
    term.poll(Some(Duration::from_millis(50)))?;

    Ok(())
}
