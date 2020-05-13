use std::{boxed::Box, error::Error, time::Duration};
use surf_n_term::{Cell, Face, Renderer, Surface, SystemTerminal, Terminal, ViewMut};

fn main() -> Result<(), Box<dyn Error>> {
    let bg = Face::default().with_bg(Some("#3c3836".parse()?));
    let purple = Face::default().with_bg(Some("#d3869b".parse()?));
    let green = Face::default().with_bg(Some("#b8bb26".parse()?));
    let red = Face::default().with_bg(Some("#fb4934".parse()?));

    let mut surface: Surface<Cell> = Surface::new(10, 20);
    surface.fill(|_, _, cell| cell.with_face(bg));
    surface
        .view_mut(..1, ..2)
        .fill(|_, _, cell| cell.with_face(purple));
    surface
        .view_mut(-1.., -2..)
        .fill(|_, _, cell| cell.with_face(green));
    surface
        .view_mut(.., 3..4)
        .fill(|_, _, cell| cell.with_face(red));
    surface
        .view_mut(3..4, ..-1)
        .fill(|_, _, cell| cell.with_face(red));

    let mut term = SystemTerminal::new()?;
    term.render(&surface)?;
    term.poll(Some(Duration::from_secs(0)))?;

    Ok(())
}
