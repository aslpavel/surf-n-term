use std::{boxed::Box, error::Error, time::Duration};
use surf_n_term::{Cell, Face, Renderer, Surface, SystemTerminal, Terminal, ViewMutExt};

fn main() -> Result<(), Box<dyn Error>> {
    let bg = Face::default().with_bg(Some("#3c3836".parse()?));
    let purple = Face::default().with_bg(Some("#d3869b".parse()?));
    let green = Face::default().with_bg(Some("#b8bb26".parse()?));
    let red = Face::default().with_bg(Some("#fb4934".parse()?));

    let mut surface: Surface<Cell> = Surface::new(10, 20);
    surface.fill(Cell::new(bg, None));
    surface.view_mut(..1, ..2).fill(Cell::new(purple, None));
    surface.view_mut(-1.., -2..).fill(Cell::new(green, None));
    surface.view_mut(.., 3..4).fill(Cell::new(red, None));
    surface.view_mut(3..4, ..-1).fill(Cell::new(red, None));

    let mut term = SystemTerminal::new()?;
    term.render(&surface)?;
    term.poll(Some(Duration::from_secs(0)))?;

    Ok(())
}
