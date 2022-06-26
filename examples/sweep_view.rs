use std::io::Write;
use surf_n_term::{
    render::TerminalRenderer,
    view::{Align, Container, Flex, Text, View},
    Error, Position, SurfaceMut, SystemTerminal, Terminal, TerminalCommand, TerminalSurfaceExt,
};

const HIGHT: usize = 10;

fn sweep_view() -> Result<impl View, Error> {
    /*
    let input = Flex::row(
        Text::default().face("")
    )
    */
    let list = Flex::row().add_child(Text::text(" test "));
    Ok(Container::new(list)
        .color("#00f000".parse()?)
        .vertical(Align::Fill))
}

fn main() -> Result<(), Error> {
    let mut term = SystemTerminal::new()?;

    // reserve space, sweep uses more sophisticated method with scroll
    for _ in 0..HIGHT {
        write!(term, "\r\n")?;
    }
    term.execute(TerminalCommand::CursorMove {
        row: -(HIGHT as i32),
        col: 0,
    })?;

    // rendering
    let pos = term.position()?;
    let mut render = TerminalRenderer::new(&mut term, false)?;
    let mut surf = render.view();
    {
        // render sub-surface
        let mut surf = surf.view_mut(pos.row..pos.row + HIGHT, ..100);
        surf.erase("bg=#ff0000".parse()?);
        surf.draw_view(sweep_view()?)?;
    }
    render.frame(&mut term)?;

    // move below view
    term.execute(TerminalCommand::CursorTo(Position {
        row: pos.row + HIGHT,
        col: 0,
    }))?;

    Ok(())
}
