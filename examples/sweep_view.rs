use std::io::Write;
use surf_n_term::{
    render::TerminalRenderer,
    view::{Align, Container, Flex, Text, View},
    Error, Face, FaceAttrs, Position, SurfaceMut, SystemTerminal, Terminal, TerminalCommand,
    TerminalSurfaceExt, RGBA,
};

const HIGHT: usize = 10;

fn sweep_view() -> Result<impl View, Error> {
    let fg: RGBA = "#3c3836".parse()?;
    let bg: RGBA = "#fbf1c7".parse()?;
    let accent: RGBA = "#8f3f71".parse()?;
    /*
    let input = Flex::row(
        Text::default().face("")
    )
    */

    let prompt_face = Face::new(Some(bg), Some(accent), FaceAttrs::EMPTY);
    let default = Face::new(Some(fg), Some(bg), FaceAttrs::EMPTY);
    let list = Flex::row()
        .add_child(
            Text::text(" Input ")
                .with_face(prompt_face)
                .add_text(Text::text(" ").with_face(prompt_face.invert().with_bg(Some(bg)))),
        )
        .add_flex_child(
            1.0,
            Container::new(Text::text("query").with_face(default))
                .with_horizontal(Align::End)
                .with_color(bg),
        );
    Ok(Container::new(list)
        .with_color("#00f000".parse()?)
        .with_horizontal(Align::Fill))
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
