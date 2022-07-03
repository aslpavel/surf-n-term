use std::io::Write;
use surf_n_term::{
    render::TerminalRenderer,
    view::{Align, Axis, Container, Flex, ScrollBar, Text, View},
    Blend, Color, Error, Face, FaceAttrs, Position, SurfaceMut, SystemTerminal, Terminal,
    TerminalCommand, TerminalSurfaceExt, RGBA,
};

const HIGHT: usize = 10;

fn sweep_view<'a>(items: impl IntoIterator<Item = &'a str>) -> Result<impl View + 'a, Error> {
    let fg: RGBA = "#3c3836".parse()?;
    let bg: RGBA = "#fbf1c7".parse()?;
    let accent: RGBA = "#8f3f71".parse()?;

    let input_face = Face::new(Some(fg), Some(bg), FaceAttrs::EMPTY);
    let list_default_face = Face::new(
        Some(bg.blend(fg.with_alpha(0.9), Blend::Over)),
        Some(bg),
        FaceAttrs::EMPTY,
    );
    let list_selected_face = Face::new(
        Some(bg.blend(fg, Blend::Over)),
        Some(bg.blend(fg.with_alpha(0.1), Blend::Over)),
        FaceAttrs::EMPTY,
    );
    let scrollbar_face = Face::new(
        Some(accent.with_alpha(0.8)),
        Some(accent.with_alpha(0.5)),
        FaceAttrs::EMPTY,
    );
    let prompt_face = Face::new(Some(bg), Some(accent), FaceAttrs::EMPTY);

    // prompt | input | status
    let input_view = Flex::row()
        .add_child(
            Text::text(" Input ")
                .with_face(prompt_face)
                .add_text(Text::text(" ").with_face(prompt_face.invert().with_bg(Some(bg)))),
        )
        .add_flex_child(
            1.0,
            Container::new(Text::text("query").with_face(input_face))
                .with_horizontal(Align::Fill)
                .with_color(bg),
        )
        .add_child(
            Text::text("")
                .with_face(prompt_face)
                .add_text(Text::text("").with_face(prompt_face.invert()))
                .add_text(" 30/127 1us [fuzzy] "),
        );

    let items_list = items
        .into_iter()
        .enumerate()
        .fold(Flex::column(), |list, (index, item)| {
            let (tag, face) = if index == 1 {
                let tag = Text::text(" ●  ").with_face(list_selected_face.with_fg(Some(accent)));
                (tag, list_selected_face)
            } else {
                let tag = Text::text("    ").with_face(list_default_face);
                (tag, list_default_face)
            };
            list.add_child(
                Container::new(
                    Flex::row().add_child(tag).add_flex_child(
                        1.0,
                        Container::new(Text::text(item).with_face(face))
                            .with_horizontal(Align::Fill)
                            .with_color(bg),
                    ),
                )
                .with_color(face.bg.unwrap()),
            )
        });

    let list = Flex::row()
        .add_flex_child(1.0, Container::new(items_list).with_horizontal(Align::Fill))
        .add_child(ScrollBar::new(Axis::Vertical, scrollbar_face, 127, 40, 30));
    Ok(Container::new(
        Flex::column()
            .add_child(input_view)
            .add_flex_child(1.0, Container::new(list).with_color(bg)),
    )
    .with_vertical(Align::Fill))
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
        surf.draw_view(sweep_view([
            "first item",
            "multi\n line",
            "thrid element",
            "and another",
            "and plus one",
            "one more",
            "some other elements",
        ])?)?;
    }
    render.frame(&mut term)?;

    // move below view
    term.execute(TerminalCommand::CursorTo(Position {
        row: pos.row + HIGHT,
        col: 0,
    }))?;

    Ok(())
}
