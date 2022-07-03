use std::io::Write;
use surf_n_term::{
    render::TerminalRenderer,
    view::{Align, Axis, Container, Flex, ScrollBar, Text, View},
    BBox, Blend, Color, Error, Face, FaceAttrs, FillRule, Glyph, Path, Position, Size, SurfaceMut,
    SystemTerminal, Terminal, TerminalCommand, TerminalSurfaceExt, RGBA,
};

const HIGHT: usize = 10;

fn sweep_view<'a>(items: impl IntoIterator<Item = &'a str>) -> Result<impl View + 'a, Error> {
    let fg: RGBA = "#3c3836".parse()?;
    let bg: RGBA = "#fbf1c7".parse()?;
    let accent: RGBA = "#8f3f71".parse()?;
    let icon = Glyph::new(
        "M19.873 3.49a.75.75 0 1 0-.246-1.48l-6 1a.75.75 0 0 0-.613.593L12.736
        5H5.75a.75.75 0 0 0-.75.75v4a3.25 3.25 0 0 0 3 3.24v.51c0 1.953 1.4 3.579
        3.25 3.93v3.07h-2.5a.75.75 0 0 0 0 1.5h6.5a.75.75 0 0 0 0-1.5h-2.5v-3.07A4.001
        4.001 0 0 0 16 13.5v-.51a3.25 3.25 0 0 0 3-3.24v-4a.75.75 0 0 0-.75-.75h-3.985
        l.119-.595 5.49-.915ZM17.5 8h-3.835l.3-1.5H17.5V8Zm-4.135 1.5H17.5v.25a1.75
        1.75 0 0 1-1.75 1.75h-.5a.75.75 0 0 0-.75.75v1.25a2.5 2.5 0 0 1-5 0v-1.25
        a.75.75 0 0 0-.75-.75h-.5A1.75 1.75 0 0 1 6.5 9.75V9.5h5.335l-.82 4.103a.75
        .75 0 1 0 1.47.294l.88-4.397ZM12.135 8H6.5V6.5h5.935l-.3 1.5Z"
            .parse::<Path>()?,
        FillRule::NonZero,
        Some(BBox::new((0.0, 0.0), (24.0, 24.0))),
        Size::new(1, 3),
    );

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
            Text::text("")
                .with_face(prompt_face)
                .add_text(Text::glyph(icon))
                .add_text("Input ")
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
