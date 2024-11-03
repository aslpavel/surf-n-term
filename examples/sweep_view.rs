use std::time::Duration;
use surf_n_term::{
    render::TerminalRenderer,
    view::{
        Align, Axis, Container, Flex, FlexChild, FlexRef, Frame, ScrollBar, ScrollBarPosition,
        Text, View, ViewContext,
    },
    CellWrite, Color, Error, Face, FaceAttrs, Position, SurfaceMut, SystemTerminal, Terminal,
    TerminalCommand, TerminalSurfaceExt, RGBA,
};

const HIGHT: usize = 10;
const ICON: &str = "
{
    \"path\": \"M19.873 3.49a.75.75 0 1 0-.246-1.48l-6 1a.75.75 0 0 0-.613.593L12.736\\n\
    5H5.75a.75.75 0 0 0-.75.75v4a3.25 3.25 0 0 0 3 3.24v.51c0 1.953 1.4 3.579\\n\
    3.25 3.93v3.07h-2.5a.75.75 0 0 0 0 1.5h6.5a.75.75 0 0 0 0-1.5h-2.5v-3.07A4.001\\n\
    4.001 0 0 0 16 13.5v-.51a3.25 3.25 0 0 0 3-3.24v-4a.75.75 0 0 0-.75-.75h-3.985\\n\
    l.119-.595 5.49-.915ZM17.5 8h-3.835l.3-1.5H17.5V8Zm-4.135 1.5H17.5v.25a1.75\\n\
    1.75 0 0 1-1.75 1.75h-.5a.75.75 0 0 0-.75.75v1.25a2.5 2.5 0 0 1-5 0v-1.25\\n\
    a.75.75 0 0 0-.75-.75h-.5A1.75 1.75 0 0 1 6.5 9.75V9.5h5.335l-.82 4.103a.75\\n\
    .75 0 1 0 1.47.294l.88-4.397ZM12.135 8H6.5V6.5h5.935l-.3 1.5Z\",
    \"view\": [0, 0, 24, 24],
    \"size\": [1, 3]
}
";

fn sweep_view<'a>(items: impl IntoIterator<Item = &'a str>) -> Result<impl View + 'a, Error> {
    let fg: RGBA = "#ebdbb2".parse()?;
    let bg: RGBA = "#282828".parse()?;
    let accent: RGBA = "#d3869b".parse()?;
    let icon = serde_json::from_str(ICON)?;

    let input_face = Face::new(Some(fg), Some(bg), FaceAttrs::EMPTY);
    let list_default_face = Face::new(
        Some(bg.blend_over(fg.with_alpha(0.9))),
        Some(bg),
        FaceAttrs::EMPTY,
    );
    let list_selected_face = Face::new(
        Some(fg),
        Some(bg.blend_over(fg.with_alpha(0.05))),
        FaceAttrs::EMPTY,
    );
    let scrollbar_face = Face::new(
        Some(accent.with_alpha(0.8)),
        Some(accent.with_alpha(0.5)),
        FaceAttrs::EMPTY,
    );
    let prompt_face = Face::new(Some(bg), Some(accent), FaceAttrs::EMPTY);

    // prompt | input | status
    let input_view = FlexRef::new((
        FlexChild::new(
            Text::new()
                .with_face(prompt_face)
                .with_glyph(icon)
                .put_fmt("Input ", None)
                .put_fmt(" ", Some(prompt_face.invert().with_bg(Some(bg))))
                .take(),
        ),
        FlexChild::new(
            Container::new(Text::new().put_fmt("query", Some(input_face)).take())
                .with_horizontal(Align::Expand)
                .with_color(bg),
        )
        .flex(1.0),
        FlexChild::new(
            Text::new()
                .put_fmt("", Some(prompt_face.invert()))
                .put_fmt(" 30/127 1us [fuzzy] ", Some(prompt_face))
                .take(),
        ),
    ));

    let items_list = items
        .into_iter()
        .enumerate()
        .fold(Flex::column(), |list, (index, item)| {
            let (tag, face) = if index == 1 {
                let tag = Text::new()
                    .put_fmt(" ●  ", Some(list_selected_face.with_fg(Some(accent))))
                    .take();
                (tag, list_selected_face)
            } else {
                let tag = Text::new().put_fmt("    ", Some(list_default_face)).take();
                (tag, list_default_face)
            };
            list.add_child(
                Container::new(
                    Flex::row().add_child(tag).add_flex_child(
                        1.0,
                        Container::new(Text::new().put_fmt(item, Some(face)).take())
                            .with_horizontal(Align::Expand)
                            .with_face(face),
                    ),
                )
                .with_color(face.bg.unwrap()),
            )
        });

    let list = Flex::row()
        .add_flex_child(
            1.0,
            Container::new(items_list).with_horizontal(Align::Expand),
        )
        .add_child(ScrollBar::new(
            Axis::Vertical,
            scrollbar_face,
            ScrollBarPosition {
                offset: 0.3150,
                visible: 0.2362,
            },
        ));

    let result = Container::new(
        Flex::column()
            .add_child(input_view)
            .add_flex_child(1.0, Container::new(list).with_color(bg)),
    )
    .with_vertical(Align::Expand);

    Ok(Frame::new(result, bg, accent, 0.2, 0.9))
}

fn render(term: &mut dyn Terminal) -> Result<(), Error> {
    // reserve space, sweep uses more sophisticated method with scroll
    term.execute(TerminalCommand::Scroll(HIGHT as i32))?;
    term.execute(TerminalCommand::CursorMove {
        row: -(HIGHT as i32),
        col: 0,
    })?;

    // rendering
    let pos = term.position()?;
    let mut render = TerminalRenderer::new(term, false)?;
    let mut surf = render.surface();
    {
        // render sub-surface
        let ctx = ViewContext::new(term.dyn_ref())?;
        let mut surf = surf.view_mut(pos.row..pos.row + HIGHT, ..100);
        surf.erase("bg=#ff0000".parse()?);
        surf.draw_view(
            &ctx,
            None,
            sweep_view([
                "first item",
                "multi\n line",
                "thrid element",
                "and another",
                "and plus one",
                "one more",
                "some other elements",
            ])?,
        )?;
    }
    render.frame(term)?;

    // move below view
    term.execute(TerminalCommand::CursorTo(Position {
        row: pos.row + HIGHT,
        col: 0,
    }))?;
    term.poll(Some(Duration::from_millis(100)))?;

    Ok(())
}

fn main() -> Result<(), Error> {
    let mut term = SystemTerminal::new()?;
    render(&mut term)?;

    Ok(())
}
