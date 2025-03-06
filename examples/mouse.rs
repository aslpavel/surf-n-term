#![allow(clippy::reversed_empty_ranges)] // those range are not actually empty
use surf_n_term::{
    Cell, CellWrite, Face, KeyName, Position, Size, Surface, SurfaceMut, SystemTerminal, Terminal,
    TerminalAction, TerminalCommand, TerminalEvent, TerminalSurfaceExt,
    view::{Align, Container, Margins, Text, View, ViewContext, ViewLayoutStore},
};
use tracing_subscriber::{EnvFilter, fmt::format::FmtSpan};

type Error = Box<dyn std::error::Error + Sync + Send + 'static>;
const CURSOR: &str = r#"
{
    "view_box": [0, 0, 100, 100],
    "size": [2, 5],
    "frame": {
        "margin": [5, 5, 5, 5],
        "border_radius": [0, 50, 50, 50],
        "border_width": [3, 3, 3, 3],
        "border_color": "gruv-aqua-0",
        "padding": [10, 5, 10, 5],
        "fill_color": "gruv-aqua-2"
    },
    "path": "M50,89.47Q41.67,89.47 34.85,87.04Q28.03,84.61 23.11,80.16Q18.18,75.72 15.51,69.58Q12.84,63.44 12.84,56.02Q12.84,50.16 14.55,45.02Q12.96,40.19 12.14,35.83Q11.32,31.46 11.32,27.11Q11.32,19.62 13.48,15.07Q15.63,10.53 19.5,10.53Q21.81,10.53 24.74,12.32Q27.67,14.11 30.84,17.26Q34.01,20.41 37,24.48Q40.03,23.6 43.3,23.19Q46.57,22.77 50,22.77Q53.43,22.77 56.7,23.19Q59.97,23.6 63,24.48Q65.99,20.41 69.16,17.26Q72.33,14.11 75.28,12.32Q78.23,10.53 80.5,10.53Q84.37,10.53 86.52,15.07Q88.68,19.62 88.68,27.11Q88.68,31.46 87.88,35.83Q87.08,40.19 85.45,45.02Q87.16,50.16 87.16,56.02Q87.16,63.44 84.49,69.58Q81.82,75.72 76.89,80.16Q71.97,84.61 65.15,87.04Q58.33,89.47 50,89.47ZM11.12,77.11L8.65,75.12Q10.29,73.13 12.84,71.53Q15.39,69.94 18.36,68.96Q21.33,67.98 24.2,67.86L24.2,71.05Q21.81,71.17 19.3,71.99Q16.79,72.81 14.63,74.14Q12.48,75.48 11.12,77.11ZM5.86,66.23L4.55,63.28Q7.06,62.12 10.29,61.48Q13.52,60.85 16.55,60.85Q18.06,60.85 19.44,61.02Q20.81,61.2 22.05,61.56L21.21,64.51Q20.14,64.23 19.08,64.11Q18.02,64 16.75,64Q13.92,64 10.94,64.59Q7.97,65.19 5.86,66.23ZM20.18,57.89Q14.59,55.7 5.02,55.7L5.02,52.51Q10.05,52.51 14.15,53.13Q18.26,53.75 21.37,54.94L20.18,57.89ZM50,75.12Q46.65,75.12 43.98,73.13Q41.31,71.13 39.59,67.38L42.34,65.95Q43.66,68.82 45.65,70.37Q47.65,71.93 50,71.93Q52.35,71.93 54.35,70.37Q56.34,68.82 57.66,65.95L60.41,67.38Q58.69,71.13 56.02,73.13Q53.35,75.12 50,75.12ZM36.52,58.13Q29.82,56.66 26.12,53.47Q22.41,50.28 22.41,46.05Q22.41,43.34 24.12,41.55Q25.84,39.75 28.39,39.75Q30.18,39.75 32.34,40.83Q33.53,39.19 35.03,38.26Q36.52,37.32 38.16,37.32Q40.75,37.32 42.48,39.41Q44.22,41.51 44.22,44.58Q44.22,50.96 36.52,58.13ZM42.07,68.42Q39.75,68.42 37.74,67.66Q35.73,66.91 34.05,65.43L36.12,63.04Q37.28,64.07 38.86,64.65Q40.43,65.23 42.07,65.23Q44.42,65.23 46.55,63.88Q48.68,62.52 48.68,60.09L51.32,60.09Q51.32,62.52 53.47,63.88Q55.62,65.23 57.93,65.23Q59.57,65.23 61.14,64.65Q62.72,64.07 63.88,63.04L65.95,65.43Q64.27,66.91 62.26,67.66Q60.25,68.42 57.93,68.42Q55.66,68.42 53.47,67.46Q51.28,66.51 50,64.99Q48.72,66.51 46.53,67.46Q44.34,68.42 42.07,68.42ZM50,62.4Q48.41,62.4 46.91,61.14Q45.41,59.89 45.41,58.65Q45.41,57.58 46.65,57Q47.89,56.42 50,56.42Q52.11,56.42 53.35,57Q54.59,57.58 54.59,58.65Q54.59,59.89 53.11,61.14Q51.63,62.4 50,62.4ZM21.57,38.72Q20.61,36.24 20.14,32.87Q19.66,29.51 19.66,26.59Q19.66,24.08 20.02,22.39Q20.37,20.69 21.09,20.69Q21.77,20.69 23.27,22.07Q24.76,23.44 26.67,25.7Q28.59,27.95 30.42,30.54Q28.19,31.78 25.76,34.13Q23.33,36.48 21.57,38.72ZM50,86.28Q60.25,86.28 67.88,82.46Q75.52,78.63 79.74,71.81Q83.97,64.99 83.97,56.02Q83.97,50.08 82.1,45.02Q83.25,41.75 83.99,38.76Q84.73,35.77 85.11,32.87Q85.49,29.98 85.49,27.11Q85.49,21.05 84.07,17.38Q82.66,13.72 80.5,13.72Q78.71,13.72 75.94,15.65Q73.17,17.58 70.12,20.85Q67.07,24.12 64.27,28.19Q61,27.07 57.44,26.52Q53.87,25.96 50,25.96Q46.45,25.96 43.02,26.48Q39.59,26.99 35.73,28.19Q32.97,24.12 29.88,20.85Q26.79,17.58 24.06,15.65Q21.33,13.72 19.5,13.72Q17.34,13.72 15.93,17.38Q14.51,21.05 14.51,27.11Q14.51,29.98 14.89,32.87Q15.27,35.77 16.03,38.76Q16.79,41.75 17.9,45.02Q16.03,50.08 16.03,56.02Q16.03,64.99 20.26,71.81Q24.48,78.63 32.12,82.46Q39.75,86.28 50,86.28ZM88.88,77.11Q87.56,75.48 85.39,74.14Q83.21,72.81 80.72,71.99Q78.23,71.17 75.8,71.05L75.8,67.86Q78.67,67.98 81.64,68.96Q84.61,69.94 87.16,71.53Q89.71,73.13 91.35,75.12L88.88,77.11ZM44.7,35.89Q43.34,35.89 41.67,33.63Q39.99,31.38 38.16,25.16L47.97,23.64Q48.05,24.56 48.09,25.44Q48.13,26.32 48.13,27.15Q48.13,31.18 47.23,33.53Q46.33,35.89 44.7,35.89ZM63.48,58.13Q55.78,50.96 55.78,44.58Q55.78,41.51 57.52,39.41Q59.25,37.32 61.84,37.32Q65.19,37.32 67.66,40.83Q69.82,39.75 71.61,39.75Q74.2,39.75 75.9,41.55Q77.59,43.34 77.59,46.05Q77.59,50.28 73.88,53.47Q70.18,56.66 63.48,58.13ZM94.14,66.23Q92.07,65.19 89.07,64.59Q86.08,64 83.25,64Q81.98,64 80.92,64.11Q79.86,64.23 78.79,64.51L77.95,61.56Q79.23,61.2 80.58,61.02Q81.94,60.85 83.45,60.85Q86.48,60.85 89.71,61.48Q92.94,62.12 95.45,63.28L94.14,66.23ZM55.3,35.89Q53.71,35.89 52.79,33.53Q51.87,31.18 51.87,27.15Q51.87,26.32 51.91,25.44Q51.95,24.56 52.03,23.64L61.84,25.16Q60.01,31.38 58.33,33.63Q56.66,35.89 55.3,35.89ZM79.82,57.89L78.63,54.94Q81.74,53.75 85.85,53.13Q89.95,52.51 94.98,52.51L94.98,55.7Q85.41,55.7 79.82,57.89ZM78.43,38.72Q76.67,36.48 74.24,34.13Q71.81,31.78 69.58,30.54Q71.41,27.95 73.33,25.7Q75.24,23.44 76.75,22.07Q78.27,20.69 78.91,20.69Q79.63,20.69 79.98,22.39Q80.34,24.08 80.34,26.59Q80.34,29.51 79.86,32.87Q79.39,36.24 78.43,38.72Z"
}
"#;

fn stats_view(
    term: &SystemTerminal,
    event: &TerminalEvent,
    event_count: usize,
) -> Result<impl View, Error> {
    let name = "fg=#fbf1c7,bold".parse()?;
    let term_size = term.size().unwrap_or_default();
    let cell_size = term_size.cells_in_pixels(Size::new(1, 1));
    let text = Text::new()
        .put_fmt("Count     ", Some(name))
        .put_fmt(&format_args!("{}\n", event_count), None)
        .put_fmt("Events    ", Some(name))
        .put_fmt(&format_args!("{:?}\n", event), None)
        .put_fmt("Received  ", Some(name))
        .put_fmt(&format_args!("{}\n", term.stats().recv), None)
        .put_fmt("Send      ", Some(name))
        .put_fmt(&format_args!("{}\n", term.stats().send), None)
        .put_fmt("Term size ", Some(name))
        .put_fmt(
            &format_args!(
                "{}x{} {}x{} ({}x{})\n",
                term_size.cells.height,
                term_size.cells.width,
                term_size.pixels.height,
                term_size.pixels.height,
                cell_size.height,
                cell_size.width
            ),
            None,
        )
        .take();
    Ok(Container::new(text)
        .with_margins(Margins {
            right: 2,
            left: 2,
            top: 1,
            bottom: 1,
        })
        .with_face("fg=gruv-fg,bg=gruv-blue-1".parse()?))
}

fn main() -> Result<(), Error> {
    let appender = tracing_appender::rolling::never("/tmp", "surf-n-term-mouse.log");
    tracing_subscriber::fmt()
        .with_span_events(FmtSpan::CLOSE)
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(appender)
        .init();

    let mut term = SystemTerminal::new()?;

    // enable mouse
    term.execute_many(TerminalCommand::mouse_events_set(true, true))?;
    term.execute_many([
        TerminalCommand::altscreen_set(true),
        TerminalCommand::auto_wrap_set(false),
        TerminalCommand::visible_cursor_set(false),
    ])?;
    // term.duplicate_output("/tmp/mouse-example.txt")?;

    let q = TerminalEvent::Key("q".parse()?);
    let ctrlc = TerminalEvent::Key("ctrl+c".parse()?);
    let glyph_face: Face = "fg=gruv-1".parse()?;
    let glyph = if term.capabilities().glyphs {
        Cell::new_glyph(glyph_face, serde_json::from_str(CURSOR)?)
    } else {
        Cell::new_char(glyph_face, ' ')
    };

    let mut event_count = 0;
    let mut pos = Position::new(0, 0);
    term.waker().wake()?;
    let mut layout_store = ViewLayoutStore::new();
    term.run_render(|term, event, mut surf| -> Result<_, Error> {
        event_count += 1;
        surf.draw_check_pattern("fg=gruv-1,bg=gruv-3".parse()?);

        match event {
            None => return Ok(TerminalAction::Wait),
            Some(event) if event == q || event == ctrlc => return Ok(TerminalAction::Quit(())),
            Some(event) => {
                // update mouse position
                match event {
                    TerminalEvent::Mouse(mouse) if mouse.name == KeyName::MouseMove => {
                        pos = mouse.pos;
                    }
                    _ => (),
                }

                let size = surf.size();
                surf.view_mut(1..-1, 1..-1).draw_view(
                    &ViewContext::new(term)?,
                    Some(&mut layout_store),
                    Container::new(stats_view(term, &event, event_count)?)
                        .with_vertical(if pos.row > size.height / 2 {
                            Align::Start
                        } else {
                            Align::End
                        })
                        .with_horizontal(if pos.col > size.width / 2 {
                            Align::Start
                        } else {
                            Align::End
                        })
                        .with_margins(Margins {
                            left: 1,
                            right: 1,
                            ..Default::default()
                        }),
                )?;
            }
        };

        if let Some(cell) = surf.get_mut(pos) {
            *cell = glyph.clone();
        }
        Ok(TerminalAction::Wait)
    })?;

    // switch off alt screen
    term.execute(TerminalCommand::altscreen_set(false))?;
    term.poll(Some(std::time::Duration::new(0, 0)))?;

    Ok(())
}
