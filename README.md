### Surf-N-Term
![Build Status](https://github.com/aslpavel/surf-n-term/actions/workflows/rust.yml/badge.svg)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Crate](https://img.shields.io/crates/v/surf-n-term.svg)](https://crates.io/crates/surf-n-term)
[![API Docs](https://docs.rs/surf-n-term/badge.svg)](https://docs.rs/surf-n-term)
This crate is used to interract with Posix terminal. It can be used to
- Read events from the terminal
- Send commands to the terminal
- Render on a surface which will be reconciled with current content of the terminal
- Issue direct commends to the terminal
- Supports kitty/sixel image protocol

### Simple example
```rust
use surf_n_term::{Terminal, TerminalEvent, Error};

fn main() -> Result<(), Error> {
    let ctrl_c = TerminalEvent::Key("ctrl+c".parse()?);
    let mut term = SystemTerminal::new()?;
    term.run_render(|term, event, mut view| -> Result<_, Error> {
        // This function will be executed on each event from terminal
        // - term  - implementes Terminal trait
        // - event - is a TerminalEvent
        // - view  - is a Suface that can be used to render on, see render module for defails
        match event {
            Some(event) if &event == &ctrl_c => {
                // exit if 'ctrl+c' is pressed
                Ok(TerminalAction::Quit(()))
            }
            _ => {
                // do some rendering by updating the view
                Ok(TerminalAction::Wait)
            },
        }
    })?;
    Ok(())
}
```
Full examples can be found in example submodule
```sh
$ cargo run --example mandelbrot
$ cargo run --example mouse
$ cargo run --example events
```

### Used by
- you should checkout my [sweep](https://github.com/aslpavel/sweep-rs) program to interactively filtter through list of items
