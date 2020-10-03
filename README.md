### Surf-N-Term
This crate is used to interract with Posix terminal. It can be used to
- Read events from the terminal
- Send commands to the terminal
- Render on a surface which will be reconciled with current content of the terminal
- Issue direct commends to the terminal
- Supports kitty/sixel image protocol

### Simple example
```
use surf_n_term::{Terminal, Error};

fn main() -> Result<(), Error> {
    let mut term = SystemTerminal::new()?;
    term.run_render(|term, event, mut view| -> Result<_, Error> {
        // This function will be executed on each event from terminal
        // - term  - implementes Terminal trait
        // - event - is a TerminalEvent
        // - view  - is a Suface that can be used to render on, see render module for defails
        ...
    })?;
}
```
Full examples can be found in example submodule
```
$ cargo run --example mandelbrot
$ cargo run --example mouse
$ cargo run --example events
```

### Used by
- you should checkout my [sweep](https://github.com/aslpavel/sweep-rs) program to interactively filtter through list of items
