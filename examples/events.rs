use std::{boxed::Box, error::Error};
use tty_surface::{SystemTerminal, Terminal};

fn main() -> Result<(), Box<dyn Error>> {
    let mut term = SystemTerminal::new()?;
    while let Some(event) = term.poll(None)? {
        println!("{:?}", event);
    }
    Ok(())
}
