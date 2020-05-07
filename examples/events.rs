use std::{boxed::Box, error::Error, time::Duration};
use tty_surface::{Key, KeyName, SystemTerminal, Terminal, TerminalEvent};

fn main() -> Result<(), Box<dyn Error>> {
    let esc = Key::from(KeyName::Esc);

    let timeout = Duration::from_secs(10);
    println!("Program will exit after {:?} of idling ...", timeout);

    let mut term = SystemTerminal::new()?;
    while let Some(event) = term.poll(Some(timeout))? {
        match event {
            TerminalEvent::Key(key) if key == esc => break,
            _ => println!("{:?}", event),
        }
    }
    Ok(())
}
