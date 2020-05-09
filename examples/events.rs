use std::{boxed::Box, error::Error, time::Duration};
use surf_n_term::{
    DecMode, Key, KeyName, SystemTerminal, Terminal, TerminalCommand, TerminalEvent,
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut term = SystemTerminal::new()?;

    // query DEC modes
    use TerminalCommand::*;
    term.execute(DecModeReport(DecMode::MouseReport))?;

    // read terminal events
    let timeout = Duration::from_secs(10);
    println!("Program will exit after {:?} of idling ...", timeout);
    let q = Key::from(KeyName::Char('q'));
    while let Some(event) = term.poll(Some(timeout))? {
        match event {
            TerminalEvent::Key(key) if key == q => break,
            _ => println!("{:?}", event),
        }
    }

    Ok(())
}
