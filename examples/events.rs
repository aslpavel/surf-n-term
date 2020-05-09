use std::{boxed::Box, error::Error, time::Duration};
use surf_n_term::{Key, KeyName, SystemTerminal, Terminal, TerminalEvent};

fn main() -> Result<(), Box<dyn Error>> {
    let q = Key::from(KeyName::Char('q'));

    let timeout = Duration::from_secs(10);
    println!("Program will exit after {:?} of idling ...", timeout);

    let mut term = SystemTerminal::new()?;
    // term.execute(TerminalCommand::MouseReport {
    //     enable: true,
    //     motion: true,
    // })?;
    while let Some(event) = term.poll(Some(timeout))? {
        match event {
            TerminalEvent::Key(key) if key == q => break,
            _ => println!("{:?}", event),
        }
    }
    // term.execute(TerminalCommand::MouseReport {
    //     enable: false,
    // motion: false,
    // })?;
    // term.poll(Some(Duration::new(0, 0)))?;
    Ok(())
}
