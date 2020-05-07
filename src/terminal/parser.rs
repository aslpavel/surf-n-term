use super::{
    automata::{DFAState, DFA, NFA},
    Key, KeyName, TerminalError, TerminalEvent,
};
use std::{fmt, io::BufRead};

pub trait Decoder {
    type Item;
    fn decode(&mut self, input: &mut dyn BufRead) -> Result<Option<Self::Item>, TerminalError>;
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum TTYTag {
    Event(TerminalEvent),
}

impl fmt::Debug for TTYTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TTYTag::Event(event) => write!(f, "{:?}", event)?,
        }
        Ok(())
    }
}

impl From<TerminalEvent> for TTYTag {
    fn from(event: TerminalEvent) -> TTYTag {
        TTYTag::Event(event)
    }
}

#[derive(Debug)]
pub struct TTYDecoder {
    /// DFA that represents all possible states of the parser
    automata: DFA<TTYTag>,
    /// Current DFA state of the parser
    state: DFAState,
    /// Bytes consumed since the initialization of DFA
    buffer: Vec<u8>,
    /// Possible match was found, but has not failed yet, this decoder tries
    /// to find longest match as some escape sequnces are ambigious.
    possible: Option<TerminalEvent>,
}

impl Decoder for TTYDecoder {
    type Item = TerminalEvent;

    fn decode(&mut self, input: &mut dyn BufRead) -> Result<Option<Self::Item>, TerminalError> {
        let mut consumed = 0;
        for symbol in input.fill_buf()?.iter() {
            match self.automata.transition(self.state, *symbol) {
                Some(state) => {
                    let info = self.automata.info(state);
                    if info.accepting {
                        if let Some(tag) = info.tags.iter().next() {
                            match tag {
                                TTYTag::Event(event) => {
                                    self.possible.replace(event.clone());
                                }
                            }
                        }
                    }
                    self.state = state;
                }
                None => {
                    let buffer = std::mem::replace(&mut self.buffer, Vec::new());
                    let event = match self.possible.take() {
                        None => Some(TerminalEvent::Raw(buffer)),
                        event => event,
                    };
                    input.consume(consumed);
                    self.state = self.automata.start();
                    return Ok(event);
                }
            }
            consumed += 1;
            self.buffer.push(*symbol);
        }
        input.consume(consumed);
        Ok(None)
    }
}

impl TTYDecoder {
    pub fn new() -> Self {
        let mut cmds: Vec<NFA<TTYTag>> = Vec::new();

        type N = NFA<TTYTag>;
        let key = |string, key| N::from(string).tag(TerminalEvent::Key(Key::new(key)));

        // F{1-12}
        cmds.push(key("\x1bOP", KeyName::F1));
        cmds.push(key("\x1bOQ", KeyName::F2));
        cmds.push(key("\x1bOR", KeyName::F3));
        cmds.push(key("\x1bOS", KeyName::F4));
        cmds.push(key("\x1b[15~", KeyName::F5));
        cmds.push(key("\x1b[17~", KeyName::F6));
        cmds.push(key("\x1b[18~", KeyName::F7));
        cmds.push(key("\x1b[19~", KeyName::F8));
        cmds.push(key("\x1b[20~", KeyName::F9));
        cmds.push(key("\x1b[21~", KeyName::F10));
        cmds.push(key("\x1b[23~", KeyName::F11));
        cmds.push(key("\x1b[24~", KeyName::F12));

        let automata = N::choice(cmds).compile();
        let state = automata.start();
        Self {
            automata,
            state,
            buffer: Default::default(),
            possible: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Write};

    #[test]
    fn test_decoder() -> Result<(), TerminalError> {
        let mut cursor = Cursor::new(Vec::new());
        let mut decoder = TTYDecoder::new();

        // send incomplete sequence
        write!(cursor.get_mut(), "\x1b")?;

        assert_eq!(decoder.decode(&mut cursor)?, None);
        assert_eq!(cursor.position(), 1);

        // send rest of the sequence, plus full other sequence, and some garbage
        write!(cursor.get_mut(), "OR\x1b[15~AB")?;

        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::Key(Key::new(KeyName::F3)))
        );
        assert_eq!(cursor.position(), 3);

        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::Key(Key::new(KeyName::F5)))
        );
        assert_eq!(cursor.position(), 3 + 5);

        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::Raw(vec![65])),
        );

        Ok(())
    }
}
