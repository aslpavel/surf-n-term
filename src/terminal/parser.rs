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
    /// Rescheduled data, that needs to be parsed again in **reversed order**
    rescheduled: Vec<u8>,
    /// Found non-terminal match with its size in the buffer
    possible: Option<(TerminalEvent, usize)>,
}

impl Decoder for TTYDecoder {
    type Item = TerminalEvent;

    fn decode(&mut self, input: &mut dyn BufRead) -> Result<Option<Self::Item>, TerminalError> {
        // process rescheduled data first
        while let Some(byte) = self.rescheduled.pop() {
            let event = self.decode_byte(byte);
            if event.is_some() {
                return Ok(event);
            }
        }

        // process input
        let mut consumed = 0;
        let mut output = None;
        for byte in input.fill_buf()?.iter() {
            consumed += 1;
            if let Some(event) = self.decode_byte(*byte) {
                output.replace(event);
                break;
            }
        }
        input.consume(consumed);
        Ok(output)
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

        cmds.push(key("\x1b", KeyName::Esc));

        let automata = N::choice(cmds).compile();
        let state = automata.start();
        Self {
            automata,
            state,
            rescheduled: Default::default(),
            buffer: Default::default(),
            possible: None,
        }
    }

    /// Process single byte
    fn decode_byte(&mut self, byte: u8) -> Option<TerminalEvent> {
        match self.automata.transition(self.state, byte) {
            Some(state) => {
                // put byte into a buffer since automata resulted in a valid state
                self.buffer.push(byte);
                self.state = state;

                let info = self.automata.info(state);
                if info.accepting {
                    let tag = info
                        .tags
                        .iter()
                        .next()
                        .expect("[TTYDecoder] found untagged accepting state");
                    let event = self.event_from_tag(tag);
                    if info.terminal {
                        // report event
                        self.reset();
                        return Some(event);
                    } else {
                        // stash event (there might be a longer chain to be accepted)
                        self.possible.replace((event, self.buffer.len()));
                    }
                }
                None
            }
            None => match self.possible.take() {
                None => {
                    if self.buffer.is_empty() {
                        self.buffer.push(byte);
                    } else {
                        self.rescheduled.push(byte);
                    }
                    let event =
                        TerminalEvent::Raw(std::mem::replace(&mut self.buffer, Default::default()));
                    self.reset();
                    Some(event)
                }
                Some((event, size)) => {
                    // report possible event and reschedule reminder to be parsed again
                    self.rescheduled.push(byte);
                    self.rescheduled.extend(self.buffer.drain(size..).rev());
                    self.reset();
                    Some(event)
                }
            },
        }
    }

    /// Reset automata and buffer, but keep rescheduled data
    fn reset(&mut self) {
        self.possible.take();
        self.buffer.clear();
        self.state = self.automata.start();
    }

    /// Convert tag to a TerminalEvent
    ///
    /// Buffer must contain currently matched sequence.
    fn event_from_tag(&self, tag: &TTYTag) -> TerminalEvent {
        match tag {
            TTYTag::Event(event) => event.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Write};

    #[test]
    fn test_basic() -> Result<(), TerminalError> {
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
            Some(TerminalEvent::Raw(vec![b'A'])),
        );
        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::Raw(vec![b'B'])),
        );
        assert_eq!(decoder.decode(&mut cursor)?, None);

        Ok(())
    }

    #[test]
    fn test_reschedule() -> Result<(), TerminalError> {
        let mut cursor = Cursor::new(Vec::new());
        let mut decoder = TTYDecoder::new();

        // write possible match sequnce
        write!(cursor.get_mut(), "\x1bO")?;
        assert_eq!(decoder.decode(&mut cursor)?, None);
        assert_eq!(cursor.position(), 2);

        // fail to rollback to possible match
        write!(cursor.get_mut(), "T")?;

        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::Key(Key::new(KeyName::Esc)))
        );

        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::Raw(vec![b'O'])),
        );
        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::Raw(vec![b'T'])),
        );
        assert_eq!(decoder.decode(&mut cursor)?, None);

        Ok(())
    }
}
