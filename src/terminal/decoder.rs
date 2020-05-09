use super::{
    automata::{DFAState, DFA, NFA},
    Decoder, Key, KeyMod, KeyName, Mouse, TerminalError, TerminalEvent, TerminalSize,
};
use std::{fmt, io::BufRead};

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

impl Default for TTYDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl TTYDecoder {
    pub fn new() -> Self {
        let automata = tty_decoder_dfa();
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
                    let event = tty_decoder_event(tag, &self.buffer);
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
                    let event = TerminalEvent::Raw(std::mem::take(&mut self.buffer));
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
}

fn tty_decoder_dfa() -> DFA<TTYTag> {
    let mut cmds: Vec<NFA<TTYTag>> = Vec::new();

    // construct NFA for basic key (no additional parsing is needed)
    fn basic_key(seq: &str, key: impl Into<Key>) -> NFA<TTYTag> {
        NFA::from(seq).tag(TerminalEvent::Key(key.into()))
    }

    // [Reference](http://www.leonerd.org.uk/hacks/fixterms/)
    // but it does not always match real behaviour

    cmds.push(basic_key("\x1b", KeyName::Esc));

    for byte in (0..=255u8).filter(|c| c.is_ascii_lowercase() || c.is_ascii_digit()) {
        let c = char::from(byte);
        cmds.push(basic_key(
            &format!("\x1b{}", c),
            (KeyName::Char(c), KeyMod::ALT),
        ));
        cmds.push(basic_key(
            &(char::from(byte & 0x1f)).to_string(),
            (KeyName::Char(c), KeyMod::CTRL),
        ));
    }

    for (name, code) in [
        (KeyName::Home, "1"),
        (KeyName::End, "4"),
        (KeyName::PageUp, "5"),
        (KeyName::PageDown, "6"),
        (KeyName::F5, "15"),
        (KeyName::F6, "17"),
        (KeyName::F7, "18"),
        (KeyName::F8, "19"),
        (KeyName::F9, "20"),
        (KeyName::F10, "21"),
        (KeyName::F11, "23"),
        (KeyName::F12, "24"),
    ]
    .iter()
    {
        cmds.push(basic_key(&format!("\x1b[{}~", code), *name));
        for mode in 1..8 {
            cmds.push(basic_key(
                &format!("\x1b[{};{}~", code, mode + 1),
                (*name, KeyMod::from_bits(mode)),
            ));
        }
    }
    for (name, code_empty, code) in [
        (KeyName::Up, "[", "A"),
        (KeyName::Down, "[", "B"),
        (KeyName::Right, "[", "C"),
        (KeyName::Left, "[", "D"),
        (KeyName::End, "[", "F"),
        (KeyName::Home, "[", "H"),
        (KeyName::F1, "O", "P"),
        (KeyName::F2, "O", "Q"),
        (KeyName::F3, "O", "R"),
        (KeyName::F4, "O", "S"),
    ]
    .iter()
    {
        cmds.push(basic_key(&format!("\x1b{}{}", code_empty, code), *name));
        for mode in 1..8 {
            cmds.push(basic_key(
                &format!("\x1b[1;{}{}", mode + 1, code),
                (*name, KeyMod::from_bits(mode)),
            ));
        }
    }

    // response to `CursorReport` ("\x1b[6n")
    cmds.push(
        NFA::sequence(vec![
            NFA::from("\x1b["),
            NFA::number(),
            NFA::from(";"),
            NFA::number(),
            NFA::from("R"),
        ])
        .tag(TTYTag::CursorPosition),
    );

    // size of the terminal in cells response to "\x1b[18t"
    cmds.push(
        NFA::sequence(vec![
            NFA::from("\x1b[8;"),
            NFA::number(),
            NFA::from(";"),
            NFA::number(),
            NFA::from("t"),
        ])
        .tag(TTYTag::TerminalSizeCells),
    );

    // size of the terminal in pixels response to "\x1b[14t"
    cmds.push(
        NFA::sequence(vec![
            NFA::from("\x1b[4;"),
            NFA::number(),
            NFA::from(";"),
            NFA::number(),
            NFA::from("t"),
        ])
        .tag(TTYTag::TerminalSizePixels),
    );

    // mouse events
    cmds.push(
        NFA::sequence(vec![
            NFA::from("\x1b[<"),
            NFA::number(),
            NFA::from(";"),
            NFA::number(),
            NFA::from(";"),
            NFA::number(),
            NFA::predicate(|b| b == b'm' || b == b'M'),
        ])
        .tag(TTYTag::MouseSGR),
    );

    // character event
    cmds.push(tty_decoder_char().tag(TTYTag::Char));

    NFA::choice(cmds).compile()
}

fn tty_decoder_char() -> NFA<TTYTag> {
    let printable = NFA::predicate(|b| b >= b' ' && b <= b'~');
    let utf8_two = NFA::predicate(|b| b >> 5 == 0b110);
    let utf8_three = NFA::predicate(|b| b >> 4 == 0b1110);
    let utf8_four = NFA::predicate(|b| b >> 3 == 0b11110);
    let utf8_tail = NFA::predicate(|b| b >> 6 == 0b10);
    NFA::choice(vec![
        printable,
        utf8_two + utf8_tail.clone(),
        utf8_three + utf8_tail.clone() + utf8_tail.clone(),
        utf8_four + utf8_tail.clone() + utf8_tail.clone() + utf8_tail,
    ])
}

/// Convert tag plus current match to a TerminalEvent
fn tty_decoder_event(tag: &TTYTag, data: &[u8]) -> TerminalEvent {
    use TTYTag::*;
    match tag {
        Event(event) => event.clone(),
        Char => {
            let string = std::str::from_utf8(data).expect("[TTYDecoder] utf8 expected");
            let code = string.chars().next().expect("[TTYDecoder] utf8 expected");
            TerminalEvent::Key(KeyName::Char(code).into())
        }
        CursorPosition => {
            // "\x1b[{row};{col}R"
            let mut nums = data[2..data.len() - 1].split(|b| *b == b';');
            let row = tty_number(&mut nums);
            let col = tty_number(&mut nums);
            TerminalEvent::CursorPosition { row, col }
        }
        TerminalSizeCells | TerminalSizePixels => {
            // "\x1b[(4|8);{height};{width}t"
            let mut nums = data[4..data.len() - 1].split(|b| *b == b';');
            let height = tty_number(&mut nums);
            let width = tty_number(&mut nums);
            if tag == &TerminalSizeCells {
                TerminalEvent::Size(TerminalSize {
                    height,
                    width,
                    width_pixels: 0,
                    height_pixels: 0,
                })
            } else {
                TerminalEvent::Size(TerminalSize {
                    height: 0,
                    width: 0,
                    width_pixels: width,
                    height_pixels: height,
                })
            }
        }
        MouseSGR => {
            // "\x1b[<{event};{row};{col}(m|M)"
            // https://invisible-island.net/xterm/ctlseqs/ctlseqs.html#h2-Mouse-Tracking
            let mut nums = data[3..data.len() - 1].split(|b| *b == b';');
            let event = tty_number(&mut nums);
            let row = tty_number(&mut nums);
            let col = tty_number(&mut nums);

            let mut mode = KeyMod::EMPTY;
            if event & 4 != 0 {
                mode = mode | KeyMod::SHIFT;
            }
            if event & 8 != 0 {
                mode = mode | KeyMod::ALT;
            }
            if event & 16 != 0 {
                mode = mode | KeyMod::CTRL;
            }
            if data[data.len() - 1] == b'M' {
                mode = mode | KeyMod::PRESS;
            }

            let button = event & 3;
            let name = if event & 64 != 0 {
                if button == 0 {
                    KeyName::MouseWheelDown
                } else if button == 1 {
                    KeyName::MouseWheelUp
                } else {
                    KeyName::MouseMove
                }
            } else if button == 0 {
                KeyName::MouseLeft
            } else if button == 1 {
                KeyName::MouseMiddle
            } else if button == 2 {
                KeyName::MouseRight
            } else {
                KeyName::MouseMove
            };

            TerminalEvent::Mouse(Mouse {
                name,
                mode,
                row,
                col,
            })
        }
    }
}

fn tty_number(nums: &mut dyn Iterator<Item = &[u8]>) -> usize {
    let data = nums.next().expect("[TTYDecoder] number expected");
    let mut result = 0usize;
    let mut mult = 1usize;
    for byte in data.iter().rev() {
        result += (byte - b'0') as usize * mult;
        mult *= 10;
    }
    result
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum TTYTag {
    Event(TerminalEvent),
    Char,
    CursorPosition,
    TerminalSizeCells,
    TerminalSizePixels,
    MouseSGR,
}

impl fmt::Debug for TTYTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use TTYTag::*;
        match self {
            Event(event) => write!(f, "{:?}", event)?,
            CursorPosition => write!(f, "CPR")?,
            TerminalSizeCells => write!(f, "TSC")?,
            TerminalSizePixels => write!(f, "TSP")?,
            MouseSGR => write!(f, "SGR")?,
            Char => write!(f, "CHR")?,
        }
        Ok(())
    }
}

impl From<TerminalEvent> for TTYTag {
    fn from(event: TerminalEvent) -> TTYTag {
        TTYTag::Event(event)
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
            Some(TerminalEvent::Key(KeyName::F3.into()))
        );
        assert_eq!(cursor.position(), 3);

        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::Key(KeyName::F5.into()))
        );
        assert_eq!(cursor.position(), 3 + 5);

        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::Key(KeyName::Char('A').into())),
        );
        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::Key(KeyName::Char('B').into())),
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
            Some(TerminalEvent::Key(KeyName::Esc.into()))
        );

        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::Key(KeyName::Char('O').into())),
        );
        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::Key(KeyName::Char('T').into())),
        );
        assert_eq!(decoder.decode(&mut cursor)?, None);

        Ok(())
    }

    #[test]
    fn test_cursor_position() -> Result<(), TerminalError> {
        let mut cursor = Cursor::new(Vec::new());
        let mut decoder = TTYDecoder::new();

        write!(cursor.get_mut(), "\x1b[97;15R")?;

        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::CursorPosition { row: 97, col: 15 }),
        );

        Ok(())
    }

    #[test]
    fn test_terminal_size() -> Result<(), TerminalError> {
        let mut cursor = Cursor::new(Vec::new());
        let mut decoder = TTYDecoder::new();

        write!(cursor.get_mut(), "\x1b[4;3104;1482t")?;
        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::Size(TerminalSize {
                width: 0,
                height: 0,
                width_pixels: 1482,
                height_pixels: 3104,
            })),
        );

        write!(cursor.get_mut(), "\x1b[8;101;202t")?;
        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::Size(TerminalSize {
                width: 202,
                height: 101,
                width_pixels: 0,
                height_pixels: 0,
            })),
        );

        Ok(())
    }

    #[test]
    fn test_mouse_sgr() -> Result<(), TerminalError> {
        let mut cursor = Cursor::new(Vec::new());
        let mut decoder = TTYDecoder::new();

        write!(cursor.get_mut(), "\x1b[<0;24;14M")?;
        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::Mouse(Mouse {
                name: KeyName::MouseLeft,
                mode: KeyMod::PRESS,
                row: 24,
                col: 14
            }))
        );

        write!(cursor.get_mut(), "\x1b[<26;33;26m")?;
        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::Mouse(Mouse {
                name: KeyName::MouseRight,
                mode: KeyMod::ALT | KeyMod::CTRL,
                row: 33,
                col: 26
            }))
        );

        write!(cursor.get_mut(), "\x1b[<65;142;30M")?;
        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::Mouse(Mouse {
                name: KeyName::MouseWheelUp,
                mode: KeyMod::PRESS,
                row: 142,
                col: 30,
            }))
        );

        Ok(())
    }

    #[test]
    fn test_char() -> Result<(), TerminalError> {
        let mut cursor = Cursor::new(Vec::new());
        let mut decoder = TTYDecoder::new();

        write!(cursor.get_mut(), "\u{1F431}")?;

        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::Key(KeyName::Char('🐱').into())),
        );

        Ok(())
    }
}
