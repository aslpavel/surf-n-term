use crate::{
    automata::{DFAState, DFA, NFA},
    error::Error,
    terminal::{DecModeStatus, Mouse, Size, TerminalColor, TerminalEvent, TerminalSize},
    Key, KeyMod, KeyName,
};
use lazy_static::lazy_static;
use std::{collections::BTreeMap, fmt, io::BufRead};

pub trait Decoder {
    type Item;
    type Error: From<std::io::Error>;

    fn decode<B: BufRead>(&mut self, buf: B) -> Result<Option<Self::Item>, Self::Error>;

    fn decode_into<B: BufRead>(
        &mut self,
        mut buf: B,
        out: &mut Vec<Self::Item>,
    ) -> Result<usize, Self::Error> {
        let mut count = 0;
        while let Some(item) = self.decode(&mut buf)? {
            out.push(item);
            count += 1;
        }
        Ok(count)
    }
}

lazy_static! {
    static ref UTF8DFA: DFA<()> = {
        let printable = NFA::predicate(|b| b >> 7 == 0b0);
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
        .compile()
    };
}

pub struct Utf8Decoder {
    state: DFAState,
    offset: usize,
    buffer: [u8; 4],
}

impl Default for Utf8Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Utf8Decoder {
    pub fn new() -> Self {
        Self {
            state: UTF8DFA.start(),
            offset: 0,
            buffer: [0; 4],
        }
    }

    fn consume(&mut self) -> char {
        let result = utf8_decode(&self.buffer[..self.offset]);
        self.reset();
        result
    }

    fn push(&mut self, byte: u8) {
        self.buffer[self.offset] = byte;
        self.offset += 1;
    }

    fn reset(&mut self) {
        self.state = UTF8DFA.start();
        self.offset = 0;
    }
}

impl Decoder for Utf8Decoder {
    type Item = char;
    type Error = std::io::Error;

    fn decode<B: BufRead>(&mut self, mut buf: B) -> Result<Option<Self::Item>, Self::Error> {
        let mut consume = 0;
        for byte in buf.fill_buf()?.iter() {
            consume += 1;
            match UTF8DFA.transition(self.state, *byte) {
                None => {
                    self.reset();
                    buf.consume(consume);
                    use std::io::{Error, ErrorKind};
                    return Err(Error::new(ErrorKind::InvalidInput, "utf8 decoder failed"));
                }
                Some(state) if UTF8DFA.info(state).accepting => {
                    self.push(*byte);
                    buf.consume(consume);
                    return Ok(Some(self.consume()));
                }
                Some(state) => {
                    self.push(*byte);
                    self.state = state;
                }
            }
        }
        buf.consume(consume);
        Ok(None)
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
    type Error = Error;

    fn decode<B: BufRead>(&mut self, mut input: B) -> Result<Option<Self::Item>, Self::Error> {
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
                    let event = tty_decoder_event(tag, &self.buffer)
                        .unwrap_or_else(|| TerminalEvent::Raw(self.buffer.clone()));
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
    cmds.push(basic_key("\x7f", KeyName::Backspace));

    // ascii keys with modifiers
    for byte in (0..=255u8).filter(|c| c.is_ascii_lowercase()) {
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

    for byte in (0..=255u8).filter(|c| c.is_ascii_digit()) {
        let c = char::from(byte);
        cmds.push(basic_key(
            &format!("\x1b{}", c),
            (KeyName::Char(c), KeyMod::ALT),
        ));
    }

    for (name, code) in [
        (KeyName::Home, "1"),
        (KeyName::Delete, "3"),
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

    // DEC mode report
    cmds.push(
        NFA::sequence(vec![
            NFA::from("\x1b[?"),
            NFA::number(),
            NFA::from(";"),
            NFA::number(),
            NFA::from("$y"),
        ])
        .tag(TTYTag::DecMode),
    );

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

    // character event (mostly utf8 but one-byte codes are restricted to the printable set)
    let char_nfa = {
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
    };
    cmds.push(char_nfa.tag(TTYTag::Char));

    // kitty image response "\x1b_Gkey=value(,key=value)*;response\x1b\\"
    let kitty_image_response = {
        let key_value = NFA::sequence(vec![
            NFA::predicate(|b| b.is_ascii_alphanumeric()).some(),
            NFA::from("="),
            NFA::predicate(|b| b.is_ascii_alphanumeric()).some(),
        ]);
        NFA::sequence(vec![
            NFA::from("\x1b_G"),
            key_value.clone(),
            NFA::sequence(vec![NFA::from(","), key_value]).many(),
            NFA::from(";"),
            NFA::predicate(|b| b != b'\x1b').many(),
            NFA::from("\x1b\\"),
        ])
    };
    cmds.push(kitty_image_response.tag(TTYTag::KittyImage));

    // Termcap/Terminfo response to XTGETTCAP
    let terminfo_response = {
        let hex = NFA::predicate(|b| matches!(b, b'A'..=b'F' | b'a'..=b'f' | b'0'..=b'9'));
        let hex = hex.clone() + hex;
        let key_value = NFA::sequence(vec![hex.clone().some(), NFA::from("="), hex.clone().some()]);
        NFA::choice(vec![
            // success
            NFA::sequence(vec![
                NFA::from("\x1bP1+r"),
                NFA::sequence(vec![
                    key_value.clone(),
                    NFA::sequence(vec![NFA::from(";"), key_value]).many(),
                ])
                .optional(),
            ]),
            // failure
            NFA::sequence(vec![
                NFA::from("\x1bP0+r"),
                NFA::sequence(vec![
                    hex.clone().some(),
                    NFA::sequence(vec![NFA::from(";"), hex.some()]).many(),
                ])
                .optional(),
            ]),
        ]) + NFA::from("\x1b\\")
    };
    cmds.push(terminfo_response.tag(TTYTag::Terminfo));

    // DA1 Deivece attributes https://vt100.net/docs/vt510-rm/DA1.html
    // "\x1b[?<attr_1>;...<attr_n>c"
    cmds.push(
        NFA::sequence(vec![
            NFA::from("\x1b[?"),
            (NFA::number() + NFA::from(";").optional()).some(),
            NFA::from("c"),
        ])
        .tag(TTYTag::DeviceAttrs),
    );

    // OSC response
    // "\x1b]<number>;.*\x1b\\"
    cmds.push(
        NFA::sequence(vec![
            NFA::from("\x1b]"),
            NFA::number(),
            NFA::from(";"),
            NFA::predicate(|c| c != b'\x1b' && c != b'\x07').some(),
            (NFA::from("\x1b\\") | NFA::from("\x07")),
        ])
        .tag(TTYTag::OperatingSystemControl),
    );

    NFA::choice(cmds).compile()
}

/// Convert tag plus current match to a TerminalEvent
fn tty_decoder_event(tag: &TTYTag, data: &[u8]) -> Option<TerminalEvent> {
    use TTYTag::*;
    let event = match tag {
        Event(event) => event.clone(),
        Char => TerminalEvent::Key(KeyName::Char(utf8_decode(data)).into()),
        DecMode => {
            // "\x1b[?{mode};{status}$y"
            let mut nums = numbers_decode(&data[3..data.len() - 2]);
            TerminalEvent::DecMode {
                mode: crate::terminal::DecMode::from_usize(nums.next()?)?,
                status: DecModeStatus::from_usize(nums.next()?)?,
            }
        }
        CursorPosition => {
            // "\x1b[{row};{col}R"
            let mut nums = numbers_decode(&data[2..data.len() - 1]);
            TerminalEvent::CursorPosition {
                row: nums.next()? - 1,
                col: nums.next()? - 1,
            }
        }
        TerminalSizeCells | TerminalSizePixels => {
            // "\x1b[(4|8);{height};{width}t"
            let mut nums = numbers_decode(&data[4..data.len() - 1]);
            let height = nums.next()?;
            let width = nums.next()?;
            if tag == &TerminalSizeCells {
                TerminalEvent::Size(TerminalSize {
                    cells: Size { height, width },
                    pixels: Size {
                        height: 0,
                        width: 0,
                    },
                })
            } else {
                TerminalEvent::Size(TerminalSize {
                    cells: Size {
                        height: 0,
                        width: 0,
                    },
                    pixels: Size { width, height },
                })
            }
        }
        MouseSGR => {
            // "\x1b[<{event};{row};{col}(m|M)"
            // https://invisible-island.net/xterm/ctlseqs/ctlseqs.html#h2-Mouse-Tracking
            let mut nums = numbers_decode(&data[3..data.len() - 1]);
            let event = nums.next()?;
            let col = nums.next()? - 1;
            let row = nums.next()? - 1;

            let mut mode = KeyMod::from_bits(((event >> 2) & 7) as u8);
            if data[data.len() - 1] == b'M' {
                mode |= KeyMod::PRESS;
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
        KittyImage => {
            // "\x1b_Gkey=value(,key=value)*;response\x1b\\"
            let mut iter = (&data[3..data.len() - 2]).splitn(2, |b| *b == b';');
            let mut id = 0; // id can not be zero according to the spec
            for (key, value) in key_value_decode(b',', iter.next()?) {
                if key == b"i" {
                    id = number_decode(value)? as u64;
                }
            }
            let msg = iter.next()?;
            let error = if msg == b"OK" {
                None
            } else {
                Some(String::from_utf8_lossy(msg).to_string())
            };
            TerminalEvent::KittyImage { id, error }
        }
        Terminfo => {
            // "\x1bP(0|1)+rkey=value(;key=value)\x1b\\"
            let mut termcap = BTreeMap::new();
            if data[2] == b'1' {
                for (key, value) in key_value_decode(b';', &data[5..data.len() - 2]) {
                    termcap.insert(
                        hex_decode(key).map(char::from).collect(),
                        Some(hex_decode(value).map(char::from).collect()),
                    );
                }
            } else {
                for key in data[5..data.len() - 2].split(|b| *b == b';') {
                    termcap.insert(hex_decode(key).map(char::from).collect(), None);
                }
            }
            TerminalEvent::Termcap(termcap)
        }
        DeviceAttrs => {
            // "\x1b[?<attr_1>;...<attr_n>c"
            TerminalEvent::DeviceAttrs(
                numbers_decode(&data[3..data.len() - 1])
                    .filter(|v| v > &0)
                    .collect(),
            )
        }
        OperatingSystemControl => {
            // "\x1b]<number>;.*(\x1b\\|\x07)"
            let data = if data[data.len() - 1] == b'\x07' {
                &data[2..data.len() - 1]
            } else {
                &data[2..data.len() - 2]
            };
            let mut args = data.split(|c| *c == b';');
            let id = number_decode(args.next()?)?;
            let name = match id {
                10 => TerminalColor::Foreground,
                11 => TerminalColor::Background,
                4 => TerminalColor::Palette(number_decode(args.next()?)?),
                _ => return None,
            };
            let color = std::str::from_utf8(args.next()?).ok()?.parse().ok()?;
            TerminalEvent::Color { name, color }
        }
    };
    Some(event)
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum TTYTag {
    Event(TerminalEvent),
    Char,
    CursorPosition,
    TerminalSizeCells,
    TerminalSizePixels,
    MouseSGR,
    DecMode,
    KittyImage,
    Terminfo,
    DeviceAttrs,
    OperatingSystemControl,
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
            DecMode => write!(f, "DCM")?,
            KittyImage => write!(f, "KI")?,
            Terminfo => write!(f, "CAP")?,
            DeviceAttrs => write!(f, "DA1")?,
            OperatingSystemControl => write!(f, "OSC")?,
        }
        Ok(())
    }
}

impl From<TerminalEvent> for TTYTag {
    fn from(event: TerminalEvent) -> TTYTag {
        TTYTag::Event(event)
    }
}

/// key=value(,key=value)*
fn key_value_decode(sep: u8, data: &[u8]) -> impl Iterator<Item = (&[u8], &[u8])> + '_ {
    data.split(move |b| *b == sep).filter_map(|kv| {
        let mut iter = kv.splitn(2, |b| *b == b'=');
        let key = iter.next()?;
        let value = iter.next()?;
        Some((key, value))
    })
}

/// Semi-colon separated positve numers
fn numbers_decode(data: &[u8]) -> impl Iterator<Item = usize> + '_ {
    data.split(|b| *b == b';').filter_map(number_decode)
}

// Decode positive integer number
fn number_decode(data: &[u8]) -> Option<usize> {
    let mut result = 0usize;
    let mut mult = 1usize;
    for b in data.iter().rev() {
        match b {
            b'0'..=b'9' => {
                result += (b - b'0') as usize * mult;
                mult *= 10;
            }
            _ => return None,
        }
    }
    Some(result)
}

// Convert slice to a character
//
// NOTE: this function must only be used on a validated buffer
// containing single UTF8 character.
fn utf8_decode(slice: &[u8]) -> char {
    let first = slice[0] as u32;
    let mut code: u32 = match slice.len() {
        1 => first & 127,
        2 => first & 31,
        3 => first & 15,
        4 => first & 7,
        _ => panic!("[utf8_deocde] invalid code point slice"),
    };
    for byte in slice[1..].iter() {
        code <<= 6;
        code |= (*byte as u32) & 63;
    }
    unsafe { std::char::from_u32_unchecked(code) }
}

pub fn hex_decode(slice: &[u8]) -> impl Iterator<Item = u8> + '_ {
    let value = |byte| match byte {
        b'A'..=b'F' => Some(byte - b'A' + 10),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'0'..=b'9' => Some(byte - b'0'),
        _ => None,
    };
    slice
        .chunks(2)
        .map(move |pair| Some(value(pair[0])? << 4 | value(pair[1])?))
        .take_while(|value| value.is_some())
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Write};

    #[test]
    fn test_basic() -> Result<(), Error> {
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
    fn test_reschedule() -> Result<(), Error> {
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
    fn test_cursor_position() -> Result<(), Error> {
        let mut cursor = Cursor::new(Vec::new());
        let mut decoder = TTYDecoder::new();

        write!(cursor.get_mut(), "\x1b[97;15R")?;

        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::CursorPosition { row: 96, col: 14 }),
        );

        Ok(())
    }

    #[test]
    fn test_terminal_size() -> Result<(), Error> {
        let mut cursor = Cursor::new(Vec::new());
        let mut decoder = TTYDecoder::new();

        write!(cursor.get_mut(), "\x1b[4;3104;1482t")?;
        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::Size(TerminalSize {
                cells: Size {
                    width: 0,
                    height: 0,
                },
                pixels: Size {
                    width: 1482,
                    height: 3104,
                }
            })),
        );

        write!(cursor.get_mut(), "\x1b[8;101;202t")?;
        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::Size(TerminalSize {
                cells: Size {
                    width: 202,
                    height: 101,
                },
                pixels: Size {
                    width: 0,
                    height: 0,
                }
            })),
        );

        Ok(())
    }

    #[test]
    fn test_mouse_sgr() -> Result<(), Error> {
        let mut cursor = Cursor::new(Vec::new());
        let mut decoder = TTYDecoder::new();

        write!(cursor.get_mut(), "\x1b[<0;94;14M")?;
        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::Mouse(Mouse {
                name: KeyName::MouseLeft,
                mode: KeyMod::PRESS,
                row: 13,
                col: 93
            }))
        );

        write!(cursor.get_mut(), "\x1b[<26;33;26m")?;
        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::Mouse(Mouse {
                name: KeyName::MouseRight,
                mode: KeyMod::ALT | KeyMod::CTRL,
                row: 25,
                col: 32
            }))
        );

        write!(cursor.get_mut(), "\x1b[<65;142;30M")?;
        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::Mouse(Mouse {
                name: KeyName::MouseWheelUp,
                mode: KeyMod::PRESS,
                row: 29,
                col: 141,
            }))
        );

        Ok(())
    }

    #[test]
    fn test_char() -> Result<(), Error> {
        let mut cursor = Cursor::new(Vec::new());
        let mut decoder = TTYDecoder::new();

        write!(cursor.get_mut(), "\u{1F431}")?;

        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::Key(KeyName::Char('🐱').into())),
        );

        Ok(())
    }

    #[test]
    fn test_dec_mode() -> Result<(), Error> {
        use crate::DecMode;

        let mut cursor = Cursor::new(Vec::new());
        let mut decoder = TTYDecoder::new();

        write!(cursor.get_mut(), "\x1b[?1000;1$y")?;

        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::DecMode {
                mode: DecMode::MouseReport,
                status: DecModeStatus::Enabled,
            }),
        );

        write!(cursor.get_mut(), "\x1b[?2017;0$y")?;

        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::DecMode {
                mode: DecMode::KittyKeyboard,
                status: DecModeStatus::NotRecognized,
            }),
        );

        Ok(())
    }

    #[test]
    fn test_utf8_decoder() -> Result<(), Error> {
        let mut cursor = Cursor::new(Vec::new());
        let mut decoder = Utf8Decoder::new();
        assert_eq!(decoder.decode(&mut cursor)?, None);

        // one byte char
        write!(cursor.get_mut(), "!")?;
        assert_eq!(decoder.decode(&mut cursor)?, Some('!'));

        // two byte char
        write!(cursor.get_mut(), "¢")?;
        assert_eq!(decoder.decode(&mut cursor)?, Some('¢'));

        // three byte char
        write!(cursor.get_mut(), "€")?;
        assert_eq!(decoder.decode(&mut cursor)?, Some('€'));

        // four byte char
        write!(cursor.get_mut(), "𐍈")?;
        assert_eq!(decoder.decode(&mut cursor)?, Some('𐍈'));

        // partial
        let c = b"\xd1\x8f"; // я
        cursor.get_mut().write(&c[..1])?;
        assert_eq!(decoder.decode(&mut cursor)?, None);
        cursor.get_mut().write(&c[1..])?;
        assert_eq!(decoder.decode(&mut cursor)?, Some('я'));

        // invalid
        cursor.get_mut().write(&c[..1])?;
        assert_eq!(decoder.decode(&mut cursor)?, None);
        cursor.get_mut().write(&c[..1])?;
        assert!(decoder.decode(&mut cursor).is_err());

        // valid after invalid
        write!(cursor.get_mut(), "𐍈€")?;
        assert_eq!(decoder.decode(&mut cursor)?, Some('𐍈'));
        assert_eq!(decoder.decode(&mut cursor)?, Some('€'));

        Ok(())
    }

    #[test]
    fn test_hex_decode() {
        let mut iter = hex_decode(b"d3869B");
        assert_eq!(iter.next(), Some(211));
        assert_eq!(iter.next(), Some(134));
        assert_eq!(iter.next(), Some(155));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_kitty() -> Result<(), Error> {
        let mut cursor = Cursor::new(Vec::new());
        let mut decoder = TTYDecoder::new();

        write!(cursor.get_mut(), "\x1b_Gi=127;OK\x1b\\")?;
        write!(
            cursor.get_mut(),
            "\x1b_Gi=31,ignored=attr;error message\x1b\\"
        )?;

        let mut result = Vec::new();
        decoder.decode_into(&mut cursor, &mut result)?;
        assert_eq!(
            result,
            vec![
                TerminalEvent::KittyImage {
                    id: 127,
                    error: None
                },
                TerminalEvent::KittyImage {
                    id: 31,
                    error: Some("error message".to_string())
                },
            ]
        );

        Ok(())
    }

    #[test]
    fn test_terminfo() -> Result<(), Error> {
        let mut cursor = Cursor::new(Vec::new());
        let mut decoder = TTYDecoder::new();

        write!(
            cursor.get_mut(),
            "\x1bP1+r62656c=5e47;626f6c64=1b5b316d\x1b\\"
        )?;
        write!(
            cursor.get_mut(),
            "\x1bP1+r736d637570=1b5b3f3130343968\x1b\\",
        )?;
        write!(cursor.get_mut(), "\x1bP0+r73757266;7465726d\x1b\\")?;

        let mut result = Vec::new();
        decoder.decode_into(&mut cursor, &mut result)?;
        assert_eq!(
            result,
            vec![
                TerminalEvent::Termcap(
                    vec![("bel", "^G"), ("bold", "\u{1b}[1m")]
                        .into_iter()
                        .map(|(k, v)| (k.to_string(), Some(v.to_string())))
                        .collect()
                ),
                TerminalEvent::Termcap(
                    Some(("smcup", "\u{1b}[?1049h"))
                        .into_iter()
                        .map(|(k, v)| (k.to_string(), Some(v.to_string())))
                        .collect()
                ),
                TerminalEvent::Termcap(
                    vec!["surf", "term"]
                        .into_iter()
                        .map(|k| (k.to_string(), None))
                        .collect()
                ),
            ]
        );

        Ok(())
    }

    #[test]
    fn test_da1() -> Result<(), Error> {
        let mut cursor = Cursor::new(Vec::new());
        let mut decoder = TTYDecoder::new();

        write!(cursor.get_mut(), "\x1b[?62;c\x1b[?64;4c")?;

        let mut result = Vec::new();
        decoder.decode_into(&mut cursor, &mut result)?;
        assert_eq!(
            result,
            vec![
                TerminalEvent::DeviceAttrs(Some(62).into_iter().collect()),
                TerminalEvent::DeviceAttrs(vec![64, 4].into_iter().collect()),
            ]
        );

        Ok(())
    }

    #[test]
    fn test_osc() -> Result<(), Error> {
        let mut cursor = Cursor::new(Vec::new());
        let mut decoder = TTYDecoder::new();

        write!(
            cursor.get_mut(),
            "\x1b]4;1;rgb:cc/24/1d\x1b\\\x1b]10;#ebdbb2\x07"
        )?;

        let mut result = Vec::new();
        decoder.decode_into(&mut cursor, &mut result)?;
        assert_eq!(
            result,
            vec![
                TerminalEvent::Color {
                    name: TerminalColor::Palette(1),
                    color: "#cc241d".parse()?,
                },
                TerminalEvent::Color {
                    name: TerminalColor::Foreground,
                    color: "#ebdbb2".parse()?,
                }
            ]
        );

        Ok(())
    }
}
