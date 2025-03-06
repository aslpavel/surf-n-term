//! Decoders
use crate::{
    Face, FaceModify, Key, KeyMod, KeyName, Position, RGBA, TerminalCommand, UnderlineStyle,
    automata::{DFA, DFAState, NFA},
    error::Error,
    terminal::{DecModeStatus, Mouse, Size, TerminalColor, TerminalEvent, TerminalSize},
};
use either::Either;
use smallvec::SmallVec;
use std::{
    collections::BTreeMap,
    fmt,
    io::{BufRead, Read},
    ops::Deref,
    sync::{Arc, LazyLock},
};

/// Decoder interface
pub trait Decoder {
    type Item;
    type Error: From<std::io::Error>;

    /// Decode single item from provided buffer
    fn decode<B: BufRead>(&mut self, buf: B) -> Result<Option<Self::Item>, Self::Error>;

    /// Decode all available items from provided buffer and put them into output vector
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

static UTF8DFA: LazyLock<DFA<()>> = LazyLock::new(|| utf8_nfa(UTF8Mode::Canonical).compile());
static TTY_EVENT_AUTOMATA: LazyLock<MatcherAutomata<TerminalEvent>> = LazyLock::new(|| {
    MatcherAutomata::new([
        Box::new(BasicEventsMatcher) as Box<dyn Matcher<Item = TerminalEvent>>,
        Box::new(CursorPositionMatcher),
        Box::new(DecModeMatcher),
        Box::new(DeviceAttrsMatcher),
        Box::new(
            GraphicRenditionMatcher
                .map(|face| TerminalEvent::Command(TerminalCommand::FaceModify(face))),
        ),
        Box::new(KittyImageMatcher),
        Box::new(KittyKeyboardMatcher),
        Box::new(MouseEventMatcher),
        Box::new(OSControlMatcher),
        Box::new(ReportSettingMatcher),
        Box::new(TermCapMatcher),
        Box::new(TermSizeMatcher),
        Box::new(
            UTF8Matcher::new(UTF8Mode::Printable)
                .map(|c| TerminalEvent::Key(KeyName::Char(c).into())),
        ),
        Box::new(BracketedPasteMatcher),
    ])
});
static TTY_COMMAND_AUTOMATA: LazyLock<MatcherAutomata<TerminalCommand>> = LazyLock::new(|| {
    MatcherAutomata::new([
        Box::new(GraphicRenditionMatcher.map(TerminalCommand::FaceModify))
            as Box<dyn Matcher<Item = TerminalCommand>>,
        Box::new(UTF8Matcher::new(UTF8Mode::NotEscape).map(TerminalCommand::Char)),
    ])
});

/// UTF-8 decoder
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
                    use std::io::{Error, ErrorKind};
                    self.reset();
                    buf.consume(consume);
                    return Err(Error::new(ErrorKind::InvalidInput, "utf8 decoder failed"));
                }
                Some(state) if UTF8DFA.info(state).is_accepting => {
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
struct MatcherAutomataInner<T> {
    /// DFA that represents all possible states of the parser
    pub(crate) automata: DFA<MatcherTag<T>>,
    /// Matchers registered with TTYMatch::Index tag
    pub(crate) matchers: Vec<Box<dyn Matcher<Item = T>>>,
}

/// Compiled tty matcher automata
#[derive(Clone, Debug)]
struct MatcherAutomata<T> {
    inner: Arc<MatcherAutomataInner<T>>,
}

impl<T: Clone + Ord> MatcherAutomata<T> {
    fn new(matchers: impl IntoIterator<Item = Box<dyn Matcher<Item = T>>>) -> Self {
        let matchers: Vec<_> = matchers.into_iter().collect();
        let automata = NFA::choice(matchers.iter().enumerate().map(|(index, matcher)| {
            match matcher.matcher() {
                Either::Left(automata) => {
                    automata
                        // this call only here to convert type as [Void] cannot be created
                        .tags_map(|_| MatcherTag::Matcher(index))
                        .tag_stop_state(MatcherTag::Matcher(index))
                }
                Either::Right(automata) => automata.tags_map(MatcherTag::Item),
            }
        }))
        .compile();
        let inner = MatcherAutomataInner { automata, matchers };
        Self {
            inner: Arc::new(inner),
        }
    }
}

impl<T> Deref for MatcherAutomata<T> {
    type Target = MatcherAutomataInner<T>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

type MatcherBuffer = SmallVec<[u8; 32]>;

/// TTY Decoder
#[derive(Debug)]
struct MatcherDecoder<T> {
    automata: MatcherAutomata<T>,
    /// Current DFA state of the parser
    automata_state: DFAState,
    /// Bytes consumed since the initialization of DFA
    buffer: MatcherBuffer,
    /// Rescheduled data, that needs to be parsed again in **reversed order**
    /// This data is used when possible match was found but longer one failed
    /// to materialize, hence we need to resubmit data consumed after first match.
    rescheduled: MatcherBuffer,
    /// Possible match is filled when we have automata in the accepting state
    /// but it is not terminal (transition to other state is possible). Contains
    /// item and amount of data in the buffer when this item was found.
    item_candidate: Option<(Result<T, MatcherBuffer>, usize)>,
}

impl<T: Clone + Ord> Decoder for MatcherDecoder<T> {
    type Item = Result<T, MatcherBuffer>;
    type Error = Error;

    fn decode<B: BufRead>(&mut self, mut input: B) -> Result<Option<Self::Item>, Self::Error> {
        // process rescheduled data first
        while let Some(byte) = self.rescheduled.pop() {
            let item = self.decode_byte(byte);
            if item.is_some() {
                return Ok(item);
            }
        }

        // process input
        let mut consumed = 0;
        let mut output = None;
        for byte in input.fill_buf()?.iter() {
            consumed += 1;
            if let Some(item) = self.decode_byte(*byte) {
                output.replace(item);
                break;
            }
        }
        input.consume(consumed);

        Ok(output)
    }
}

impl<T: Clone + Ord> MatcherDecoder<T> {
    fn new(automata: MatcherAutomata<T>) -> Self {
        let state = automata.automata.start();
        Self {
            automata,
            automata_state: state,
            rescheduled: Default::default(),
            buffer: Default::default(),
            item_candidate: None,
        }
    }

    /// Process single byte
    fn decode_byte(&mut self, byte: u8) -> Option<Result<T, MatcherBuffer>> {
        self.buffer.push(byte);
        match self.automata.automata.transition(self.automata_state, byte) {
            Some(state) => {
                self.automata_state = state;
                let state_desc = self.automata.automata.info(state);
                if state_desc.is_accepting {
                    let tag = state_desc
                        .tags
                        .iter()
                        .next()
                        .expect("[MatcherDecoder] found untagged accepting state");

                    // decode events
                    let event = match tag {
                        MatcherTag::Item(event) => Ok(event.clone()),
                        MatcherTag::Matcher(index) => self.automata.matchers[*index]
                            .decode(&self.buffer)
                            .map_or_else(|| Err(self.buffer.clone()), |item| Ok(item)),
                    };

                    self.item_candidate.replace((event, self.buffer.len()));
                    if state_desc.is_terminal {
                        return self.take_candidate();
                    }
                }
                None
            }
            None => {
                let event = self.take_candidate().unwrap_or_else(|| {
                    if self.buffer.len() > 1 {
                        self.rescheduled.push(byte); // re-schedule current byte for parsing
                        self.buffer.pop();
                    }
                    self.automata_state = self.automata.automata.start();
                    Err(std::mem::take(&mut self.buffer))
                });
                Some(event)
            }
        }
    }

    /// Take last successfully parsed item
    fn take_candidate(&mut self) -> Option<Result<T, MatcherBuffer>> {
        self.item_candidate.take().map(|(item, size)| {
            self.rescheduled.extend(self.buffer.drain(size..).rev());
            self.buffer.clear();
            self.automata_state = self.automata.automata.start();
            item
        })
    }
}

/// Automata tags that are used to pick correct decoder for the event
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum MatcherTag<T> {
    /// Automata already produces valid item
    Item(T),
    /// Index of the matcher that needs to be used to decode item
    Matcher(usize),
}

pub struct TTYEventDecoder {
    matcher: MatcherDecoder<TerminalEvent>,
}

impl Default for TTYEventDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl TTYEventDecoder {
    pub fn new() -> Self {
        Self {
            matcher: MatcherDecoder::new(TTY_EVENT_AUTOMATA.clone()),
        }
    }
}

impl Decoder for TTYEventDecoder {
    type Item = TerminalEvent;
    type Error = Error;

    fn decode<B: BufRead>(&mut self, buf: B) -> Result<Option<Self::Item>, Self::Error> {
        let event = self
            .matcher
            .decode(buf)?
            .transpose()
            .unwrap_or_else(|reject| {
                if reject.is_empty() {
                    return None;
                }
                tracing::info!(
                    "[TTYEventDecoder.decode] unhandled: {:?}",
                    String::from_utf8_lossy(&reject)
                );
                Some(TerminalEvent::Raw(reject.into_vec()))
            });
        Ok(event)
    }
}

pub struct TTYCommandDecoder {
    matcher: MatcherDecoder<TerminalCommand>,
}

impl Default for TTYCommandDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl TTYCommandDecoder {
    pub fn new() -> Self {
        Self {
            matcher: MatcherDecoder::new(TTY_COMMAND_AUTOMATA.clone()),
        }
    }
}

impl Decoder for TTYCommandDecoder {
    type Item = TerminalCommand;
    type Error = Error;

    fn decode<B: BufRead>(&mut self, buf: B) -> Result<Option<Self::Item>, Self::Error> {
        let cmd = self
            .matcher
            .decode(buf)?
            .transpose()
            .unwrap_or_else(|reject| {
                if reject.is_empty() {
                    return None;
                }
                tracing::info!(
                    "[TTYEventDecoder.decode] unhandled: {:?}",
                    String::from_utf8_lossy(&reject)
                );
                Some(TerminalCommand::Raw(reject.into_vec()))
            });
        Ok(cmd)
    }
}

#[derive(Clone)]
enum Void {}

trait Matcher: fmt::Debug + Send + Sync {
    /// Item type decoded by matcher
    type Item;
    /// NFA that matches will handle, if tag is Void decoder is called, otherwise
    /// tag is used to generate an event and decoder is not called.
    fn matcher(&self) -> Either<NFA<Void>, NFA<Self::Item>>;

    /// Decoder that should produce terminal event given matched data
    fn decode(&self, data: &[u8]) -> Option<Self::Item>;

    /// Map matcher item
    fn map<F, O>(self, func: F) -> MappedMatcher<Self, F>
    where
        Self: Sized,
        F: Fn(Self::Item) -> O + Send + Sync,
    {
        MappedMatcher {
            matcher: self,
            map: func,
        }
    }
}

struct MappedMatcher<M, F> {
    matcher: M,
    map: F,
}

impl<M: fmt::Debug, F> fmt::Debug for MappedMatcher<M, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MappedMatcher")
            .field("matcher", &self.matcher)
            .finish()
    }
}

impl<M, F, O> Matcher for MappedMatcher<M, F>
where
    M: Matcher,
    F: Fn(M::Item) -> O + Send + Sync,
{
    type Item = O;

    fn matcher(&self) -> Either<NFA<Void>, NFA<Self::Item>> {
        self.matcher
            .matcher()
            .map_right(|nfa| nfa.tags_map(&self.map))
    }

    fn decode(&self, data: &[u8]) -> Option<Self::Item> {
        self.matcher.decode(data).map(&self.map)
    }
}

/// Kitty Image Response
///
/// Reference: https://sw.kovidgoyal.net/kitty/graphics-protocol/#display-images-on-screen
#[derive(Debug)]
struct KittyImageMatcher;

impl Matcher for KittyImageMatcher {
    type Item = TerminalEvent;

    fn matcher(&self) -> Either<NFA<Void>, NFA<Self::Item>> {
        // `\x1b_Gkey=value(,key=value)*;response\x1b\\`
        let key_value = NFA::sequence([
            NFA::predicate(|b| b.is_ascii_alphanumeric()).some(),
            NFA::from("="),
            NFA::predicate(|b| b.is_ascii_alphanumeric()).some(),
        ]);
        let nfa = NFA::sequence([
            NFA::from("\x1b_G"),
            key_value.clone(),
            NFA::sequence([NFA::from(","), key_value]).many(),
            NFA::from(";"),
            NFA::predicate(|b| b != b'\x1b').many(),
            NFA::from("\x1b\\"),
        ]);
        Either::Left(nfa)
    }

    fn decode(&self, data: &[u8]) -> Option<Self::Item> {
        let mut iter = data[3..data.len() - 2].splitn(2, |b| *b == b';');
        let mut id = 0; // id can not be zero according to the spec
        let mut placement = None;
        for (key, value) in key_value_decode(b',', iter.next()?) {
            match key {
                b"i" => id = number_decode(value)? as u64,
                b"p" => {
                    placement.replace(number_decode(value)? as u64);
                }
                _ => {}
            }
        }
        let msg = iter.next()?;
        let error = if msg == b"OK" {
            None
        } else {
            Some(String::from_utf8_lossy(msg).to_string())
        };
        Some(TerminalEvent::KittyImage {
            id,
            placement,
            error,
        })
    }
}

/// Kitty Keyboard Protocol
///
/// Reference: https://sw.kovidgoyal.net/kitty/keyboard-protocol
#[derive(Debug)]
struct KittyKeyboardMatcher;

// https://sw.kovidgoyal.net/kitty/keyboard-protocol/#progressive-enhancement
// 0b0001 - Disambiguate escape codes
// 0b0100 - Report alternate keys
// 0b1000 - Report all keys as escape codes
//          NOTE: 0b1000 breaks shift+key as it no longer generated as uppercase
pub(crate) const KEYBOARD_LEVEL: usize = 0b0101;

impl Matcher for KittyKeyboardMatcher {
    type Item = TerminalEvent;

    fn matcher(&self) -> Either<NFA<Void>, NFA<Self::Item>> {
        let nfa = NFA::sequence([
            NFA::from("\x1b["),
            NFA::choice([
                NFA::from("?") + NFA::digit().some(),
                NFA::predicate(|c| matches!(c, b';' | b':' | b'0'..=b'9')).many(),
            ]),
            NFA::from("u"),
        ]);
        Either::Left(nfa)
    }

    fn decode(&self, data: &[u8]) -> Option<Self::Item> {
        let data = &data[2..data.len() - 1]; // skip CSI and `u`
        if data[0] == b'?' {
            let level = number_decode(&data[1..data.len()])?;
            return Some(TerminalEvent::KeyboardLevel(level));
        }

        // CSI unicode-key-code:alternate-key-codes ; modifiers:event-type ; text-as-codepoints u
        let mut fields = data.split(|c| *c == b';');

        // decode key
        let name = match fields.next() {
            Some(codes) => {
                // TODO: decode alternative keys
                let mut codes = numbers_decode(codes, b':');
                keyboard_decode_key(codes.next().unwrap_or(1))?
            }
            None => return None,
        };

        // decode modifiers
        let mode = match fields.next() {
            Some(modes) => {
                let mut modes = numbers_decode(modes, b':');
                let mode = match modes.next() {
                    Some(mode) if mode > 1 => KeyMod::from_bits((mode - 1) as u32),
                    _ => KeyMod::EMPTY,
                };
                let event_type = modes.next().unwrap_or(0);
                // TODO: decode press/release/repeat
                if event_type != 0 {
                    return None;
                }
                mode
            }
            None => KeyMod::EMPTY,
        };

        // TODO: decode text as code point
        let _text = fields.next();

        Some(TerminalEvent::Key(Key { name, mode }))
    }
}

fn keyboard_decode_key(code: usize) -> Option<KeyName> {
    // https://sw.kovidgoyal.net/kitty/keyboard-protocol/#functional-key-definitions
    let key = match code {
        27 => KeyName::Esc,
        13 => KeyName::Enter,
        9 => KeyName::Tab,
        127 => KeyName::Backspace,
        code @ 57376..=57398 => KeyName::F(code - 57376 + 13),
        code if code <= u32::MAX as usize && !(57344..=63743).contains(&code) => {
            KeyName::Char(char::from_u32(code as u32)?)
        }
        _ => return None,
    };
    Some(key)
}

/// DECRPM - DEC mode report
///
/// Reference: https://www.vt100.net/docs/vt510-rm/DECRPM
#[derive(Debug)]
struct DecModeMatcher;

impl Matcher for DecModeMatcher {
    type Item = TerminalEvent;

    fn matcher(&self) -> Either<NFA<Void>, NFA<Self::Item>> {
        let nfa = NFA::sequence([
            NFA::from("\x1b[?"),
            NFA::number(),
            NFA::from(";"),
            NFA::number(),
            NFA::from("$y"),
        ]);
        Either::Left(nfa)
    }

    fn decode(&self, data: &[u8]) -> Option<Self::Item> {
        // "\x1b[?{mode};{status}$y"
        let mut nums = numbers_decode(&data[3..data.len() - 2], b';');
        Some(TerminalEvent::DecMode {
            mode: crate::terminal::DecMode::from_usize(nums.next()?)?,
            status: DecModeStatus::from_usize(nums.next()?)?,
        })
    }
}

/// DA1 - Primary Device Attributes
///
/// Reference: https://vt100.net/docs/vt510-rm/DA1.html
#[derive(Debug)]
struct DeviceAttrsMatcher;

impl Matcher for DeviceAttrsMatcher {
    type Item = TerminalEvent;

    fn matcher(&self) -> Either<NFA<Void>, NFA<Self::Item>> {
        // "\x1b[?<attr_1>;...<attr_n>c"
        let nfa = NFA::sequence([
            NFA::from("\x1b[?"),
            (NFA::number() + NFA::from(";").optional()).some(),
            NFA::from("c"),
        ]);
        Either::Left(nfa)
    }

    fn decode(&self, data: &[u8]) -> Option<Self::Item> {
        Some(TerminalEvent::DeviceAttrs(
            numbers_decode(&data[3..data.len() - 1], b';')
                .filter(|v| v > &0)
                .collect(),
        ))
    }
}

/// OSC - Operating System Command Response
///
/// Reference: https://invisible-island.net/xterm/ctlseqs/ctlseqs.html#h3-Operating-System-Commands
#[derive(Debug)]
struct OSControlMatcher;

impl Matcher for OSControlMatcher {
    type Item = TerminalEvent;

    fn matcher(&self) -> Either<NFA<Void>, NFA<Self::Item>> {
        // "\x1b]<number>;.*\x1b\\"
        let nfa = NFA::sequence([
            NFA::from("\x1b]"),
            NFA::number(),
            NFA::from(";"),
            NFA::predicate(|c| c != b'\x1b' && c != b'\x07').some(),
            (NFA::from("\x1b\\") | NFA::from("\x07")),
        ]);
        Either::Left(nfa)
    }

    fn decode(&self, data: &[u8]) -> Option<Self::Item> {
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
        let color = parse_color(std::str::from_utf8(args.next()?).ok()?)?;
        Some(TerminalEvent::Color { name, color })
    }
}

fn parse_color(color_str: &str) -> Option<RGBA> {
    if let Ok(color) = color_str.parse() {
        return Some(color);
    }

    // rgb:r{1-4}/g{1-4}/b{1-4}
    // This format is used when querying colors with OCS,
    // reference [`xparsecolor`](https://linux.die.net/man/3/xparsecolor)
    fn parse_component(string: &str) -> Option<u8> {
        let value = usize::from_str_radix(string, 16).ok()?;
        let value = match string.len() {
            4 => value / 256,
            3 => value / 16,
            2 => value,
            1 => value * 17,
            _ => return None,
        };
        Some(value.clamp(0, 255) as u8)
    }
    let rgb = color_str.strip_prefix("rgb:")?;
    let mut iter = rgb.split('/');
    Some(RGBA::new(
        parse_component(iter.next()?)?,
        parse_component(iter.next()?)?,
        parse_component(iter.next()?)?,
        255,
    ))
}

/// DECRPSS - Report Selection or Setting
///
/// Reference: https://vt100.net/docs/vt510-rm/DECRPSS.html
#[derive(Debug)]
struct ReportSettingMatcher;

impl Matcher for ReportSettingMatcher {
    type Item = TerminalEvent;

    fn matcher(&self) -> Either<NFA<Void>, NFA<Self::Item>> {
        // "\x1bP{0|1}$p{data}\x1b\\"
        let nfa = NFA::sequence([
            NFA::from("\x1bP"),              // DCS
            NFA::from("0") | NFA::from("1"), // response code
            NFA::from("$r"),
            NFA::predicate(|c| c != b'\x1b').many(), // data
            NFA::from("\x1b\\"),                     // ST
        ]);
        Either::Left(nfa)
    }

    fn decode(&self, data: &[u8]) -> Option<Self::Item> {
        // DECRPSS "\x1bP{0|1}$p{data}\x1b\\"
        let code = data[2];
        let payload = &data[5..data.len() - 2];
        if code != b'1' {
            return None;
        }
        if payload.ends_with(b"m") {
            let face = sgr_face(&payload[..payload.len() - 1]).apply(Face::default());
            Some(TerminalEvent::FaceGet(face))
        } else {
            tracing::info!(
                "[ReportSettingMatcher.decode] unhandled DECRPSS: {:?}",
                payload
            );
            None
        }
    }
}

/// SGR - Set Graphic Rendition
#[derive(Debug, Default)]
struct GraphicRenditionMatcher;

impl Matcher for GraphicRenditionMatcher {
    type Item = FaceModify;

    fn matcher(&self) -> Either<NFA<Void>, NFA<Self::Item>> {
        let code = NFA::predicate(|c| matches!(c, b'0'..=b'9' | b':')).many();
        let nfa = NFA::sequence([
            NFA::from("\x1b["),
            (code + NFA::from(";").optional()).some(),
            NFA::from("m"),
        ]);
        Either::Left(nfa)
    }

    fn decode(&self, data: &[u8]) -> Option<Self::Item> {
        // "\x1b[(<cmd>;?)*m"
        Some(sgr_face(&data[2..data.len() - 1]))
    }
}

/// SGR Mouse event
///
/// Reference: https://invisible-island.net/xterm/ctlseqs/ctlseqs.html#h2-Mouse-Tracking
#[derive(Debug)]
struct MouseEventMatcher;

impl Matcher for MouseEventMatcher {
    type Item = TerminalEvent;

    fn matcher(&self) -> Either<NFA<Void>, NFA<Self::Item>> {
        let nfa = NFA::sequence([
            NFA::from("\x1b[<"),
            NFA::number(),
            NFA::from(";"),
            NFA::number(),
            NFA::from(";"),
            NFA::number(),
            NFA::predicate(|b| b == b'm' || b == b'M'),
        ]);
        Either::Left(nfa)
    }

    fn decode(&self, data: &[u8]) -> Option<Self::Item> {
        // "\x1b[<{event};{row};{col}(m|M)"
        let mut nums = numbers_decode(&data[3..data.len() - 1], b';');
        let event = nums.next()?;
        let col = nums.next()? - 1;
        let row = nums.next()? - 1;

        let mut mode = KeyMod::from_bits(((event >> 2) & 7) as u32);
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

        Some(TerminalEvent::Mouse(Mouse {
            name,
            mode,
            pos: Position { row, col },
        }))
    }
}

/// Request Termcap/Terminfo String (XTGETTCAP)
///
/// Reference: https://invisible-island.net/xterm/ctlseqs/ctlseqs.html#h3-Application-Program-Command-functions
#[derive(Debug)]
struct TermCapMatcher;

impl Matcher for TermCapMatcher {
    type Item = TerminalEvent;

    fn matcher(&self) -> Either<NFA<Void>, NFA<Self::Item>> {
        let hex = NFA::predicate(|b| b.is_ascii_hexdigit());
        let hex = hex.clone() + hex;
        let key_value = NFA::sequence([hex.clone().some(), NFA::from("="), hex.clone().some()]);
        let nfa = NFA::choice([
            // success
            NFA::sequence([
                NFA::from("\x1bP1+r"),
                NFA::sequence([
                    key_value.clone(),
                    NFA::sequence([NFA::from(";"), key_value]).many(),
                ])
                .optional(),
            ]),
            // failure
            NFA::sequence([
                NFA::from("\x1bP0+r"),
                NFA::sequence([
                    hex.clone().some(),
                    NFA::sequence([NFA::from(";"), hex.some()]).many(),
                ])
                .optional(),
            ]),
        ]) + NFA::from("\x1b\\");
        Either::Left(nfa)
    }

    fn decode(&self, data: &[u8]) -> Option<Self::Item> {
        // `\x1bP(0|1)+rkey=value(;key=value)\x1b\\`
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
        Some(TerminalEvent::Termcap(termcap))
    }
}

/// UTF8
///
/// Mostly utf8, but one-byte codes are restricted to the printable set
#[derive(Debug)]
struct UTF8Matcher {
    mode: UTF8Mode,
}

impl UTF8Matcher {
    fn new(mode: UTF8Mode) -> Self {
        Self { mode }
    }
}

impl Matcher for UTF8Matcher {
    type Item = char;

    fn matcher(&self) -> Either<NFA<Void>, NFA<Self::Item>> {
        Either::Left(utf8_nfa(self.mode))
    }

    fn decode(&self, data: &[u8]) -> Option<Self::Item> {
        Some(utf8_decode(data))
    }
}

/// DSR-CPR - Device status report - cursor position report
///
/// Reference: https://vt100.net/docs/vt510-rm/DSR-CPR.html
#[derive(Debug)]
struct CursorPositionMatcher;

impl Matcher for CursorPositionMatcher {
    type Item = TerminalEvent;

    fn matcher(&self) -> Either<NFA<Void>, NFA<Self::Item>> {
        let nfa = NFA::sequence([
            NFA::from("\x1b["),
            NFA::number(),
            NFA::from(";"),
            NFA::number(),
            NFA::from("R"),
        ]);
        Either::Left(nfa)
    }

    fn decode(&self, data: &[u8]) -> Option<Self::Item> {
        // "\x1b[{row};{col}R"
        let mut nums = numbers_decode(&data[2..data.len() - 1], b';');
        Some(TerminalEvent::CursorPosition(Position {
            row: nums.next()? - 1,
            col: nums.next()? - 1,
        }))
    }
}

/// XTWINOPS - Window manipulation response
///
/// Reference: https://invisible-island.net/xterm/ctlseqs/ctlseqs.html
/// "\x1b[18t" - Report the size of the text area in characters ("\x1b[8{height};{width}t")
/// "\x1b[14t" - Report text area size in pixels ("\x1b[4{height};{width}t")
///
/// This matcher expects to receive two responses at once and used as a fallback to
/// way to get terminal size if ioctl does not work as expected.
#[derive(Debug)]
struct TermSizeMatcher;

impl Matcher for TermSizeMatcher {
    type Item = TerminalEvent;

    fn matcher(&self) -> Either<NFA<Void>, NFA<Self::Item>> {
        let size = NFA::sequence([
            NFA::from(";"),
            NFA::number(),
            NFA::from(";"),
            NFA::number(),
            NFA::from("t"),
        ]);
        let nfa = NFA::sequence([NFA::from("\x1b[8"), size.clone(), NFA::from("\x1b[4"), size]);
        Either::Left(nfa)
    }

    fn decode(&self, data: &[u8]) -> Option<Self::Item> {
        // "\x1b[8;{cell_height};{cell_width}t\x1b[4;{pixel_height};{pixel_width}t"
        let mut chunks = data.split(|c| *c == b'\x1b');
        chunks.next()?; // empty
        let cell_size = chunks.next()?;
        let mut nums = numbers_decode(&cell_size[3..cell_size.len() - 1], b';');
        let cell_height = nums.next()?;
        let cell_width = nums.next()?;
        let pixel_size = chunks.next()?;
        let mut nums = numbers_decode(&pixel_size[3..pixel_size.len() - 1], b';');
        let pixel_height = nums.next()?;
        let pixel_width = nums.next()?;
        Some(TerminalEvent::Size(TerminalSize {
            cells: Size {
                height: cell_height,
                width: cell_width,
            },
            pixels: Size {
                height: pixel_height,
                width: pixel_width,
            },
        }))
    }
}

#[derive(Debug)]
struct BracketedPasteMatcher;

impl Matcher for BracketedPasteMatcher {
    type Item = TerminalEvent;

    fn matcher(&self) -> Either<NFA<Void>, NFA<Self::Item>> {
        let nfa = NFA::sequence([
            NFA::from("\x1b[200~"),
            NFA::predicate(|b| b != b'\x1b').many(),
            NFA::from("\x1b[201~"),
        ]);
        Either::Left(nfa)
    }

    fn decode(&self, data: &[u8]) -> Option<Self::Item> {
        let text = String::from_utf8(data[6..data.len() - 6].into()).ok()?;
        Some(TerminalEvent::Paste(text))
    }
}

#[derive(Debug)]
struct BasicEventsMatcher;

impl Matcher for BasicEventsMatcher {
    type Item = TerminalEvent;

    fn matcher(&self) -> Either<NFA<Void>, NFA<Self::Item>> {
        Either::Right(basic_events_nfa())
    }

    fn decode(&self, _data: &[u8]) -> Option<Self::Item> {
        None
    }
}

/// NFA for TerminalEvent events, that do not require parsing
fn basic_events_nfa() -> NFA<TerminalEvent> {
    let mut cmds: Vec<NFA<TerminalEvent>> = Vec::new();

    // construct NFA for basic key (no additional parsing is needed)
    fn basic_key(seq: &str, key: impl Into<Key>) -> NFA<TerminalEvent> {
        NFA::from(seq).tag_stop_state(TerminalEvent::Key(key.into()))
    }

    // [Reference](http://www.leonerd.org.uk/hacks/fixterms/)
    // but it does not always match real behavior

    cmds.push(basic_key("\x1b", KeyName::Esc));
    cmds.push(basic_key("\x7f", KeyName::Backspace));
    cmds.push(basic_key("\x00", (KeyName::Char(' '), KeyMod::CTRL)));

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
    for byte in (0..=255u8).filter(|c| c.is_ascii_uppercase()) {
        let c = char::from(byte);
        cmds.push(basic_key(
            &format!("\x1b{}", c),
            (
                KeyName::Char(c.to_ascii_lowercase()),
                KeyMod::ALT | KeyMod::SHIFT,
            ),
        ));
    }

    // alt+punctuation
    for byte in (0..=255u8).filter(|c| c.is_ascii_punctuation()) {
        let c = char::from(byte);
        cmds.push(basic_key(
            &format!("\x1b{}", c),
            (KeyName::Char(c), KeyMod::ALT),
        ));
    }

    // alt+digit
    for byte in (0..=255u8).filter(|c| c.is_ascii_digit()) {
        let c = char::from(byte);
        cmds.push(basic_key(
            &format!("\x1b{}", c),
            (KeyName::Char(c), KeyMod::ALT),
        ));
    }

    for (name, code) in [
        (KeyName::Home, "1"),
        (KeyName::Insert, "2"),
        (KeyName::Delete, "3"),
        (KeyName::End, "4"),
        (KeyName::PageUp, "5"),
        (KeyName::PageDown, "6"),
        (KeyName::Insert, "7"),
        (KeyName::End, "8"),
        (KeyName::F(1), "11"),
        (KeyName::F(2), "12"),
        (KeyName::F(3), "13"),
        (KeyName::F(4), "14"),
        (KeyName::F(5), "15"),
        (KeyName::F(6), "17"),
        (KeyName::F(7), "18"),
        (KeyName::F(8), "19"),
        (KeyName::F(9), "20"),
        (KeyName::F(10), "21"),
        (KeyName::F(11), "23"),
        (KeyName::F(12), "24"),
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
        (KeyName::F(1), "O", "P"),
        (KeyName::F(1), "[", "P"),
        (KeyName::F(2), "O", "Q"),
        (KeyName::F(2), "[", "Q"),
        (KeyName::F(3), "O", "R"),
        (KeyName::F(3), "[", "R"),
        (KeyName::F(4), "O", "S"),
        (KeyName::F(4), "[", "S"),
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

    NFA::choice(cmds)
}

const CUBE: [u8; 6] = [0x00, 0x5f, 0x87, 0xaf, 0xd7, 0xff];
const GREYS: [u8; 24] = [
    0x08, 0x12, 0x1c, 0x26, 0x30, 0x3a, 0x44, 0x4e, 0x58, 0x62, 0x6c, 0x76, 0x80, 0x8a, 0x94, 0x9e,
    0xa8, 0xb2, 0xbc, 0xc6, 0xd0, 0xda, 0xe4, 0xee,
];
const COLORS: [RGBA; 16] = [
    RGBA::new(0, 0, 0, 255),
    RGBA::new(128, 0, 0, 255),
    RGBA::new(0, 128, 0, 255),
    RGBA::new(128, 128, 0, 255),
    RGBA::new(0, 0, 128, 255),
    RGBA::new(128, 0, 128, 255),
    RGBA::new(0, 128, 128, 255),
    RGBA::new(192, 192, 192, 255),
    RGBA::new(128, 128, 128, 255),
    RGBA::new(255, 0, 0, 255),
    RGBA::new(0, 255, 0, 255),
    RGBA::new(255, 255, 0, 255),
    RGBA::new(0, 0, 255, 255),
    RGBA::new(255, 0, 255, 255),
    RGBA::new(0, 255, 255, 255),
    RGBA::new(255, 255, 255, 255),
];

fn sgr_color<'a>(mut cmds: impl Iterator<Item = &'a [u8]>) -> Option<RGBA> {
    match number_decode(cmds.next()?)? {
        5 => {
            // color from 256 color palette
            let mut index = number_decode(cmds.next()?)?;
            if index < 16 {
                Some(COLORS[index])
            } else if index < 232 {
                index -= 16;
                let ri = index / 36;
                index -= ri * 36;
                let gi = index / 6;
                index -= gi * 6;
                let bi = index;
                Some(RGBA::new(CUBE[ri], CUBE[gi], CUBE[bi], 255))
            } else if index < 256 {
                let v = GREYS[index - 232];
                Some(RGBA::new(v, v, v, 255))
            } else {
                None
            }
        }
        2 => {
            // true color
            //
            // It can contain either three or four components
            // in the case of four first component is ignored
            match [
                cmds.next().and_then(number_decode),
                cmds.next().and_then(number_decode),
                cmds.next().and_then(number_decode),
                cmds.next().and_then(number_decode),
            ] {
                [Some(r), Some(g), Some(b), None] | [_, Some(r), Some(g), Some(b)] => {
                    Some(RGBA::new(r as u8, g as u8, b as u8, 255))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// Apply SGR commands to the provided Face
fn sgr_face(data: &[u8]) -> FaceModify {
    let mut face = FaceModify::default();
    let mut groups = data.split(|b| matches!(b, b';'));
    while let Some(group) = groups.next() {
        let mut args = group.split(|b| matches!(b, b':'));
        let cmd = args.next().and_then(number_decode);
        let args_empty = args.size_hint().0 == 0;
        let mut sgr_color_thunk = || {
            if args_empty {
                sgr_color(&mut groups)
            } else {
                sgr_color(&mut args)
            }
        };
        match cmd {
            Some(0) | None => {
                face = FaceModify {
                    reset: true,
                    ..FaceModify::default()
                }
            }
            // bold
            Some(1) => face.bold = Some(true),
            Some(21) => face.bold = Some(false),
            // italic
            Some(3) => face.italic = Some(true),
            Some(23) => face.italic = Some(false),
            // underline
            Some(4) => match args.next().and_then(number_decode) {
                Some(2) => face.underline = Some(UnderlineStyle::Double),
                Some(3) => face.underline = Some(UnderlineStyle::Curly),
                Some(4) => face.underline = Some(UnderlineStyle::Dotted),
                Some(5) => face.underline = Some(UnderlineStyle::Dashed),
                _ => face.underline = Some(UnderlineStyle::Straight),
            },
            Some(24) => face.underline = Some(UnderlineStyle::None),
            // blink
            Some(5) => face.blink = Some(true),
            Some(25) => face.blink = Some(false),
            // strike
            Some(9) => face.strike = Some(true),
            Some(29) => face.strike = Some(false),
            // foreground color
            Some(38) => face.fg = sgr_color_thunk(),
            // background color
            Some(48) => face.bg = sgr_color_thunk(),
            // underline color
            Some(58) => face.underline_color = sgr_color_thunk(),
            // named colors
            Some(v @ 30..=37) => face.fg = Some(COLORS[v - 30]),
            Some(v @ 90..=97) => face.fg = Some(COLORS[v - 82]),
            Some(v @ 40..=48) => face.bg = Some(COLORS[v - 40]),
            Some(v @ 100..=107) => face.bg = Some(COLORS[v - 92]),
            _ => {}
        }
    }
    face
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

/// Semi-colon separated positive numbers
fn numbers_decode(data: &[u8], sep: u8) -> impl Iterator<Item = usize> + '_ {
    data.split(move |b| *b == sep).filter_map(number_decode)
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

#[derive(Debug, Clone, Copy)]
enum UTF8Mode {
    Canonical,
    Printable,
    NotEscape,
}

fn utf8_nfa<T: Clone>(mode: UTF8Mode) -> NFA<T> {
    let utf8_one = match mode {
        UTF8Mode::Canonical => NFA::predicate(|b| b >> 7 == 0b0),
        UTF8Mode::Printable => NFA::predicate(|b| (b' '..=b'~').contains(&b)),
        UTF8Mode::NotEscape => NFA::predicate(|b| b >> 7 == 0b0 && b != b'\x1b'),
    };
    let utf8_two = NFA::predicate(|b| b >> 5 == 0b110);
    let utf8_three = NFA::predicate(|b| b >> 4 == 0b1110);
    let utf8_four = NFA::predicate(|b| b >> 3 == 0b11110);
    let utf8_tail = NFA::predicate(|b| b >> 6 == 0b10);
    NFA::choice([
        utf8_one,
        utf8_two + utf8_tail.clone(),
        utf8_three + utf8_tail.clone() + utf8_tail.clone(),
        utf8_four + utf8_tail.clone() + utf8_tail.clone() + utf8_tail,
    ])
}

/// Decode hex encoded slice
pub fn hex_decode(slice: &[u8]) -> impl Iterator<Item = u8> + '_ {
    let value = |byte| match byte {
        b'A'..=b'F' => Some(byte - b'A' + 10),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'0'..=b'9' => Some(byte - b'0'),
        _ => None,
    };
    slice
        .chunks(2)
        .map(move |pair| Some((value(pair[0])? << 4) | value(pair[1])?))
        .take_while(|value| value.is_some())
        .flatten()
}

const BASE64_DECODE: &[u8; 256] = b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00>\x00\x00\x00?456789:;<=\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x00\x00\x00\x00\x00\x00\x1a\x1b\x1c\x1d\x1e\x1f !\x22#$%&\'()*+,-./0123\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";

pub struct Base64Decoder<R> {
    read: R,
    buffer: [u8; 64],
    buffer_offset: usize,
    buffer_size: usize,
}

impl<R: Read> Base64Decoder<R> {
    pub fn new(read: R) -> Self {
        Self {
            read,
            buffer: [0u8; 64],
            buffer_offset: 0,
            buffer_size: 0,
        }
    }

    /// Decode 4 base64 bytes into 3 bytes
    #[inline]
    fn decode_u8x4(chunk: [u8; 4]) -> [u8; 3] {
        let [i0, i1, i2, i3] = chunk;
        let o0 = BASE64_DECODE[i0 as usize];
        let o1 = BASE64_DECODE[i1 as usize];
        let o2 = BASE64_DECODE[i2 as usize];
        let o3 = BASE64_DECODE[i3 as usize];
        let b0 = (o0 << 2) | (o1 >> 4);
        let b1 = (o1 << 4) | (o2 >> 2);
        let b2 = (o2 << 6) | o3;
        [b0, b1, b2]
    }

    /// Decode number of encoded bytes based on the padding symbol
    #[inline]
    fn decode_size(chunk: [u8; 4]) -> usize {
        let [_, _, i2, i3] = chunk;
        if i2 == b'=' {
            1
        } else if i3 == b'=' {
            2
        } else {
            3
        }
    }

    fn buffer(&self) -> &[u8] {
        &self.buffer[self.buffer_offset..self.buffer_size]
    }

    fn buffer_fill(&mut self) -> std::io::Result<()> {
        if self.buffer_offset == self.buffer_size {
            self.buffer_offset = 0;
            self.buffer_size = 0;
        }
        while self.buffer_size + 3 <= self.buffer.len() {
            let mut input = [0u8; 4];
            let size = self.read.read(&mut input)?;
            if size == 0 {
                break;
            } else if size != 4 {
                return Err(std::io::Error::other(Error::ParseError(
                    "Base64Decoder",
                    "input length is not dividable by 4".to_owned(),
                )));
            }
            let out = Self::decode_u8x4(input);
            let out_size = Self::decode_size(input);
            self.buffer[self.buffer_size..self.buffer_size + out_size]
                .copy_from_slice(&out[..out_size]);
            self.buffer_size += out_size;
        }
        Ok(())
    }
}

impl<R: Read> Read for Base64Decoder<R> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        let mut out_offset = 0;
        while out_offset < out.len() {
            if self.buffer().is_empty() {
                self.buffer_fill()?;
            }
            let buffer = self.buffer();
            if buffer.is_empty() {
                break;
            }
            let size = buffer.len().min(out.len() - out_offset);
            out[out_offset..out_offset + size].copy_from_slice(&buffer[..size]);
            out_offset += size;
            self.buffer_offset += size;
        }
        Ok(out_offset)
    }
}

#[cfg(test)]
mod tests {
    use crate::{common::Rnd, encoder::Base64Encoder};

    use super::*;
    use std::io::{Cursor, Write};

    #[test]
    fn test_basic() -> Result<(), Error> {
        let mut cursor = Cursor::new(Vec::new());
        let mut decoder = TTYEventDecoder::new();

        // send incomplete sequence
        write!(cursor.get_mut(), "\x1b")?;

        assert_eq!(decoder.decode(&mut cursor)?, None);
        assert_eq!(cursor.position(), 1);

        // send rest of the sequence, plus full other sequence, and some garbage
        write!(cursor.get_mut(), "OR\x1b[15~AB\x1bM")?;

        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::Key("f3".parse()?))
        );
        assert_eq!(cursor.position(), 3);

        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::Key(KeyName::F(5).into()))
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
        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::Key("shift+alt+m".parse()?)),
        );
        assert_eq!(decoder.decode(&mut cursor)?, None);

        Ok(())
    }

    #[test]
    fn test_reschedule() -> Result<(), Error> {
        let mut cursor = Cursor::new(Vec::new());
        let mut decoder = TTYEventDecoder::new();

        // write possible match sequence
        write!(cursor.get_mut(), "\x1bO")?;
        assert_eq!(decoder.decode(&mut cursor)?, None);
        assert_eq!(cursor.position(), 2);

        // fail to rollback to possible match
        write!(cursor.get_mut(), "T")?;

        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::Key("alt+shift+o".parse()?))
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
        assert_eq!(
            TTYEventDecoder::new().decode(Cursor::new("\x1b[97;15R"))?,
            Some(TerminalEvent::CursorPosition(Position { row: 96, col: 14 })),
        );
        Ok(())
    }

    #[test]
    fn test_terminal_size() -> Result<(), Error> {
        assert_eq!(
            TTYEventDecoder::new().decode(Cursor::new("\x1b[8;101;202t\x1b[4;3104;1482t"))?,
            Some(TerminalEvent::Size(TerminalSize {
                cells: Size {
                    width: 202,
                    height: 101,
                },
                pixels: Size {
                    width: 1482,
                    height: 3104,
                }
            })),
        );
        Ok(())
    }

    #[test]
    fn test_mouse_sgr() -> Result<(), Error> {
        let mut cursor = Cursor::new(Vec::new());
        let mut decoder = TTYEventDecoder::new();

        write!(cursor.get_mut(), "\x1b[<0;94;14M")?;
        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::Mouse(Mouse {
                name: KeyName::MouseLeft,
                mode: KeyMod::PRESS,
                pos: Position::new(13, 93),
            }))
        );

        write!(cursor.get_mut(), "\x1b[<26;33;26m")?;
        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::Mouse(Mouse {
                name: KeyName::MouseRight,
                mode: KeyMod::ALT | KeyMod::CTRL,
                pos: Position::new(25, 32),
            }))
        );

        write!(cursor.get_mut(), "\x1b[<65;142;30M")?;
        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::Mouse(Mouse {
                name: KeyName::MouseWheelUp,
                mode: KeyMod::PRESS,
                pos: Position::new(29, 141),
            }))
        );

        Ok(())
    }

    #[test]
    fn test_char() -> Result<(), Error> {
        assert_eq!(
            TTYEventDecoder::new().decode(Cursor::new("\u{1F431}"))?,
            Some(TerminalEvent::Key(KeyName::Char('🐱').into())),
        );
        Ok(())
    }

    #[test]
    fn test_dec_mode() -> Result<(), Error> {
        use crate::DecMode;

        let mut cursor = Cursor::new(Vec::new());
        let mut decoder = TTYEventDecoder::new();

        write!(cursor.get_mut(), "\x1b[?1000;1$y")?;
        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::DecMode {
                mode: DecMode::MouseReport,
                status: DecModeStatus::Enabled,
            }),
        );

        write!(cursor.get_mut(), "\x1b[?2026;0$y")?;
        assert_eq!(
            decoder.decode(&mut cursor)?,
            Some(TerminalEvent::DecMode {
                mode: DecMode::SynchronizedOutput,
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
        cursor.get_mut().write_all(&c[..1])?;
        assert_eq!(decoder.decode(&mut cursor)?, None);
        cursor.get_mut().write_all(&c[1..])?;
        assert_eq!(decoder.decode(&mut cursor)?, Some('я'));

        // invalid
        cursor.get_mut().write_all(&c[..1])?;
        assert_eq!(decoder.decode(&mut cursor)?, None);
        cursor.get_mut().write_all(&c[..1])?;
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
        let mut decoder = TTYEventDecoder::new();

        write!(cursor.get_mut(), "\x1b_Gi=127;OK\x1b\\")?;
        write!(
            cursor.get_mut(),
            "\x1b_Gi=31,p=11,ignored=attr;error message\x1b\\"
        )?;

        let mut result = Vec::new();
        decoder.decode_into(&mut cursor, &mut result)?;
        assert_eq!(
            result,
            vec![
                TerminalEvent::KittyImage {
                    id: 127,
                    placement: None,
                    error: None
                },
                TerminalEvent::KittyImage {
                    id: 31,
                    placement: Some(11),
                    error: Some("error message".to_string())
                },
            ]
        );

        Ok(())
    }

    #[test]
    fn test_terminfo() -> Result<(), Error> {
        let mut cursor = Cursor::new(Vec::new());
        let mut decoder = TTYEventDecoder::new();

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
        let mut result = Vec::new();
        TTYEventDecoder::new().decode_into(Cursor::new("\x1b[?62;c\x1b[?64;4c"), &mut result)?;
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
        let mut result = Vec::new();
        TTYEventDecoder::new().decode_into(
            Cursor::new("\x1b]4;1;rgb:cc/24/1d\x1b\\\x1b]10;#ebdbb2\x07"),
            &mut result,
        )?;
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

    #[test]
    fn test_sgr() -> Result<(), Error> {
        let mut result = Vec::new();
        TTYEventDecoder::new().decode_into(
            Cursor::new(
                "\x1b[48;5;150m\x1b[1m\x1b[38:2:255:128:64m\x1b[m\x1b[32m\x1b[1;4;91;102m\x1b[24m\x1b[4:3m\x1b[58;2;1;2;3m"
            ),
            &mut result,
        )?;

        assert_eq!(result.len(), 9);
        for (item, reference) in result.into_iter().zip([
            FaceModify {
                bg: Some("#afd787".parse()?),
                ..Default::default()
            },
            FaceModify {
                bold: Some(true),
                ..Default::default()
            },
            FaceModify {
                fg: Some("#ff8040".parse()?),
                ..Default::default()
            },
            FaceModify {
                reset: true,
                ..Default::default()
            },
            FaceModify {
                fg: Some("#008000".parse()?),
                ..Default::default()
            },
            FaceModify {
                fg: Some("#ff0000".parse()?),
                bg: Some("#00ff00".parse()?),
                bold: Some(true),
                underline: Some(UnderlineStyle::Straight),
                ..Default::default()
            },
            FaceModify {
                underline: Some(UnderlineStyle::None),
                ..Default::default()
            },
            FaceModify {
                underline: Some(UnderlineStyle::Curly),
                ..Default::default()
            },
            FaceModify {
                underline_color: Some("#010203".parse()?),
                ..Default::default()
            },
        ]) {
            assert_eq!(
                item,
                TerminalEvent::Command(TerminalCommand::FaceModify(reference))
            )
        }

        Ok(())
    }

    #[test]
    fn test_report_setting() -> Result<(), Error> {
        let mut result = Vec::new();
        TTYEventDecoder::new().decode_into(
            Cursor::new("\x1bP1$r48:2:1:2:3m\x1b\\\x1bP1$r0;48:2::6:5:4m\x1b\\"),
            &mut result,
        )?;

        assert_eq!(
            result,
            vec![
                TerminalEvent::FaceGet("bg=#010203".parse()?),
                TerminalEvent::FaceGet("bg=#060504".parse()?)
            ],
        );

        Ok(())
    }

    #[test]
    fn test_kitty_keyboard() -> Result<(), Error> {
        let mut cursor = Cursor::new(Vec::new());
        let mut decoder = TTYEventDecoder::new();

        write!(cursor.get_mut(), "\x1b[?15u")?;
        write!(cursor.get_mut(), "\x1b[27;7u")?;
        write!(cursor.get_mut(), "\x1b[99;5u")?;
        write!(cursor.get_mut(), "\x1b[1;6P")?;
        write!(cursor.get_mut(), "\x1b[9;0u")?;

        let mut result = Vec::new();
        decoder.decode_into(&mut cursor, &mut result)?;

        assert_eq!(
            result,
            vec![
                TerminalEvent::KeyboardLevel(15),
                TerminalEvent::Key("ctrl+alt+esc".parse()?),
                TerminalEvent::Key("ctrl+c".parse()?),
                TerminalEvent::Key("ctrl+shift+f1".parse()?),
                TerminalEvent::Key("tab".parse()?)
            ],
        );

        Ok(())
    }

    #[test]
    fn test_bracketed_paste() -> Result<(), Error> {
        assert_eq!(
            TTYEventDecoder::new().decode(Cursor::new("\x1b[200~some awesome text\x1b[201~"))?,
            Some(TerminalEvent::Paste("some awesome text".to_string())),
        );
        Ok(())
    }

    #[test]
    fn test_base64() -> Result<(), Error> {
        for (base64, reference) in [
            ("TWFu", "Man"),
            ("bGlnaHQgd29yay4=", "light work."),
            ("bGlnaHQgd29yaw==", "light work"),
            ("bGlnaHQgd29y", "light wor"),
            ("bWFnZ290", "maggot"),
        ] {
            let mut result = String::new();
            Base64Decoder::new(Cursor::new(base64)).read_to_string(&mut result)?;
            assert_eq!(result, reference);
        }

        Ok(())
    }

    #[test]
    fn test_base64_encode_decode() -> Result<(), Error> {
        const SIZE: usize = 4096;

        let mut rnd = Rnd::new();
        let mut data = Vec::new();
        while data.len() < SIZE {
            data.write_all(&rnd.next_u8x4())?
        }

        let mut encoder = Base64Encoder::new(Vec::new());
        encoder.write_all(&data)?;
        let encoded = encoder.finish()?;

        let mut decoded = Vec::new();
        let mut decoder = Base64Decoder::new(encoded.as_slice());
        decoder.read_to_end(&mut decoded)?;

        assert_eq!(decoded, data);
        Ok(())
    }
}
