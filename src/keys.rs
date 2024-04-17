use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Key types
use crate::Error;
use std::{
    collections::BTreeMap,
    fmt::{self, Debug, Write as _},
    str::FromStr,
    sync::Arc,
};

/// Key descriptor
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key {
    /// Key name
    pub name: KeyName,
    /// Key mode
    pub mode: KeyMod,
}

impl Key {
    pub fn new(name: KeyName, mode: KeyMod) -> Self {
        Self { name, mode }
    }
}

impl From<KeyName> for Key {
    fn from(name: KeyName) -> Self {
        Self {
            name,
            mode: KeyMod::EMPTY,
        }
    }
}

impl From<(KeyName, KeyMod)> for Key {
    fn from(pair: (KeyName, KeyMod)) -> Self {
        Self {
            name: pair.0,
            mode: pair.1,
        }
    }
}

impl fmt::Debug for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.mode.is_empty() {
            write!(f, "{:?}", self.name)?;
        } else {
            write!(f, "{:?}+{:?}", self.mode, self.name)?;
        }
        Ok(())
    }
}

impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl FromStr for Key {
    type Err = Error;

    fn from_str(string: &str) -> Result<Self, Self::Err> {
        let mut key_name = None;
        let mut key_mod = KeyMod::EMPTY;
        for attr in string.split('+') {
            match attr.to_lowercase().as_ref() {
                "alt" => key_mod |= KeyMod::ALT,
                "ctrl" => key_mod |= KeyMod::CTRL,
                "shift" => key_mod |= KeyMod::SHIFT,
                "press" => key_mod |= KeyMod::PRESS,
                "super" => key_mod |= KeyMod::SUPER,
                "hyper" => key_mod |= KeyMod::HYPER,
                "meta" => key_mod |= KeyMod::META,
                "capslock" => key_mod |= KeyMod::CAPSLOCK,
                name => match name.parse::<KeyName>() {
                    Ok(name) => {
                        if key_name.replace(name).is_some() {
                            key_name.take();
                            break;
                        }
                    }
                    _ => break,
                },
            }
        }
        match key_name {
            Some(key_name) => Ok(Key::new(key_name, key_mod)),
            _ => Err(Error::ParseError("Key", string.to_string())),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub struct KeyChord {
    keys: Arc<Vec<Key>>,
}

impl KeyChord {
    pub fn new(keys: Vec<Key>) -> KeyChord {
        Self { keys: keys.into() }
    }

    pub fn keys(&self) -> &[Key] {
        self.keys.as_ref()
    }
}

impl FromIterator<Key> for KeyChord {
    fn from_iter<T: IntoIterator<Item = Key>>(iter: T) -> Self {
        Self {
            keys: iter.into_iter().collect::<Vec<_>>().into(),
        }
    }
}

impl<'a> FromIterator<&'a Key> for KeyChord {
    fn from_iter<T: IntoIterator<Item = &'a Key>>(iter: T) -> Self {
        Self::from_iter(iter.into_iter().cloned())
    }
}

impl AsRef<[Key]> for KeyChord {
    fn as_ref(&self) -> &[Key] {
        self.keys()
    }
}

impl fmt::Display for KeyChord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, key) in self.keys().iter().enumerate() {
            write!(f, "{}", key)?;
            if index + 1 < self.keys().len() {
                write!(f, " ")?;
            }
        }
        Ok(())
    }
}

impl fmt::Debug for KeyChord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl FromStr for KeyChord {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let chord = s
            .split(' ')
            .filter(|k| !k.is_empty())
            .map(Key::from_str)
            .collect::<Result<Vec<Key>, Error>>()?;
        if chord.is_empty() {
            Err(Error::ParseError("Key", s.to_string()))
        } else {
            Ok(Self::new(chord))
        }
    }
}

impl Serialize for KeyChord {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for KeyChord {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let chord_str = std::borrow::Cow::<'de, str>::deserialize(deserializer)?;
        KeyChord::from_str(chord_str.as_ref()).map_err(serde::de::Error::custom)
    }
}

/// Key name
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KeyName {
    Backspace,
    Char(char),
    Delete,
    Insert,
    Down,
    End,
    Enter,
    Esc,
    F(usize),
    Home,
    Left,
    MouseLeft,
    MouseMiddle,
    MouseMove,
    MouseRight,
    MouseWheelDown,
    MouseWheelUp,
    PageDown,
    PageUp,
    Right,
    Tab,
    Up,
}

impl KeyName {
    pub fn is_mouse(&self) -> bool {
        use KeyName::*;
        matches!(
            self,
            MouseLeft | MouseMiddle | MouseRight | MouseMove | MouseWheelUp | MouseWheelDown
        )
    }
}

impl fmt::Debug for KeyName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KeyName::Backspace => write!(f, "backspace"),
            KeyName::Char(c) => match c {
                ' ' => write!(f, "space"),
                '\t' => write!(f, "tab"),
                '\n' => write!(f, "enter"),
                'a'..='z' | '0'..='9' => write!(f, "{}", c),
                '`' | '-' | '=' | '[' | ']' | '\\' | ';' | ',' | '.' | '/' => write!(f, "{}", c),
                _ => write!(f, "\"{}\"", c),
            },
            KeyName::Delete => write!(f, "delete"),
            KeyName::Insert => write!(f, "insert"),
            KeyName::Down => write!(f, "down"),
            KeyName::End => write!(f, "end"),
            KeyName::Enter => write!(f, "enter"),
            KeyName::Esc => write!(f, "esc"),
            KeyName::F(index) => write!(f, "f{}", index),
            KeyName::Home => write!(f, "home"),
            KeyName::Left => write!(f, "left"),
            KeyName::MouseLeft => write!(f, "mouseleft"),
            KeyName::MouseMiddle => write!(f, "mousemiddle"),
            KeyName::MouseMove => write!(f, "mousemove"),
            KeyName::MouseRight => write!(f, "mouseright"),
            KeyName::MouseWheelDown => write!(f, "mousewheeldown"),
            KeyName::MouseWheelUp => write!(f, "mousewheelup"),
            KeyName::PageDown => write!(f, "pagedown"),
            KeyName::PageUp => write!(f, "pageup"),
            KeyName::Right => write!(f, "right"),
            KeyName::Tab => write!(f, "tab"),
            KeyName::Up => write!(f, "up"),
        }
    }
}

impl fmt::Display for KeyName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl FromStr for KeyName {
    type Err = Error;

    #[allow(clippy::match_str_case_mismatch)]
    fn from_str(string: &str) -> Result<Self, Self::Err> {
        let key = match string.to_lowercase().as_ref() {
            "left" => KeyName::Left,
            "up" => KeyName::Up,
            "right" => KeyName::Right,
            "down" => KeyName::Down,
            "pageup" => KeyName::PageUp,
            "pagedown" => KeyName::PageDown,
            "end" => KeyName::End,
            "home" => KeyName::Home,
            "tab" => KeyName::Tab,
            "enter" => KeyName::Enter,
            "escape" => KeyName::Esc,
            "esc" => KeyName::Esc,
            "space" => KeyName::Char(' '),
            "backspace" => KeyName::Backspace,
            "delete" => KeyName::Delete,
            "insert" => KeyName::Insert,
            f if f.starts_with('f')
                && f.len() > 1
                && string[1..].chars().all(|c| c.is_ascii_digit()) =>
            {
                let index = string[1..].parse().expect("coding error");
                KeyName::F(index)
            }
            cs if cs.chars().count() == 1 => {
                let c = cs.chars().next().unwrap();
                match c {
                    c @ 'a'..='z' | c @ '0'..='9' => KeyName::Char(c),
                    '`' | '-' | '=' | '[' | ']' | '\\' | ';' | ',' | '.' | '/' => KeyName::Char(c),
                    _ => return Err(Error::ParseError("KeyName", string.to_string())),
                }
            }
            _ => return Err(Error::ParseError("KeyName", string.to_string())),
        };
        Ok(key)
    }
}

/// Key mode object
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KeyMod {
    bits: u32,
}

impl KeyMod {
    // order of bits is significant used by TTYDecoder
    pub const EMPTY: Self = KeyMod { bits: 0 };
    pub const SHIFT: Self = KeyMod { bits: 1 };
    pub const ALT: Self = KeyMod { bits: 2 };
    pub const CTRL: Self = KeyMod { bits: 4 };
    pub const SUPER: Self = KeyMod { bits: 8 };
    pub const HYPER: Self = KeyMod { bits: 16 };
    pub const META: Self = KeyMod { bits: 32 };
    pub const CAPSLOCK: Self = KeyMod { bits: 64 };
    pub const NUMLOCK: Self = KeyMod { bits: 128 };
    pub const PRESS: Self = KeyMod { bits: 256 };
    pub const ALL: Self = KeyMod { bits: 511 };

    /// Key mode is empty
    pub fn is_empty(self) -> bool {
        self == Self::EMPTY
    }

    /// Contains specified mod
    pub fn contains(self, other: Self) -> bool {
        self.bits & other.bits == other.bits
    }

    /// Create mod from byte
    pub fn from_bits(bits: u32) -> Self {
        Self {
            bits: bits & Self::ALL.bits,
        }
    }
}

impl std::ops::BitOr for KeyMod {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self {
            bits: self.bits | rhs.bits,
        }
    }
}

impl std::ops::BitOrAssign for KeyMod {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl fmt::Debug for KeyMod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            write!(f, "None")?;
        } else {
            let mut first = true;
            for (flag, name) in &[
                (Self::SHIFT, "shift"),
                (Self::ALT, "alt"),
                (Self::CTRL, "ctrl"),
                (Self::SUPER, "super"),
                (Self::HYPER, "hyper"),
                (Self::META, "meta"),
                (Self::PRESS, "press"),
                (Self::CAPSLOCK, "capslock"),
            ] {
                if self.contains(*flag) {
                    if first {
                        first = false;
                        write!(f, "{}", name)?;
                    } else {
                        write!(f, "+{}", name)?;
                    }
                }
            }
        }
        Ok(())
    }
}

impl fmt::Display for KeyMod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Result returned by KeyMap lookup
#[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum KeyMapResult<V> {
    /// Key sequence is bound to value V
    Success(V),
    /// Sequence is not bound to anything
    Failure,
    /// More key presses is required to determine if it is bound
    Continue,
}

/// Collection of key bindings
pub struct KeyMap<V> {
    mapping: BTreeMap<Key, Result<V, KeyMap<V>>>,
}

impl<V> Default for KeyMap<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V> KeyMap<V> {
    /// Create empty key binding collection
    pub fn new() -> Self {
        Self {
            mapping: Default::default(),
        }
    }

    /// Register key chord to produce the value.
    ///
    /// Returns previously registered value or key_map associated with provided chord.
    pub fn register(&mut self, chord: impl AsRef<[Key]>, value: V) -> Option<Result<V, KeyMap<V>>> {
        fn register_rec<'a, V>(key_map: &'a mut KeyMap<V>, chord: &[Key]) -> &'a mut KeyMap<V> {
            match chord.split_first() {
                None => key_map,
                Some((key, chord)) => {
                    let next = key_map
                        .mapping
                        .entry(*key)
                        .and_modify(|r| {
                            if r.is_ok() {
                                *r = Err(KeyMap::new())
                            }
                        })
                        .or_insert_with(|| Err(KeyMap::new()))
                        .as_mut()
                        .err()
                        .unwrap();
                    register_rec(next, chord)
                }
            }
        }

        match chord.as_ref().split_last() {
            Some((key, chord)) => register_rec(self, chord).mapping.insert(*key, Ok(value)),
            None => None,
        }
    }

    /// Override mapping with the other
    pub fn register_override(&mut self, other: &Self)
    where
        V: Clone,
    {
        other.for_each(|chord, value| {
            self.register(chord, value.clone());
        })
    }

    /// Run closure for each chord bound
    pub fn for_each(&self, mut f: impl FnMut(&'_ [Key], &'_ V)) {
        fn for_each_rec<V, F>(chord: &mut Vec<Key>, map: &KeyMap<V>, f: &mut F)
        where
            F: FnMut(&[Key], &V),
        {
            for (key, entry) in map.mapping.iter() {
                chord.push(*key);
                match entry {
                    Ok(value) => f(chord.as_slice(), value),
                    Err(map) => for_each_rec(chord, map, f),
                }
                chord.pop();
            }
        }
        for_each_rec(&mut Vec::new(), self, &mut f)
    }

    /// Lookup value given full chord path
    pub fn lookup(&self, chord: &[Key]) -> KeyMapResult<&V> {
        let result = chord
            .iter()
            .enumerate()
            .try_fold(&self.mapping, |mapping, (index, key)| {
                match mapping.get(key) {
                    None => Err(None),
                    Some(Err(mapping)) => Ok(&mapping.mapping),
                    Some(Ok(value)) => Err(Some((index, value))),
                }
            });
        match result {
            Err(Some((index, value))) => {
                if chord.len() == index + 1 {
                    KeyMapResult::Success(value)
                } else {
                    KeyMapResult::Failure
                }
            }
            Err(None) => KeyMapResult::Failure,
            _ => KeyMapResult::Continue,
        }
    }

    /// The a helper function which manges provided chord state for your
    pub fn lookup_state(&self, chord: &mut Vec<Key>, key: Key) -> Option<&V> {
        chord.push(key);
        for _ in 0..2 {
            match self.lookup(chord.as_ref()) {
                KeyMapResult::Continue => return None,
                KeyMapResult::Failure => {
                    chord.clear();
                    chord.push(key);
                }
                KeyMapResult::Success(value) => {
                    chord.clear();
                    return Some(value);
                }
            }
        }
        None
    }
}

impl<V: Debug> fmt::Debug for KeyMap<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut map = f.debug_map();
        let mut chord_str = String::new();
        self.for_each(|chord, value| {
            chord_str.clear();
            for key in chord.iter() {
                write!(chord_str, "{} ", key).expect("in memory wirte failed");
            }
            chord_str.pop();
            map.entry(&chord_str, value);
        });
        map.finish()?;
        Ok(())
    }
}

pub type KeyHandler<O> = Arc<dyn Fn(&[Key]) -> O>;

pub struct KeyMapHandler<O> {
    keymap: KeyMap<KeyHandler<O>>,
    state: Vec<Key>,
}

impl<O> Default for KeyMapHandler<O> {
    fn default() -> Self {
        Self::new()
    }
}

impl<O> KeyMapHandler<O> {
    pub fn new() -> Self {
        Self {
            keymap: Default::default(),
            state: Default::default(),
        }
    }

    pub fn register(&mut self, chrod: &[Key], handler: KeyHandler<O>) {
        self.keymap.register(chrod, handler);
    }

    pub fn handle(&mut self, key: Key) -> Option<O> {
        self.state.push(key);
        for _ in 0..2 {
            match self.keymap.lookup(self.state.as_ref()) {
                KeyMapResult::Continue => return None,
                KeyMapResult::Failure => {
                    self.state.clear();
                    self.state.push(key);
                }
                KeyMapResult::Success(handler) => {
                    let handler_result = (handler)(&self.state);
                    self.state.clear();
                    return Some(handler_result);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn test_key_map() -> Result<(), Error> {
        let c0 = KeyChord::from_str("ctrl+x")?;
        let c1 = KeyChord::from_str("ctrl+x f")?;
        let c2 = KeyChord::from_str("ctrl+x a b")?;

        let mut key_map = KeyMap::new();

        key_map.register(c0.keys(), 0);
        assert_eq!(key_map.lookup(c0.keys()), KeyMapResult::Success(&0));
        assert_eq!(key_map.lookup(c1.keys()), KeyMapResult::Failure);

        key_map.register(c1.keys(), 1);
        key_map.register(c2.keys(), 2);
        assert_eq!(key_map.lookup(c0.keys()), KeyMapResult::Continue);
        assert_eq!(key_map.lookup(c1.keys()), KeyMapResult::Success(&1));
        assert_eq!(key_map.lookup(c2.keys()), KeyMapResult::Success(&2));
        Ok(())
    }

    #[test]
    fn test_key_map_handler() -> Result<(), Error> {
        let a = "a".parse()?;
        let b = "b".parse()?;
        let c = "c".parse()?;
        let d = "d".parse()?;

        let events = Arc::new(Mutex::new(BTreeMap::new()));
        let count = Arc::new({
            let events = events.clone();
            move |chord: &[Key]| {
                let mut events = events.lock().unwrap();
                *events.entry(Vec::from(chord)).or_insert(0) += 1;
            }
        });
        let mut handler = KeyMapHandler::new();
        handler.register(&[a], count.clone());
        handler.register(&[b], count.clone());
        handler.register(&[c, d], count.clone());

        handler.handle(a);
        handler.handle(b);
        handler.handle(c);
        handler.handle(a);
        handler.handle(c);
        handler.handle(d);

        let reference: BTreeMap<Vec<Key>, usize> =
            vec![(vec![a], 2), (vec![b], 1), (vec![c, d], 1)]
                .into_iter()
                .collect();
        assert_eq!(*events.lock().unwrap(), reference);

        Ok(())
    }
}
