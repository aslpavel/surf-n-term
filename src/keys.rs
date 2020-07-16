use crate::Error;
use std::{collections::HashMap, fmt, str::FromStr};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key {
    pub name: KeyName,
    pub mode: KeyMod,
}

impl Key {
    pub fn new(name: KeyName, mode: KeyMod) -> Self {
        Self { name, mode }
    }

    pub fn chord(keys: impl AsRef<str>) -> Result<Vec<Key>, Error> {
        let chord = keys
            .as_ref()
            .split(' ')
            .filter(|k| !k.is_empty())
            .map(Key::from_str)
            .collect::<Result<Vec<Key>, Error>>()?;
        if chord.is_empty() {
            Err(Error::ParseError("Key", keys.as_ref().to_string()))
        } else {
            Ok(chord)
        }
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

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KeyName {
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    Backspace,
    Char(char),
    Delete,
    Down,
    End,
    Esc,
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
    Up,
}

impl fmt::Debug for KeyName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KeyName::F1 => write!(f, "f1"),
            KeyName::F2 => write!(f, "f2"),
            KeyName::F3 => write!(f, "f3"),
            KeyName::F4 => write!(f, "f4"),
            KeyName::F5 => write!(f, "f5"),
            KeyName::F6 => write!(f, "f6"),
            KeyName::F7 => write!(f, "f7"),
            KeyName::F8 => write!(f, "f8"),
            KeyName::F9 => write!(f, "f9"),
            KeyName::F10 => write!(f, "f10"),
            KeyName::F11 => write!(f, "f11"),
            KeyName::F12 => write!(f, "f12"),
            KeyName::Left => write!(f, "left"),
            KeyName::Right => write!(f, "right"),
            KeyName::Down => write!(f, "down"),
            KeyName::Up => write!(f, "up"),
            KeyName::PageUp => write!(f, "pageup"),
            KeyName::PageDown => write!(f, "pagedown"),
            KeyName::End => write!(f, "end"),
            KeyName::Home => write!(f, "home"),
            KeyName::Esc => write!(f, "esc"),
            KeyName::Backspace => write!(f, "backspace"),
            KeyName::Delete => write!(f, "delete"),
            KeyName::MouseLeft => write!(f, "mouseleft"),
            KeyName::MouseMiddle => write!(f, "mousemiddle"),
            KeyName::MouseMove => write!(f, "mousemove"),
            KeyName::MouseRight => write!(f, "mouseright"),
            KeyName::MouseWheelDown => write!(f, "mousewheeldown"),
            KeyName::MouseWheelUp => write!(f, "mousewheelup"),
            KeyName::Char(c) => match c {
                ' ' => write!(f, "space"),
                '\t' => write!(f, "tab"),
                '\n' => write!(f, "enter"),
                'a'..='z' | '0'..='9' => write!(f, "{}", c),
                '`' | '-' | '=' | '[' | ']' | '\\' | ';' | ',' | '.' | '/' => write!(f, "{}", c),
                _ => write!(f, "\"{}\"", c),
            },
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

    fn from_str(string: &str) -> Result<Self, Self::Err> {
        let key = match string.to_lowercase().as_ref() {
            "f1" => KeyName::F1,
            "f2" => KeyName::F2,
            "f3" => KeyName::F3,
            "f4" => KeyName::F4,
            "f5" => KeyName::F5,
            "f6" => KeyName::F6,
            "f7" => KeyName::F7,
            "f8" => KeyName::F8,
            "f9" => KeyName::F9,
            "f10" => KeyName::F10,
            "f11" => KeyName::F11,
            "f12" => KeyName::F12,
            "left" => KeyName::Left,
            "up" => KeyName::Up,
            "right" => KeyName::Right,
            "down" => KeyName::Down,
            "pageup" => KeyName::PageUp,
            "pagedown" => KeyName::PageDown,
            "end" => KeyName::End,
            "home" => KeyName::Home,
            "tab" => KeyName::Char('\t'),
            "enter" => KeyName::Char('\n'),
            "escape" => KeyName::Esc,
            "space" => KeyName::Char(' '),
            "backspace" => KeyName::Backspace,
            "delete" => KeyName::Delete,
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

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KeyMod {
    bits: u8,
}

impl KeyMod {
    // order of bits is significant used by TTYDecoder
    pub const EMPTY: Self = KeyMod { bits: 0 };
    pub const SHIFT: Self = KeyMod { bits: 1 };
    pub const ALT: Self = KeyMod { bits: 2 };
    pub const CTRL: Self = KeyMod { bits: 4 };
    pub const PRESS: Self = KeyMod { bits: 8 };

    pub fn is_empty(self) -> bool {
        self == Self::EMPTY
    }

    pub fn contains(self, other: Self) -> bool {
        self.bits & other.bits == other.bits
    }

    pub fn from_bits(bits: u8) -> Self {
        Self { bits }
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
                (Self::ALT, "alt"),
                (Self::CTRL, "ctrl"),
                (Self::SHIFT, "shift"),
                (Self::PRESS, "press"),
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

#[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum KeyMapResult<V> {
    Success(V),
    Failure,
    Continue,
}

pub struct KeyMap<V> {
    mapping: HashMap<Key, Result<V, KeyMap<V>>>,
}

impl<V> Default for KeyMap<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V> KeyMap<V> {
    pub fn new() -> Self {
        Self {
            mapping: Default::default(),
        }
    }

    pub fn register(&mut self, chord: &[Key], value: V) {
        if let Some((key, chord)) = chord.split_last() {
            self.register_rec(chord).mapping.insert(*key, Ok(value));
        }
    }

    fn register_rec(&mut self, chord: &[Key]) -> &mut Self {
        match chord.split_first() {
            None => self,
            Some((key, chord)) => {
                let next = self
                    .mapping
                    .entry(*key)
                    .and_modify(|r| {
                        if r.is_ok() {
                            *r = Err(Self::new())
                        }
                    })
                    .or_insert_with(|| Err(Self::new()))
                    .as_mut()
                    .err()
                    .unwrap();
                next.register_rec(chord)
            }
        }
    }

    pub fn lookup(&self, chord: &[Key]) -> KeyMapResult<&V> {
        let result = chord
            .iter()
            .enumerate()
            .try_fold(&self.mapping, |mapping, (index, key)| {
                match mapping.get(&key) {
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

    pub fn lookup_state(&self, chord: &mut Vec<Key>, key: Key) -> Option<&V> {
        chord.push(key);
        match self.lookup(chord.as_ref()) {
            KeyMapResult::Continue => None,
            KeyMapResult::Failure => {
                chord.clear();
                None
            }
            KeyMapResult::Success(value) => {
                chord.clear();
                Some(value)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_map() -> Result<(), Error> {
        let c0 = Key::chord("ctrl+x")?;
        let c1 = Key::chord("ctrl+x f")?;
        let c2 = Key::chord("ctrl+x a b")?;

        let mut key_map = KeyMap::new();

        key_map.register(c0.as_ref(), 0);
        assert_eq!(key_map.lookup(c0.as_ref()), KeyMapResult::Success(&0));
        assert_eq!(key_map.lookup(c1.as_ref()), KeyMapResult::Failure);

        key_map.register(c1.as_ref(), 1);
        key_map.register(c2.as_ref(), 2);
        assert_eq!(key_map.lookup(c0.as_ref()), KeyMapResult::Continue);
        assert_eq!(key_map.lookup(c1.as_ref()), KeyMapResult::Success(&1));
        assert_eq!(key_map.lookup(c2.as_ref()), KeyMapResult::Success(&2));

        Ok(())
    }
}
