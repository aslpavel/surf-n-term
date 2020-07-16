use crate::Error;
use std::{fmt, str::FromStr};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key {
    pub name: KeyName,
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
