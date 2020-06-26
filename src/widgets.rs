use crate::{
    Cell, Error, Face, FaceAttrs, Key, KeyMod, KeyName, SurfaceMut, TerminalEvent,
    TerminalSurfaceExt, RGBA,
};

pub struct Theme {
    pub cursor: Face,
    pub input: Face,
}

impl Theme {
    pub fn from_palette(fg: RGBA, bg: RGBA, accent: RGBA) -> Self {
        Self {
            cursor: Face::new(Some(bg), Some(accent), FaceAttrs::EMPTY),
            input: Face::new(Some(fg), Some(bg), FaceAttrs::EMPTY),
        }
    }
}

pub struct Input {
    /// string before cursor
    before: Vec<char>,
    /// reversed string after cursor
    after: Vec<char>,
    /// visible offset
    offset: usize,
}

impl Input {
    pub fn new() -> Self {
        Self {
            before: Default::default(),
            after: Default::default(),
            offset: 0,
        }
    }

    pub fn handle(&mut self, event: &TerminalEvent) {
        match *event {
            TerminalEvent::Key(Key { name, mode }) if mode == KeyMod::EMPTY => {
                if let KeyName::Char(c) = name {
                    // insert char
                    self.before.push(c);
                } else if name == KeyName::Backspace {
                    // delete previous char
                    self.before.pop();
                } else if name == KeyName::Left {
                    // delete next char
                    self.after.extend(self.before.pop());
                } else if name == KeyName::Right {
                    // move cursor forward
                    self.before.extend(self.after.pop());
                } else if name == KeyName::Delete {
                    // move curosr backward
                    self.after.pop();
                }
            }
            TerminalEvent::Key(Key { name, mode }) if mode == KeyMod::CTRL => {
                if name == KeyName::Char('e') {
                    // move curosor to the end of input
                    self.before.extend(self.after.drain(..).rev());
                } else if name == KeyName::Char('a') {
                    // move cursor to the start of input
                    self.after.extend(self.before.drain(..).rev());
                } else if name == KeyName::Char('k') {
                    self.after.clear();
                }
            }
            TerminalEvent::Key(Key { name, mode }) if mode == KeyMod::ALT => {
                if name == KeyName::Char('f') {
                    // next word
                    while let Some(c) = self.after.pop() {
                        if is_word_separator(c) {
                            self.before.push(c);
                        } else {
                            self.after.push(c);
                            break;
                        }
                    }
                    while let Some(c) = self.after.pop() {
                        if is_word_separator(c) {
                            self.after.push(c);
                            break;
                        } else {
                            self.before.push(c);
                        }
                    }
                } else if name == KeyName::Char('b') {
                    // previous word
                    while let Some(c) = self.before.pop() {
                        if is_word_separator(c) {
                            self.after.push(c);
                        } else {
                            self.before.push(c);
                            break;
                        }
                    }
                    while let Some(c) = self.before.pop() {
                        if is_word_separator(c) {
                            self.before.push(c);
                            break;
                        } else {
                            self.after.push(c);
                        }
                    }
                }
            }
            _ => (),
        }
    }

    pub fn get(&self) -> impl Iterator<Item = char> + '_ {
        self.before.iter().chain(self.after.iter().rev()).copied()
    }

    pub fn set(&mut self, text: &str) {
        self.before.clear();
        self.after.clear();
        self.before.extend(text.chars());
        self.offset = 0;
    }

    pub fn render(
        &mut self,
        theme: &Theme,
        mut surf: impl SurfaceMut<Item = Cell>,
    ) -> Result<(), Error> {
        surf.erase(theme.input.bg);
        let size = surf.width() * surf.height();
        if size < 2 {
            return Ok(());
        } else if self.offset > self.before.len() {
            self.offset = self.before.len();
        } else if self.offset + size < self.before.len() + 1 {
            self.offset = self.before.len() - size + 1;
        }
        let mut writer = surf.writer().face(theme.input);
        for c in self.before[self.offset..].iter() {
            writer.put(Cell::new(theme.input, Some(*c)));
        }
        let mut iter = self.after.iter().rev();
        writer.put(Cell::new(theme.cursor, iter.next().copied()));
        for c in iter {
            writer.put(Cell::new(theme.input, Some(*c)));
        }
        Ok(())
    }
}

fn is_word_separator(c: char) -> bool {
    // TODO: extend word separator set
    match c {
        ' ' => true,
        _ => false,
    }
}
