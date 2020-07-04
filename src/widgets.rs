use crate::{
    common::clamp, Blend, Cell, Color, Error, Face, FaceAttrs, Key, KeyMod, KeyName, SurfaceMut,
    TerminalEvent, TerminalSurfaceExt, TerminalWritable, RGBA,
};
use std::{io::Write, str::FromStr};

#[derive(Clone, Debug)]
pub struct Theme {
    pub fg: RGBA,
    pub bg: RGBA,
    pub accent: RGBA,
    pub cursor: Face,
    pub input: Face,
    pub list_default: Face,
    pub list_selected: Face,
    pub scrollbar_on: Face,
    pub scrollbar_off: Face,
}

impl Theme {
    pub fn from_palette(fg: RGBA, bg: RGBA, accent: RGBA) -> Self {
        let cursor = Face::new(
            Some(bg),
            Some(bg.blend(accent.with_alpha(0.8), Blend::Over)),
            FaceAttrs::EMPTY,
        );
        let input = Face::new(Some(fg), Some(bg), FaceAttrs::EMPTY);
        let list_default = Face::new(
            Some(bg.blend(fg.with_alpha(0.9), Blend::Over)),
            Some(bg),
            FaceAttrs::EMPTY,
        );
        let list_selected = Face::new(
            Some(bg.blend(fg.with_alpha(0.9), Blend::Over)),
            Some(bg.blend(fg.with_alpha(0.1), Blend::Over)),
            FaceAttrs::EMPTY,
        );
        let scrollbar_on = Face::new(None, Some(accent.with_alpha(0.8)), FaceAttrs::EMPTY);
        let scrollbar_off = Face::new(None, Some(accent.with_alpha(0.5)), FaceAttrs::EMPTY);
        Self {
            fg,
            bg,
            accent,
            cursor,
            input,
            list_default,
            list_selected,
            scrollbar_on,
            scrollbar_off,
        }
    }

    pub fn light() -> Self {
        Self::from_palette(
            "#3c3836".parse().unwrap(),
            "#fbf1c7".parse().unwrap(),
            "#8f3f71".parse().unwrap(),
        )
    }

    pub fn dark() -> Self {
        Self::from_palette(
            "#ebdbb2".parse().unwrap(),
            "#282828".parse().unwrap(),
            "#d3869b".parse().unwrap(),
        )
    }
}

impl FromStr for Theme {
    type Err = Error;

    fn from_str(string: &str) -> Result<Self, Self::Err> {
        string.split(',').try_fold(Theme::light(), |theme, attrs| {
            let mut iter = attrs.splitn(2, '=');
            let key = iter.next().unwrap_or_default().trim().to_lowercase();
            let value = iter.next().unwrap_or_default().trim();
            let theme = match key.as_str() {
                "fg" => Theme::from_palette(value.parse()?, theme.bg, theme.accent),
                "bg" => Theme::from_palette(theme.fg, value.parse()?, theme.accent),
                "accent" | "base" => Theme::from_palette(theme.fg, theme.bg, value.parse()?),
                "light" => Theme::light(),
                "dark" => Theme::dark(),
                _ => return Err(Error::ParseFaceError),
            };
            Ok(theme)
        })
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
                    // move cursor backward
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

pub trait ListItems {
    type Item: TerminalWritable;
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> Option<Self::Item>;
}

pub struct List<T> {
    items: T,
    offset: i32,
    cursor: i32,
    height_hint: i32,
}

impl<T: ListItems> List<T> {
    pub fn new(items: T) -> Self {
        Self {
            items,
            offset: 0,
            cursor: 0,
            height_hint: 1,
        }
    }

    pub fn items(&self) -> &T {
        &self.items
    }

    pub fn items_set(&mut self, items: T) {
        self.offset = 0;
        self.cursor = 0;
        self.items = items
    }

    pub fn current(&self) -> Option<T::Item> {
        self.items.get(self.cursor as usize)
    }

    pub fn handle(&mut self, event: &TerminalEvent) {
        match *event {
            TerminalEvent::Key(Key { name, mode }) if mode == KeyMod::EMPTY => {
                if name == KeyName::Down {
                    self.cursor += 1;
                } else if name == KeyName::Up {
                    self.cursor -= 1;
                } else if name == KeyName::PageDown {
                    self.cursor += self.height_hint;
                } else if name == KeyName::PageUp {
                    self.cursor -= self.height_hint;
                }
                self.cursor = clamp(self.cursor, 0, self.items.len() as i32 - 1);
            }
            _ => (),
        }
    }

    pub fn render(
        &mut self,
        theme: &Theme,
        mut surf: impl SurfaceMut<Item = Cell>,
    ) -> Result<(), Error> {
        self.height_hint = surf.height() as i32;
        surf.erase(theme.list_default.bg);
        if surf.height() < 1 || surf.width() < 5 {
            return Ok(());
        }
        if self.offset > self.cursor {
            self.offset = self.cursor;
        } else if self.offset + surf.height() as i32 - 1 < self.cursor {
            self.offset = self.cursor - surf.height() as i32 + 1;
        }

        for row in 0..(surf.height() as i32) {
            let item = match self.items.get((self.offset + row) as usize) {
                Some(item) => item,
                None => break,
            };
            let mut line = surf.view_mut(row, ..-1);
            let mut writer = if row + self.offset == self.cursor {
                line.erase(theme.list_selected.bg);
                let mut writer = line
                    .writer()
                    .face(theme.list_selected.with_fg(Some(theme.accent)));
                writer.write_all(" ● ".as_ref())?;
                writer.face(theme.list_selected)
            } else {
                let mut writer = line.writer().face(theme.list_default);
                writer.write_all("   ".as_ref())?;
                writer
            };
            writer.display(item)?;
        }

        // scroll bar
        let (sb_offset, sb_filled) = if self.items.len() != 0 {
            let sb_filled = clamp(surf.height().pow(2) / self.items.len(), 1, surf.height()) as i32;
            let sb_offset =
                (surf.height() as i32 - sb_filled) * (self.cursor + 1) / self.items.len() as i32;
            (sb_offset, sb_filled + sb_offset)
        } else {
            (0, surf.height() as i32)
        };
        let range = 0..(surf.height() as i32);
        let mut sb = surf.view_mut(.., -1);
        let mut sb_writer = sb.writer();
        for i in range {
            if i < sb_offset || i >= sb_filled {
                sb_writer.put_char(' ', theme.scrollbar_off);
            } else {
                sb_writer.put_char(' ', theme.scrollbar_on);
            }
        }

        Ok(())
    }
}
