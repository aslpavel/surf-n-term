use crate::{
    common::clamp, Blend, Cell, Color, Error, Face, FaceAttrs, Key, KeyMod, KeyName, Surface,
    SurfaceMut, TerminalEvent, TerminalSurfaceExt, TerminalWritable, RGBA,
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
        let cursor = {
            let cursor_bg = bg.blend(accent.with_alpha(0.8), Blend::Over);
            let cursor_fg = cursor_bg.best_contrast(bg, fg);
            Face::new(Some(cursor_fg), Some(cursor_bg), FaceAttrs::EMPTY)
        };
        let input = Face::new(Some(fg), Some(bg), FaceAttrs::EMPTY);
        let list_default = Face::new(
            Some(bg.blend(fg.with_alpha(0.8), Blend::Over)),
            Some(bg),
            FaceAttrs::EMPTY,
        );
        let list_selected = Face::new(
            Some(bg.blend(fg.with_alpha(0.8), Blend::Over)),
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
                _ => return Err(Error::ParseError("Theme", string.to_string())),
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

impl Default for Input {
    fn default() -> Self {
        Self::new()
    }
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
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct List<T> {
    items: T,
    offset: usize,
    cursor: usize,
    height_hint: usize,
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
        self.items.get(self.cursor)
    }

    pub fn handle(&mut self, event: &TerminalEvent) {
        match *event {
            TerminalEvent::Key(Key { name, mode }) if mode == KeyMod::EMPTY => {
                if name == KeyName::Down {
                    self.cursor += 1;
                } else if name == KeyName::Up && self.cursor > 0 {
                    self.cursor -= 1;
                } else if name == KeyName::PageDown {
                    self.cursor += self.height_hint;
                } else if name == KeyName::PageUp && self.cursor >= self.height_hint {
                    self.cursor -= self.height_hint;
                }
            }
            TerminalEvent::Key(Key { name, mode }) if mode == KeyMod::CTRL => {
                if name == KeyName::Char('n') {
                    self.cursor += 1;
                } else if name == KeyName::Char('p') && self.cursor > 0 {
                    self.cursor -= 1;
                }
            }
            _ => (),
        }
        if self.items.len() > 0 {
            self.cursor = clamp(self.cursor, 0, self.items.len() - 1);
        } else {
            self.cursor = 0;
        }
    }

    pub fn render(
        &mut self,
        theme: &Theme,
        mut surf: impl SurfaceMut<Item = Cell>,
    ) -> Result<(), Error> {
        surf.erase(theme.list_default.bg);
        if surf.height() < 1 || surf.width() < 5 {
            return Ok(());
        }
        if self.offset > self.cursor {
            self.offset = self.cursor;
        } else if self.offset + surf.height() - 1 < self.cursor {
            self.offset = self.cursor - surf.height() + 1;
        }

        // items
        let width = surf.width() - 4; // exclude left border and scroll bar
        let items: Vec<_> = (self.offset..self.offset + surf.height())
            .filter_map(|index| {
                let item = self.items.get(index)?;
                let height = item.length_hint().unwrap_or(0) / width + 1;
                Some((index, height, item))
            })
            .collect();
        // make sure items will fit
        let mut cursor_found = false;
        let mut items_height = 0;
        let mut first = 0;
        for (index, height, _item) in items.iter() {
            items_height += height;
            if items_height > surf.height() {
                if cursor_found {
                    break;
                }
                while items_height > surf.height() {
                    items_height -= items[first].1;
                    first += 1;
                }
            }
            cursor_found = cursor_found || *index == self.cursor;
        }
        self.height_hint = items.len();
        self.offset += first;
        // render items
        let mut row = 0;
        for (index, height, item) in items[first..].iter() {
            let mut item_surf = surf.view_mut(row..row + height, ..-1);
            row += height;
            if item_surf.is_empty() {
                break;
            }
            if *index == self.cursor {
                item_surf.erase(theme.list_selected.bg);
                let mut writer = item_surf
                    .writer()
                    .face(theme.list_selected.with_fg(Some(theme.accent)));
                writer.write_all(" ● ".as_ref())?;
            } else {
                let mut writer = item_surf.writer().face(theme.list_default);
                writer.write_all("   ".as_ref())?;
            };
            let mut text_surf = item_surf.view_mut(.., 3..);
            let writer = text_surf.writer();
            if *index == self.cursor {
                writer.face(theme.list_selected).display(item)?;
            } else {
                writer.face(theme.list_default).display(item)?;
            }
        }

        // scroll bar
        let (sb_offset, sb_filled) = if self.items.len() != 0 {
            let sb_filled = clamp(
                surf.height() * items.len() / self.items.len(),
                1,
                surf.height(),
            );
            let sb_offset = (surf.height() - sb_filled) * (self.cursor + 1) / self.items.len();
            (sb_offset, sb_filled + sb_offset)
        } else {
            (0, surf.height())
        };
        let range = 0..surf.height();
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
