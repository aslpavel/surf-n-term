use rasterize::{RGBA, SVG_COLORS};
use serde::{
    de::{self, DeserializeSeed},
    Deserialize, Deserializer,
};

use super::{BoxConstraint, Layout, View, ViewContext, ViewLayout, ViewMutLayout};
use crate::{
    glyph::GlyphDeserializer, surface::ViewBounds, Cell, Error, Face, FaceDeserializer, Glyph,
    Position, Size, TerminalSurface, TerminalSurfaceExt,
};
use std::{collections::HashMap, fmt::Write as _};

impl View for str {
    fn render(
        &self,
        ctx: &ViewContext,
        surf: TerminalSurface<'_>,
        layout: ViewLayout<'_>,
    ) -> Result<(), Error> {
        let mut surf = layout.apply_to(surf);
        let mut writer = surf.writer(ctx);
        self.chars().for_each(|c| {
            writer.put_char(c, Face::default());
        });
        Ok(())
    }

    fn layout(
        &self,
        ctx: &ViewContext,
        ct: BoxConstraint,
        mut layout: ViewMutLayout<'_>,
    ) -> Result<(), Error> {
        let mut size = Size::empty();
        let mut cursor = Position::origin();
        let face = Face::default();
        self.chars().for_each(|c| {
            Cell::new_char(face, c).layout(ctx, ct.max.width, true, &mut size, &mut cursor);
        });
        *layout = Layout::new().with_size(ct.clamp(size));
        Ok(())
    }
}

impl View for String {
    fn render(
        &self,
        ctx: &ViewContext,
        surf: TerminalSurface<'_>,
        layout: ViewLayout<'_>,
    ) -> Result<(), Error> {
        self.as_str().render(ctx, surf, layout)
    }

    fn layout(
        &self,
        ctx: &ViewContext,
        ct: BoxConstraint,
        layout: ViewMutLayout<'_>,
    ) -> Result<(), Error> {
        self.as_str().layout(ctx, ct, layout)
    }
}

#[derive(Clone, Default, Debug)]
pub struct Text {
    cells: Vec<Cell>,
    wraps: bool,
    // face used write next symbol (not actual face of the text)
    face: Face,
}

impl Text {
    /// Create new empty text
    pub fn new() -> Self {
        Self {
            cells: Vec::default(),
            wraps: true,
            face: Face::default(),
        }
    }

    /// Number of cells
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// Check if [Text] is empty
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Remove all cell and reset face
    pub fn clear(&mut self) {
        self.cells.clear();
        self.face = Face::default();
    }

    /// Get current value of wraps flag
    pub fn wraps(&self) -> bool {
        self.wraps
    }

    /// Create new surface with update wraps flag
    pub fn with_wraps(self, wraps: bool) -> Self {
        Self { wraps, ..self }
    }

    /// Set wraps flag
    pub fn set_wraps(&mut self, wraps: bool) -> &mut Self {
        self.wraps = wraps;
        self
    }

    /// Currently set face
    pub fn face(&self) -> &Face {
        &self.face
    }

    /// Overlay face for the length of the scope
    pub fn with_face(&mut self, face: Face, scope: impl FnOnce(&mut Self)) -> &mut Self {
        let face_old = self.face;
        scope(self.set_face(face_old.overlay(&face)));
        self.set_face(face_old);
        self
    }

    /// Set face that will to add new cells
    pub fn set_face(&mut self, face: Face) -> &mut Self {
        self.face = face;
        self
    }

    /// Append character to the end of the [Text]
    pub fn put_char(&mut self, c: char) -> &mut Self {
        self.cells.push(Cell::new_char(self.face, c));
        self
    }

    /// Append a given string to the end of [Text]
    pub fn push_str(&mut self, str: &str, face: Option<Face>) -> &mut Self {
        self.with_face(face.unwrap_or_default(), |text| {
            str.chars().for_each(|c| {
                text.put_char(c);
            })
        });
        self
    }

    /// Append glyph to the end of the [Text]
    pub fn put_glyph(&mut self, glyph: Glyph) -> &mut Self {
        self.cells.push(Cell::new_glyph(self.face, glyph));
        self
    }

    /// Put cell
    pub fn put_cell(&mut self, cell: Cell) -> &mut Self {
        let face = self.face.overlay(&cell.face());
        self.cells.push(cell.with_face(face));
        self
    }

    /// Append text to the end of the [Text]
    pub fn push_text(&mut self, text: &Text) -> &mut Self {
        text.cells.iter().cloned().for_each(|cell| {
            self.put_cell(cell);
        });
        self
    }

    /// Append format argument to the end of the [Text]
    pub fn push_fmt<T>(&mut self, value: &T) -> &mut Self
    where
        T: std::fmt::Display + ?Sized,
    {
        write!(self, "{}", value).expect("in memory write failed");
        self
    }

    /// Mark range of cell with the provided face
    pub fn mark(&mut self, face: Face, bounds: impl ViewBounds) -> &mut Self {
        if let Some((start, end)) = bounds.view_bounds(self.cells.len()) {
            self.cells[start..end].iter_mut().for_each(|cell| {
                *cell = cell.clone().with_face(cell.face().overlay(&face));
            })
        }
        self
    }

    /// Take current value replacing it with the default
    pub fn take(&mut self) -> Self {
        std::mem::replace(self, Text::new())
    }
}

impl std::fmt::Write for Text {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        s.chars().for_each(|c| {
            self.put_char(c);
        });
        Ok(())
    }
}

impl<'a> From<&'a str> for Text {
    fn from(value: &'a str) -> Self {
        let mut span = Text::new();
        span.write_str(value).expect("memory write failed");
        span
    }
}

impl View for Text {
    fn render(
        &self,
        ctx: &ViewContext,
        surf: TerminalSurface<'_>,
        layout: ViewLayout<'_>,
    ) -> Result<(), Error> {
        let mut surf = layout.apply_to(surf);
        let mut writer = surf.writer(ctx).with_wraps(self.wraps);
        self.cells.iter().for_each(|cell| {
            writer.put(cell.clone());
        });
        Ok(())
    }

    fn layout(
        &self,
        ctx: &ViewContext,
        ct: BoxConstraint,
        mut layout: ViewMutLayout<'_>,
    ) -> Result<(), Error> {
        let mut size = Size::empty();
        let mut cursor = Position::origin();
        self.cells.iter().for_each(|cell| {
            cell.layout(ctx, ct.max.width, self.wraps, &mut size, &mut cursor);
        });
        *layout = Layout::new().with_size(ct.clamp(size));
        Ok(())
    }
}

/// Deserializer for [Text]
///
/// Text deserialized from the following recursive definition:
/// `Text = String | [Text] | { face: Face, text: Text, glyph: Glyph }`
pub struct TextDeserializer<'a> {
    pub colors: &'a HashMap<String, RGBA>,
}

impl<'de, 'a> DeserializeSeed<'de> for TextDeserializer<'a> {
    type Value = Text;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde_json::Value;

        fn collect_rec(
            text: &mut Text,
            colors: &HashMap<String, RGBA>,
            value: &Value,
        ) -> Result<(), Error> {
            match value {
                Value::String(string) => {
                    text.push_str(string, None);
                }
                Value::Object(map) => {
                    let face = match map.get("face") {
                        Some(face) => FaceDeserializer { colors }.deserialize(face)?,
                        None => Face::default(),
                    };
                    if let Some(wraps) = map.get("wraps").and_then(|w| w.as_bool()) {
                        text.wraps = wraps;
                    }
                    let face_old = text.face;
                    text.set_face(face_old.overlay(&face));
                    if let Some(glyph) = map.get("glyph") {
                        text.put_glyph(GlyphDeserializer { colors }.deserialize(glyph)?);
                    } else if let Some(text_value) = map.get("text") {
                        collect_rec(text, colors, text_value)?;
                    }

                    text.face = face_old;
                }
                Value::Array(text_values) => {
                    for text_value in text_values {
                        collect_rec(text, colors, text_value)?;
                    }
                }
                _ => {
                    return Err(Error::ParseError(
                        "Text",
                        "expected: {map|list|str}".to_owned(),
                    ))
                }
            }
            Ok(())
        }
        let mut text = Text::new();
        collect_rec(&mut text, self.colors, &Value::deserialize(deserializer)?)
            .map_err(de::Error::custom)?;
        Ok(text)
    }
}

impl<'de> Deserialize<'de> for Text {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        TextDeserializer {
            colors: &SVG_COLORS,
        }
        .deserialize(deserializer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::ViewLayoutStore;

    #[test]
    fn test_text_basic() -> Result<(), Error> {
        let tag = "[text basic]";
        let size = Size::new(5, 10);
        let ctx = ViewContext::dummy();
        let mut text = Text::new().set_face("fg=#ebdbb2".parse()?).take();
        let mut layout_store = ViewLayoutStore::new();

        writeln!(&mut text, "11")?;
        text.set_face("fg=#3c3836,bg=#ebdbb2".parse()?);
        writeln!(&mut text, "222")?;

        print!("{tag} first line longest: {:?}", text.debug(size));
        let layout = text.layout_new(&ctx, BoxConstraint::loose(size), &mut layout_store)?;
        assert_eq!(layout.size(), Size::new(2, 3));

        text.set_face(text.face().overlay(&"fg=#af3a03,bold".parse()?));
        writeln!(&mut text, "3")?;

        print!("{tag} middle line longest: {:?}", text.debug(size));
        let layout = text.layout_new(&ctx, BoxConstraint::loose(size), &mut layout_store)?;
        assert_eq!(layout.size(), Size::new(3, 3));

        Ok(())
    }

    #[test]
    fn test_text_wrap() -> Result<(), Error> {
        let tag = "[text wrap]";
        let size = Size::new(5, 10);
        let ctx = ViewContext::dummy();
        let mut text = Text::new();
        let mut layout_store = ViewLayoutStore::new();

        text.set_face("fg=gruv-red-2".parse()?);
        write!(&mut text, "1 no wrap")?;
        print!("{tag} will not wrap: {:?}", text.debug(size));
        let layout = text.layout_new(&ctx, BoxConstraint::loose(size), &mut layout_store)?;
        assert_eq!(layout.size(), Size::new(1, 9));

        text.set_face("fg=gruv-green-2".parse()?);
        writeln!(&mut text, "\n2 this will wrap")?;
        print!("{tag} will wrap: {:?}", text.debug(size));
        let layout = text.layout_new(&ctx, BoxConstraint::loose(size), &mut layout_store)?;
        assert_eq!(layout.size(), Size::new(3, 10));

        text.set_face("fg=gruv-blue-2".parse()?);
        write!(&mut text, "3 bla")?;
        print!("{tag} one more line: {:?}", text.debug(size));
        let layout = text.layout_new(&ctx, BoxConstraint::loose(size), &mut layout_store)?;
        assert_eq!(layout.size(), Size::new(4, 10));

        text.set_wraps(false);
        print!("{tag} disable wrapping: {:?}", text.debug(size));
        let layout = text.layout_new(&ctx, BoxConstraint::loose(size), &mut layout_store)?;
        assert_eq!(layout.size(), Size::new(3, 10));

        Ok(())
    }

    #[test]
    fn test_text_tab() -> Result<(), Error> {
        let tag = "[text tab]";
        let size = Size::new(6, 20);
        let ctx = ViewContext::dummy();
        let mut text = Text::new().set_face("fg=#ebdbb2".parse()?).take();
        let mut layout_store = ViewLayoutStore::new();

        writeln!(&mut text, ".\t|")?;
        print!("{tag} 1 char: {:?}", text.debug(size));
        let layout = text.layout_new(&ctx, BoxConstraint::loose(size), &mut layout_store)?;
        assert_eq!(layout.size(), Size::new(1, 9));

        writeln!(&mut text, ".......\t|")?;
        print!("{tag} 7 chars: {:?}", text.debug(size));
        let layout = text.layout_new(&ctx, BoxConstraint::loose(size), &mut layout_store)?;
        assert_eq!(layout.size(), Size::new(2, 9));

        writeln!(&mut text, "........\t|")?;
        print!("{tag} 8 chars: {:?}", text.debug(size));
        let layout = text.layout_new(&ctx, BoxConstraint::loose(size), &mut layout_store)?;
        assert_eq!(layout.size(), Size::new(3, 17));

        writeln!(&mut text, "...............\t|")?;
        print!("{tag} 15 chars: {:?}", text.debug(size));
        let layout = text.layout_new(&ctx, BoxConstraint::loose(size), &mut layout_store)?;
        assert_eq!(layout.size(), Size::new(4, 17));

        writeln!(&mut text, "................\t|")?;
        print!("{tag} 16 chars: {:?}", text.debug(size));
        let layout = text.layout_new(&ctx, BoxConstraint::loose(size), &mut layout_store)?;
        assert_eq!(layout.size(), Size::new(6, 20));

        Ok(())
    }

    #[test]
    fn test_text_serde() -> Result<(), Error> {
        use serde_json::json;

        let face0_str = "fg=black,bg=#ff0000";
        let face1_str = "fg=black,bg=#00ff00/.5";
        let text_json = json!({
            "face": face0_str,
            "text": [
                "a",
                {
                    "text": "b",
                    "face": face1_str, // overlay
                },
                "c" // face must be restored
            ]
        });

        let text = Text::deserialize(text_json)?;
        print!(
            "[text serde] nested text and faces: {:?}",
            text.debug(Size::new(1, 3))
        );

        let face0 = face0_str.parse()?;
        let face1 = face1_str.parse()?;
        assert_eq!(
            text.cells,
            vec![
                Cell::new_char(face0, 'a'),
                Cell::new_char(face0.overlay(&face1), 'b'),
                Cell::new_char(face0, 'c'),
            ]
        );

        Ok(())
    }
}
