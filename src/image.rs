//! Handling everything to do with images.
//!
//! Provides:
//!  - Protocol handling
//!  - Image object
//!  - Quantization and dithering
use crate::{
    common::{clamp, Rnd},
    decoder::Base64Decoder,
    encoder::{Base64Encoder, Encoder, TTYEncoder},
    surface::ViewBounds,
    view::{BoxConstraint, Layout, View, ViewContext, ViewLayout, ViewMutLayout},
    Cell, Color, Error, Face, FaceAttrs, Position, Shape, Size, Surface, SurfaceMut, SurfaceOwned,
    TerminalCommand, TerminalEvent, TerminalSurface, RGBA,
};
use serde::{
    de,
    ser::{self, SerializeStruct},
    Deserialize, Serialize, Serializer,
};
use std::{
    borrow::Cow,
    cmp::Ordering,
    collections::{hash_map::Entry, HashMap, HashSet},
    fmt,
    io::{Read, Write},
    ops::{Add, AddAssign, Mul},
    str::FromStr,
    sync::Arc,
};

const IMAGE_CACHE_SIZE: usize = 134217728; // 128MB

/// Arc wrapped RGBA surface with precomputed hash
#[derive(Clone)]
pub struct Image {
    data: Arc<[RGBA]>,
    shape: Shape,
}

impl Image {
    /// Create new image from the RGBA surface
    pub fn new(surf: impl Surface<Item = impl Color>) -> Self {
        let data: Vec<_> = surf
            .data()
            .iter()
            .map(|color| color.to_rgba().into())
            .collect();
        Self {
            data: data.into(),
            shape: surf.shape(),
        }
    }

    /// Create new image from parts
    pub fn from_parts(data: Arc<[RGBA]>, shape: Shape) -> Self {
        Self { data, shape }
    }

    /// Crop image
    pub fn crop<RS, CS>(&self, rows: RS, cols: CS) -> Self
    where
        RS: ViewBounds,
        CS: ViewBounds,
    {
        Self {
            data: self.data.clone(),
            shape: self.shape.view(rows, cols),
        }
    }

    /// Resize image using bi-linear interpolation
    pub fn resize(&self, size: Size) -> Self {
        if size.width <= 1 || size.height <= 1 || self.width() <= 2 || self.height() <= 2 {
            return Image::from(SurfaceOwned::new(size));
        }
        if self.size() == size {
            return self.clone();
        }

        let src_col_max = self.width() as f32 - 1.5;
        let src_row_max = self.height() as f32 - 1.5;
        let dst_col_max = size.width as f32 - 1.0;
        let dst_row_max = size.height as f32 - 1.0;

        let src_shape = self.shape();
        let src_data = self.data();

        let mut dst_data: Vec<RGBA> = Vec::new();
        dst_data.resize_with(size.width * size.height, Default::default);

        fn lerp_u8(left: u8, right: u8, t: f32) -> u8 {
            (left as f32 * (1.0 - t) + right as f32 * t) as u8
        }

        fn lerp_rgba(left: RGBA, right: RGBA, t: f32) -> RGBA {
            let [lr, lg, lb, la] = left.to_rgba();
            let [rr, rg, rb, ra] = right.to_rgba();
            RGBA::new(
                lerp_u8(lr, rr, t),
                lerp_u8(lg, rg, t),
                lerp_u8(lb, rb, t),
                lerp_u8(la, ra, t),
            )
        }

        for dst_row in 0..size.height {
            for dst_col in 0..size.width {
                let src_row_f = (dst_row as f32) * src_row_max / dst_row_max;
                let src_col_f = (dst_col as f32) * src_col_max / dst_col_max;
                let src_row = src_row_f.floor() as usize;
                let row_frac = src_row_f.fract();
                let src_col = src_col_f.floor() as usize;
                let col_frac = src_col_f.fract();

                let p00 = src_data[src_shape.offset(Position::new(src_row, src_col))];
                let p01 = src_data[src_shape.offset(Position::new(src_row, src_col + 1))];
                let p10 = src_data[src_shape.offset(Position::new(src_row + 1, src_col))];
                let p11 = src_data[src_shape.offset(Position::new(src_row + 1, src_col + 1))];

                // bi-linear interpolation
                // NOTE: not using Color.lerp as it converts to LinColor which increases
                //       conversion time x5 (see resize benchmark)
                let r0 = lerp_rgba(p00, p01, col_frac);
                let r1 = lerp_rgba(p10, p11, col_frac);
                let r = lerp_rgba(r0, r1, row_frac);

                dst_data[dst_row * size.width + dst_col] = r;
            }
        }

        Image::from_parts(dst_data.into(), Shape::from(size))
    }

    /// Size in cells
    pub fn size_cells(&self, pixels_per_cell: Size) -> Size {
        if pixels_per_cell.is_empty() || self.size().is_empty() {
            return Size::new(0, 0);
        }
        fn round_up(a: usize, b: usize) -> usize {
            let c = a / b;
            if a % b == 0 {
                c
            } else {
                c + 1
            }
        }
        Size {
            height: round_up(self.height(), pixels_per_cell.height),
            width: round_up(self.width(), pixels_per_cell.width),
        }
    }

    /// Quantize image
    ///
    /// Perform palette extraction and Floyd–Steinberg dithering.
    #[tracing::instrument(name = "[Image.quantize]", level = "debug")]
    pub fn quantize(
        &self,
        palette_size: usize,
        dither: bool,
        bg: Option<RGBA>,
    ) -> Option<(ColorPalette, SurfaceOwned<usize>)> {
        let bg = bg.unwrap_or_else(|| RGBA::new(0, 0, 0, 255));
        let palette = ColorPalette::from_image(self, palette_size, bg)?;
        let mut qimg = SurfaceOwned::new(self.size());

        // quantize and dither
        let mut errors: Vec<ColorError> = Vec::new();
        let ewidth = self.width() + 2; // to avoid check for the first and the last pixels
        if dither {
            errors.resize_with(ewidth * 2, ColorError::new);
        }
        for row in 0..self.height() {
            if dither {
                // swap error rows
                for col in 0..ewidth {
                    errors[col] = errors[col + ewidth];
                    errors[col + ewidth] = ColorError::new();
                }
            }
            // quantize and spread the error
            for col in 0..self.width() {
                let pos = Position::new(row, col);
                let mut color = *self.get(pos)?;
                if color.to_rgba()[3] < 255 {
                    color = bg.blend_over(color);
                }
                if dither {
                    color = errors[col + 1].add(color); // account for error
                }
                let (qindex, qcolor) = palette.find(color);
                qimg.set(pos, qindex);
                if dither {
                    // spread the error according to Floyd–Steinberg dithering matrix:
                    // [[0   , X   , 7/16],
                    // [3/16, 5/16, 1/16]]
                    let error = ColorError::between(color, qcolor);
                    errors[col + 2] += error * 0.4375; // 7/16
                    errors[col + ewidth] += error * 0.1875; // 3/16
                    errors[col + ewidth + 1] += error * 0.3125; // 5/16
                    errors[col + ewidth + 2] += error * 0.0625; // 1/16
                }
            }
        }
        Some((palette, qimg))
    }

    /// Write image as PNG
    pub fn write_png(&self, w: impl Write) -> Result<(), png::EncodingError> {
        let mut encoder = png::Encoder::new(w, self.width() as u32, self.height() as u32);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        let mut stream_writer = writer.stream_writer()?;
        for color in self.iter() {
            stream_writer.write_all(&color.to_rgba())?;
        }
        stream_writer.flush()?;
        Ok(())
    }

    /// ASCII image view
    pub fn ascii_view(&self) -> ImageAsciiView {
        ImageAsciiView {
            image: self.clone(),
        }
    }
}

impl PartialEq for Image {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.data, &other.data) && self.shape == other.shape
    }
}

impl Eq for Image {}

impl PartialOrd for Image {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Image {
    fn cmp(&self, other: &Self) -> Ordering {
        (Arc::as_ptr(&self.data), self.shape).cmp(&(Arc::as_ptr(&other.data), other.shape))
    }
}

impl std::hash::Hash for Image {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.data).hash(state);
        self.shape.hash(state);
    }
}

impl fmt::Debug for Image {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Image").field("size", &self.size()).finish()
    }
}

impl Surface for Image {
    type Item = RGBA;

    fn shape(&self) -> Shape {
        self.shape
    }

    fn data(&self) -> &[Self::Item] {
        &self.data
    }
}

impl From<SurfaceOwned<RGBA>> for Image {
    fn from(value: SurfaceOwned<RGBA>) -> Self {
        let shape = value.shape();
        Self {
            data: value.to_vec().into(),
            shape,
        }
    }
}

impl View for Image {
    fn render(
        &self,
        ctx: &ViewContext,
        surf: TerminalSurface<'_>,
        layout: ViewLayout<'_>,
    ) -> Result<(), Error> {
        let mut surf = layout.apply_to(surf);
        let size = surf.size();
        let ppc = ctx.pixels_per_cell();
        if let Some(cell) = surf.get_mut(Position::origin()) {
            *cell =
                Cell::new_image(self.crop(..size.height * ppc.height, ..size.width * ppc.width))
                    .with_face(cell.face());
        }
        Ok(())
    }

    fn layout(
        &self,
        ctx: &ViewContext,
        ct: BoxConstraint,
        mut layout: ViewMutLayout<'_>,
    ) -> Result<(), Error> {
        let size = ct.clamp(self.size_cells(ctx.pixels_per_cell()));
        *layout = Layout::new().with_size(size);
        Ok(())
    }
}

/// [Image] deserializer encoding is `{ data: base64(deflate(image)), size: Size, channels: u8 }`
impl<'de> Deserialize<'de> for Image {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        struct ImageVistor;

        impl<'de> de::Visitor<'de> for ImageVistor {
            type Value = Image;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                write!(
                    formatter,
                    "Map with data and size attributes, where data is `base64(deflate(RGBA)))`"
                )
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: de::MapAccess<'de>,
            {
                let mut size: Option<Size> = None;
                let mut data = Vec::new();
                let mut channels = 3;
                while let Some(key) = map.next_key::<Cow<'de, str>>()? {
                    match key.as_ref() {
                        "data" => {
                            let data_raw = map.next_value::<Cow<'de, str>>()?;
                            Base64Decoder::new(data_raw.as_bytes())
                                .read_to_end(&mut data)
                                .map_err(de::Error::custom)?;
                        }
                        "channels" => {
                            channels = map.next_value()?;
                            if !matches!(channels, 1 | 3 | 4) {
                                return Err(de::Error::custom(Error::ParseError(
                                    "Image",
                                    format!("not supported channels value {channels} expected {{1,3,4}}")
                                )));
                            }
                        }
                        "size" => {
                            size.replace(map.next_value()?);
                        }
                        _ => {
                            map.next_value::<de::IgnoredAny>()?;
                        }
                    }
                }
                let Some(size) = size else {
                    return Err(de::Error::custom(Error::ParseError(
                        "Image",
                        "missing size".to_owned(),
                    )));
                };
                let expected_size = channels * size.height * size.width;
                let data_size = data.len();
                if data_size != expected_size {
                    return Err(de::Error::custom(Error::ParseError(
                        "Image",
                        format!(
                            "data field has incorrect size {data_size} exepcted {expected_size}"
                        ),
                    )));
                }
                let surf = match channels {
                    4 => SurfaceOwned::new_with(size, |pos| {
                        let offset = 4 * (pos.row * size.width + pos.col);
                        let r = data[offset];
                        let g = data[offset + 1];
                        let b = data[offset + 2];
                        let a = data[offset + 3];
                        RGBA::new(r, g, b, a)
                    }),
                    3 => SurfaceOwned::new_with(size, |pos| {
                        let offset = 3 * (pos.row * size.width + pos.col);
                        let r = data[offset];
                        let g = data[offset + 1];
                        let b = data[offset + 2];
                        RGBA::new(r, g, b, 255)
                    }),
                    1 => SurfaceOwned::new_with(size, |pos| {
                        let v = data[pos.row * size.width + pos.col];
                        RGBA::new(v, v, v, 255)
                    }),
                    _ => unreachable!(),
                };
                Ok(Image::from(surf))
            }
        }

        deserializer.deserialize_map(ImageVistor)
    }
}

impl Serialize for Image {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut writer = Base64Encoder::new(Vec::new());
        for pixel in self.iter() {
            writer.write_all(&pixel.to_rgba()).map_err(|err| {
                ser::Error::custom(format!("[Image] faield to serialize data: {err}"))
            })?;
        }
        let data = writer.finish().map_err(|err| {
            ser::Error::custom(format!("[Image] faield to serialize data: {err}"))
        })?;
        let mut image = serializer.serialize_struct("Image", 3)?;
        image.serialize_field("size", &self.size())?;
        image.serialize_field("channels", &4)?;
        image.serialize_field(
            "data",
            std::str::from_utf8(&data).expect("base64 contains non utf8"),
        )?;
        image.end()
    }
}

/// Draw image with ASCII blocks
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ImageAsciiView {
    image: Image,
}

impl View for ImageAsciiView {
    fn render(
        &self,
        _ctx: &ViewContext,
        surf: TerminalSurface<'_>,
        layout: ViewLayout<'_>,
    ) -> Result<(), Error> {
        let mut surf = layout.apply_to(surf);
        surf.fill_with(|pos, _| {
            let fg = self.image.get(Position::new(pos.row * 2, pos.col)).copied();
            let bg = self
                .image
                .get(Position::new(pos.row * 2 + 1, pos.col))
                .copied();
            let face = Face::new(fg, bg, FaceAttrs::EMPTY);
            Cell::new_char(face, '\u{2580}')
        });
        Ok(())
    }

    fn layout(
        &self,
        _ctx: &ViewContext,
        ct: BoxConstraint,
        mut layout: ViewMutLayout<'_>,
    ) -> Result<(), Error> {
        let size = Size {
            height: self.image.height() / 2 + self.image.height() % 2,
            width: self.image.width(),
        };
        *layout = Layout::new().with_size(ct.clamp(size));
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageHandlerKind {
    Kitty,
    Sixel,
    Dummy,
}

impl ImageHandlerKind {
    pub(crate) fn into_image_handler(self, bg: Option<RGBA>) -> Box<dyn ImageHandler> {
        use ImageHandlerKind::*;
        match self {
            Kitty => Box::new(KittyImageHandler::new()),
            Sixel => Box::new(SixelImageHandler::new(bg)),
            Dummy => Box::new(DummyImageHandler),
        }
    }
}

impl FromStr for ImageHandlerKind {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use ImageHandlerKind::*;
        match s.to_ascii_lowercase().as_str() {
            "kitty" => Ok(Kitty),
            "sixel" => Ok(Sixel),
            "dummy" => Ok(Dummy),
            _ => Err(Error::ParseError(
                "ImageHandlerKind",
                format!("invalid image handler type: {}", s),
            )),
        }
    }
}

/// Image rendering/handling interface
pub trait ImageHandler: Send + Sync {
    /// Name
    fn kind(&self) -> ImageHandlerKind;

    /// Draw image
    ///
    /// Send an appropriate terminal escape sequence so the image would be rendered.
    /// Position argument `pos` assumed to be current position, and handler
    /// is not trying to change it.
    fn draw(&mut self, out: &mut dyn Write, img: &Image, pos: Position) -> Result<(), Error>;

    /// Erase image at specified position
    ///
    /// This is needed when erasing characters is not actually removing
    /// image from the terminal. For example kitty needs to send separate
    /// escape sequence to actually erase image. If position is not specified
    /// all matching images are deleted.
    fn erase(
        &mut self,
        out: &mut dyn Write,
        img: &Image,
        pos: Option<Position>,
    ) -> Result<(), Error>;

    /// Handle events from the terminal
    ///
    /// True means event has been handled and should not be propagated to a user
    fn handle(&mut self, out: &mut dyn Write, event: &TerminalEvent) -> Result<bool, Error>;
}

impl ImageHandler for Box<dyn ImageHandler> {
    fn kind(&self) -> ImageHandlerKind {
        (**self).kind()
    }

    fn draw(&mut self, out: &mut dyn Write, img: &Image, pos: Position) -> Result<(), Error> {
        (**self).draw(out, img, pos)
    }

    fn erase(
        &mut self,
        out: &mut dyn Write,
        img: &Image,
        pos: Option<Position>,
    ) -> Result<(), Error> {
        (**self).erase(out, img, pos)
    }

    fn handle(&mut self, out: &mut dyn Write, event: &TerminalEvent) -> Result<bool, Error> {
        (**self).handle(out, event)
    }
}

/// Image handler which ignores requests
pub struct DummyImageHandler;

impl ImageHandler for DummyImageHandler {
    fn kind(&self) -> ImageHandlerKind {
        ImageHandlerKind::Dummy
    }

    fn draw(&mut self, _out: &mut dyn Write, _img: &Image, _pos: Position) -> Result<(), Error> {
        Ok(())
    }

    fn erase(
        &mut self,
        _out: &mut dyn Write,
        _img: &Image,
        _pos: Option<Position>,
    ) -> Result<(), Error> {
        Ok(())
    }

    fn handle(&mut self, _out: &mut dyn Write, _event: &TerminalEvent) -> Result<bool, Error> {
        Ok(false)
    }
}

/// Image handler for kitty graphic protocol
///
/// Reference: [Kitty Graphic Protocol](https://sw.kovidgoyal.net/kitty/graphics-protocol/)
pub struct KittyImageHandler {
    imgs: HashMap<u64, Image>, // hash -> image
    suppress: Option<u8>,      // 1 - suppress OK, 2 - suppress all
}

impl KittyImageHandler {
    pub fn new() -> Self {
        Self {
            imgs: Default::default(),
            suppress: None,
        }
    }

    /// Enable suppression of OK responses from the terminal
    pub fn quiet(self) -> Self {
        Self {
            suppress: Some(1),
            ..self
        }
    }
}

impl Default for KittyImageHandler {
    fn default() -> Self {
        Self::new()
    }
}

/// Kitty id/placement_id must not exceed this value
const KITTY_MAX_ID: u64 = 4294967295;
/// We are using position to derive placement_id, and this is the limit
/// on terminal dimension (width and height).
const KITTY_MAX_DIM: u64 = 65536;

/// Identification for image data
fn kitty_image_id(img: &Image) -> u64 {
    img.hash() % KITTY_MAX_ID
}

/// Identification of particular placement of the image
///
/// In general this identification is just represents individual placement
/// but in particular implementation it is bound to a physical position on
/// the screen.
fn kitty_placement_id(pos: Position) -> u64 {
    (pos.row as u64 % KITTY_MAX_DIM) + (pos.col as u64 % KITTY_MAX_DIM) * KITTY_MAX_DIM
}

fn kitty_placement_to_pos(placement_id: u64) -> Position {
    Position {
        col: (placement_id / KITTY_MAX_DIM) as usize,
        row: (placement_id % KITTY_MAX_DIM) as usize,
    }
}

impl ImageHandler for KittyImageHandler {
    fn kind(&self) -> ImageHandlerKind {
        ImageHandlerKind::Kitty
    }

    fn draw(&mut self, out: &mut dyn Write, img: &Image, pos: Position) -> Result<(), Error> {
        tracing::trace!(
            image_handler = "kitty",
            ?pos,
            ?img,
            "[KittyImageHandler.draw]"
        );
        let img_id = kitty_image_id(img);

        // q   - suppress response from the terminal 1 - OK only, 2 - All.
        let suppress = self.suppress.unwrap_or(0);

        // transfer image if it has not been transferred yet
        if let Entry::Vacant(entry) = self.imgs.entry(img_id) {
            let _ = tracing::debug_span!(
                "[KittyImageHandler.render]",
                image_handler = "kitty",
                ?pos,
                ?img
            )
            .enter();
            // base64 encoded RGBA image data
            let mut payload_write = Base64Encoder::new(Vec::new());
            for color in img.iter() {
                payload_write.write_all(&color.to_rgba())?;
            }
            let payload = payload_write.finish()?;

            // NOTE:
            //  - data needs to be transferred in chunks
            //  - chunks should be multiple of 4, otherwise kitty complains that it is not
            //    valid base64 encoded data.
            let chunks = payload.chunks(4096);
            let count = chunks.len();
            for (index, chunk) in chunks.enumerate() {
                // control data
                let more = i32::from(index + 1 < count);
                if index == 0 {
                    // a=t  - action is transmit only
                    // f=32 - RGBA pixel format
                    // o=z  - zlib compressed data
                    // i    - image data identifier
                    // v    - height of the image
                    // s    - width of the image
                    // m    - whether more chunks will follow or not
                    write!(
                        out,
                        "\x1b_Ga=t,f=32,i={},v={},s={},m={},q={};",
                        img_id,
                        img.height(),
                        img.width(),
                        more,
                        suppress
                    )?;
                } else {
                    // only first chunk requires all attributes
                    write!(out, "\x1b_Gm={more},q={suppress};")?;
                }
                // data
                out.write_all(chunk)?;
                // epilogue
                out.write_all(b"\x1b\\")?;
            }

            // remember that image data has been send
            entry.insert(img.clone());
        }

        // request image to be shown
        let placement_id = kitty_placement_id(pos);
        // a=p - action is put image
        // i   - image data identifier
        // C=1 - do not move cursor (avoids scrolling if image is too large vertically)
        write!(
            out,
            "\x1b_Ga=p,i={img_id},C=1,p={placement_id},q={suppress};\x1b\\"
        )?;

        tracing::trace!(
            image_handler = "kitty",
            img_id,
            placement_id,
            suppress,
            "[KittyImageHandler.draw] command"
        );
        Ok(())
    }

    fn erase(
        &mut self,
        out: &mut dyn Write,
        img: &Image,
        pos: Option<Position>,
    ) -> Result<(), Error> {
        tracing::trace!(
            image_handler = "kitty",
            ?pos,
            ?img,
            "[KittyImageHandler.erase]"
        );
        // Delete image by image id and placement id
        // a=d - action delete image
        // d=i - delete by image and placement id without freeing data
        // i   - image data identifier
        // p   - placement identifier
        match pos {
            Some(pos) => write!(
                out,
                "\x1b_Ga=d,d=i,i={},p={}\x1b\\",
                kitty_image_id(img),
                kitty_placement_id(pos),
            )?,
            None => write!(out, "\x1b_Ga=d,d=i,i={}\x1b\\", kitty_image_id(img))?,
        }
        Ok(())
    }

    fn handle(&mut self, mut out: &mut dyn Write, event: &TerminalEvent) -> Result<bool, Error> {
        match event {
            TerminalEvent::KittyImage {
                id,
                placement,
                error,
            } => {
                if error.is_some() {
                    let pos = placement.map(kitty_placement_to_pos);
                    tracing::warn!(id, ?pos, "[KittyImageHandler.error]: {:?}", error);
                    if let (Some(img), Some(pos)) = (self.imgs.remove(id), pos) {
                        let mut encoder = TTYEncoder::default();
                        encoder.encode(&mut out, TerminalCommand::CursorSave)?;
                        encoder.encode(&mut out, TerminalCommand::CursorTo(pos))?;
                        // suppress responses to avoid infinite loop, for example
                        // in case of very large image.
                        let suppress = self.suppress;
                        self.suppress.replace(2);
                        self.draw(out, &img, pos)?;
                        self.suppress = suppress;
                        encoder.encode(out, TerminalCommand::CursorRestore)?;
                    }
                }
                Ok(true)
            }
            _ => Ok(false),
        }
    }
}

/// Image handler for sixel graphic protocol
///
/// References:
///  - [Sixel](https://en.wikipedia.org/wiki/Sixel)
///  - [All About SIXELs](https://www.digiater.nl/openvms/decus/vax90b1/krypton-nasa/all-about-sixels.text)
pub struct SixelImageHandler {
    imgs: lru::LruCache<u64, Vec<u8>>,
    size: usize,
    bg: Option<RGBA>,
}

impl SixelImageHandler {
    pub fn new(bg: Option<RGBA>) -> Self {
        SixelImageHandler {
            imgs: lru::LruCache::unbounded(),
            size: 0,
            bg,
        }
    }
}

impl ImageHandler for SixelImageHandler {
    fn kind(&self) -> ImageHandlerKind {
        ImageHandlerKind::Sixel
    }

    fn draw(&mut self, out: &mut dyn Write, img: &Image, pos: Position) -> Result<(), Error> {
        tracing::debug!(
            image_handler = "sixel",
            ?pos,
            ?img,
            "[SixelImageHandler.draw]"
        );
        if let Some(sixel_image) = self.imgs.get(&img.hash()) {
            out.write_all(sixel_image.as_slice())?;
            return Ok(());
        }
        let _ = tracing::debug_span!("[SixelImageHandler.render]", image_handler = "sixel");
        // make sure height is always dividable by 6
        let height = (img.height() / 6) * 6;
        // sixel color chanel has a range [0,100] colors, we need to reduce it before
        // quantization, it will produce smaller or/and better palette for this color depth
        let dimg = Image::from(img.view(..height, ..).map(|_, color| {
            let [red, green, blue, alpha] = color.to_rgba();
            let red = ((red as f32 / 2.55).round() * 2.55) as u8;
            let green = ((green as f32 / 2.55).round() * 2.55) as u8;
            let blue = ((blue as f32 / 2.55).round() * 2.55) as u8;
            RGBA::new(red, green, blue, alpha)
        }));
        let (palette, qimg) = match dimg.quantize(256, true, self.bg) {
            None => return Ok(()),
            Some(qimg) => qimg,
        };

        let mut sixel_image = Vec::new();
        // header
        sixel_image.write_all(b"\x1bPq")?;
        write!(sixel_image, "\"1;1;{};{}", qimg.width(), qimg.height())?;
        // palette
        for (index, color) in palette.colors().iter().enumerate() {
            let [red, green, blue] = color.to_rgb();
            let red = (red as f32 / 2.55).round() as u8;
            let green = (green as f32 / 2.55).round() as u8;
            let blue = (blue as f32 / 2.55).round() as u8;
            // 2 - means RGB, 1 - means HLS
            write!(sixel_image, "#{};2;{};{};{}", index, red, green, blue)?;
        }
        // color_index -> [(column, sixel_code)]
        let mut sixel_lines: HashMap<usize, Vec<(usize, u8)>> = HashMap::new();
        let mut unique_colors: HashSet<usize> = HashSet::with_capacity(6);
        for row in (0..qimg.height()).step_by(6) {
            sixel_lines.clear();
            // extract sixel line
            for col in 0..img.width() {
                // extract sixel
                let mut sixel = [0usize; 6];
                for (i, s) in sixel.iter_mut().enumerate() {
                    if let Some(index) = qimg.get(Position::new(row + i, col)) {
                        *s = *index;
                    }
                }
                // construct sixel
                unique_colors.clear();
                unique_colors.extend(sixel.iter().copied());
                for color in unique_colors.iter() {
                    let mut sixel_code = 0;
                    for (s_index, s_color) in sixel.iter().enumerate() {
                        if s_color == color {
                            sixel_code |= 1 << s_index;
                        }
                    }
                    sixel_lines
                        .entry(*color)
                        .or_default()
                        .push((col, sixel_code + 63));
                }
            }
            // render sixel line
            // commands:
            //   `#<color_index>`  - set color palette
            //   `!<number>`       - repeat next sixel multiple times
            //   `?`               - it is just empty sixel
            //   `$`               - move to the beginning of the current line
            //   `-`               - move to the next sixel line
            for (color, sixel_line) in sixel_lines.iter() {
                write!(sixel_image, "#{}", color)?; // set color

                let mut offset = 0;
                let mut codes = sixel_line.iter().peekable();
                while let Some((column, code)) = codes.next() {
                    // find shift needed to get to the correct offset
                    let shift = column - offset;
                    if shift > 0 {
                        // determine whether it is more efficient to send repeats
                        // or just blank sixel `?` multiple times
                        if shift > 3 {
                            write!(sixel_image, "!{}?", shift)?;
                        } else {
                            for _ in 0..shift {
                                sixel_image.write_all(b"?")?;
                            }
                        }
                    }
                    // find repeated sixels
                    let mut repeats = 1;
                    while let Some((column_next, code_next)) = codes.peek() {
                        if *column_next != column + repeats || code_next != code {
                            break;
                        }
                        repeats += 1;
                        codes.next();
                    }
                    // write sixel
                    if repeats > 3 {
                        write!(sixel_image, "!{}", repeats)?;
                        sixel_image.write_all(&[*code])?;
                    } else {
                        for _ in 0..repeats {
                            sixel_image.write_all(&[*code])?;
                        }
                    }
                    offset = column + repeats;
                }
                sixel_image.write_all(b"$")?;
            }
            sixel_image.write_all(b"-")?;
        }
        // EOF sixel
        sixel_image.write_all(b"\x1b\\")?;

        out.write_all(sixel_image.as_slice())?;

        self.size += sixel_image.len();
        self.imgs.put(img.hash(), sixel_image);
        while self.size > IMAGE_CACHE_SIZE {
            let Some((_, lru_image)) = self.imgs.pop_lru() else {
                break;
            };
            self.size -= lru_image.len();
        }

        Ok(())
    }

    fn erase(
        &mut self,
        _out: &mut dyn Write,
        _img: &Image,
        _pos: Option<Position>,
    ) -> Result<(), Error> {
        Ok(())
    }

    fn handle(&mut self, _out: &mut dyn Write, _event: &TerminalEvent) -> Result<bool, Error> {
        Ok(false)
    }
}

/// Color like object to track quantization
///
/// Used in Floyd–Steinberg dithering.
#[derive(Clone, Copy)]
struct ColorError([f32; 3]);

impl ColorError {
    fn new() -> Self {
        Self([0.0; 3])
    }

    /// Error between two colors
    fn between(c0: RGBA, c1: RGBA) -> Self {
        let [r0, g0, b0] = c0.to_rgb();
        let [r1, g1, b1] = c1.to_rgb();
        Self([
            r0 as f32 - r1 as f32,
            g0 as f32 - g1 as f32,
            b0 as f32 - b1 as f32,
        ])
    }

    /// Add error to the color
    fn add(self, color: RGBA) -> RGBA {
        let [r, g, b] = color.to_rgb();
        let Self([re, ge, be]) = self;
        RGBA::new(
            clamp(r as f32 + re, 0.0, 255.0) as u8,
            clamp(g as f32 + ge, 0.0, 255.0) as u8,
            clamp(b as f32 + be, 0.0, 255.0) as u8,
            255,
        )
    }
}

impl Add<Self> for ColorError {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self::Output {
        let Self([r0, g0, b0]) = self;
        let Self([r1, g1, b1]) = other;
        Self([r0 + r1, g0 + g1, b0 + b1])
    }
}

impl AddAssign for ColorError {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs
    }
}

impl Mul<f32> for ColorError {
    type Output = Self;

    #[inline]
    fn mul(self, val: f32) -> Self::Output {
        let Self([r, g, b]) = self;
        Self([r * val, g * val, b * val])
    }
}

#[derive(Debug, Clone, Copy)]
struct OcTreeLeaf {
    red_acc: usize,
    green_acc: usize,
    blue_acc: usize,
    color_count: usize,
    index: usize,
}

impl OcTreeLeaf {
    fn new() -> Self {
        Self {
            red_acc: 0,
            green_acc: 0,
            blue_acc: 0,
            color_count: 0,
            index: 0,
        }
    }

    fn from_rgba(rgba: RGBA) -> Self {
        let [r, g, b] = rgba.to_rgb();
        Self {
            red_acc: r as usize,
            green_acc: g as usize,
            blue_acc: b as usize,
            color_count: 1,
            index: 0,
        }
    }

    fn to_rgba(self) -> RGBA {
        let r = (self.red_acc / self.color_count) as u8;
        let g = (self.green_acc / self.color_count) as u8;
        let b = (self.blue_acc / self.color_count) as u8;
        RGBA::new(r, g, b, 255)
    }
}

impl AddAssign<RGBA> for OcTreeLeaf {
    fn add_assign(&mut self, rhs: RGBA) {
        let [r, g, b] = rhs.to_rgb();
        self.red_acc += r as usize;
        self.green_acc += g as usize;
        self.blue_acc += b as usize;
        self.color_count += 1;
    }
}

impl AddAssign<OcTreeLeaf> for OcTreeLeaf {
    fn add_assign(&mut self, rhs: Self) {
        self.red_acc += rhs.red_acc;
        self.green_acc += rhs.green_acc;
        self.blue_acc += rhs.blue_acc;
        self.color_count += rhs.color_count;
    }
}

#[derive(Debug, Clone)]
enum OcTreeNode {
    Leaf(OcTreeLeaf),
    Tree(Box<OcTree>),
    Empty,
}

impl OcTreeNode {
    pub fn is_empty(&self) -> bool {
        matches!(self, OcTreeNode::Empty)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OcTreeInfo {
    // total number of leafs in the subtree
    pub leaf_count: usize,
    // total number of colors in the subtree
    pub color_count: usize,
    // node (Tree|Leaf) with smallest number of colors in the subtree
    pub min_color_count: Option<usize>,
}

impl OcTreeInfo {
    // Monoidal unit
    pub fn empty() -> Self {
        Self {
            leaf_count: 0,
            color_count: 0,
            min_color_count: None,
        }
    }

    // Monoidal sum
    pub fn join(self, other: Self) -> Self {
        let leaf_count = self.leaf_count + other.leaf_count;
        let color_count = self.color_count + other.color_count;
        let min_color_count = match (self.min_color_count, other.min_color_count) {
            (Some(c0), Some(c1)) => Some(std::cmp::min(c0, c1)),
            (None, Some(c1)) => Some(c1),
            (Some(c0), None) => Some(c0),
            (None, None) => None,
        };
        Self {
            leaf_count,
            color_count,
            min_color_count,
        }
    }

    // Monoidal sum over oll infos of nodes in the slice
    fn from_slice(slice: &[OcTreeNode]) -> Self {
        slice
            .iter()
            .fold(Self::empty(), |acc, n| acc.join(n.info()))
    }
}

impl OcTreeNode {
    // Take node content and replace it with empty node
    fn take(&mut self) -> Self {
        std::mem::replace(self, Self::Empty)
    }

    // Get info associated with the node
    fn info(&self) -> OcTreeInfo {
        use OcTreeNode::*;
        match self {
            Empty => OcTreeInfo::empty(),
            Leaf(leaf) => OcTreeInfo {
                leaf_count: 1,
                color_count: leaf.color_count,
                min_color_count: Some(leaf.color_count),
            },
            Tree(tree) => tree.info,
        }
    }
}

/// Oc(tet)Tree used for color quantization
///
/// References:
/// - [OcTree color quantization](https://www.cubic.org/docs/octree.htm)
/// - [Color quantization](http://www.leptonica.org/color-quantization.html)
#[derive(Debug, Clone)]
pub struct OcTree {
    info: OcTreeInfo,
    removed: OcTreeLeaf,
    children: [OcTreeNode; 8],
}

impl Default for OcTree {
    fn default() -> Self {
        Self::new()
    }
}

impl Extend<RGBA> for OcTree {
    fn extend<T: IntoIterator<Item = RGBA>>(&mut self, colors: T) {
        for color in colors {
            self.insert(color)
        }
    }
}

impl FromIterator<RGBA> for OcTree {
    fn from_iter<T: IntoIterator<Item = RGBA>>(iter: T) -> Self {
        let mut octree = OcTree::new();
        octree.extend(iter);
        octree
    }
}

impl OcTree {
    /// Create empty OcTree
    pub fn new() -> Self {
        use OcTreeNode::Empty;
        Self {
            info: OcTreeInfo::empty(),
            removed: OcTreeLeaf::new(),
            children: [Empty, Empty, Empty, Empty, Empty, Empty, Empty, Empty],
        }
    }

    /// Get info associated with the node
    #[cfg(test)]
    fn info(&self) -> OcTreeInfo {
        self.info
    }

    /// Find nearest color inside the octree
    ///
    /// NOTE:
    ///  - to get correct palette index call build_palette first.
    ///  - prefer `ColorPalette::find` as it produces better result, and can not return None.
    pub fn find(&self, color: RGBA) -> Option<(usize, RGBA)> {
        use OcTreeNode::*;
        let mut tree = self;
        for index in OcTreePath::new(color) {
            match &tree.children[index] {
                Empty => break,
                Leaf(leaf) => return Some((leaf.index, leaf.to_rgba())),
                Tree(next_tree) => tree = next_tree,
            }
        }
        None
    }

    /// Extract all colors present in the octree and update leaf color indices
    pub fn build_palette(&mut self) -> Vec<RGBA> {
        fn palette_rec(node: &mut OcTreeNode, palette: &mut Vec<RGBA>) {
            use OcTreeNode::*;
            match node {
                Empty => {}
                Leaf(ref mut leaf) => {
                    leaf.index = palette.len();
                    palette.push(leaf.to_rgba());
                }
                Tree(tree) => {
                    for child in tree.children.iter_mut() {
                        palette_rec(child, palette)
                    }
                }
            }
        }

        let mut palette = Vec::new();
        for child in self.children.iter_mut() {
            palette_rec(child, &mut palette);
        }
        palette
    }

    /// Update node with provided function.
    fn node_update(&mut self, index: usize, func: impl FnOnce(OcTreeNode) -> OcTreeNode) {
        self.children[index] = func(self.children[index].take());
        self.info = OcTreeInfo::from_slice(&self.children);
    }

    /// Insert color into the octree
    pub fn insert(&mut self, color: RGBA) {
        // Recursive insertion of the color into a node
        fn insert_rec(node: OcTreeNode, mut path: OcTreePath) -> OcTreeNode {
            use OcTreeNode::*;
            match path.next() {
                Some(index) => match node {
                    Empty => {
                        let mut tree = OcTree::new();
                        tree.node_update(index, move |node| insert_rec(node, path));
                        Tree(Box::new(tree))
                    }
                    Leaf(mut leaf) => {
                        leaf += path.rgba();
                        Leaf(leaf)
                    }
                    Tree(mut tree) => {
                        tree.node_update(index, move |node| insert_rec(node, path));
                        Tree(tree)
                    }
                },
                None => match node {
                    Empty => Leaf(OcTreeLeaf::from_rgba(path.rgba())),
                    Leaf(mut leaf) => {
                        leaf += path.rgba();
                        Leaf(leaf)
                    }
                    Tree(_) => unreachable!(),
                },
            }
        }

        let mut path = OcTreePath::new(color);
        let index = path.next().expect("OcTreePath can not be empty");
        self.node_update(index, |node| insert_rec(node, path));
    }

    /// Prune until desired number of colors is left
    pub fn prune_until(&mut self, color_count: usize) {
        let prune_count = color_count.max(8);
        while self.info.leaf_count > prune_count {
            self.prune();
        }
    }

    /// Remove the node with minimal number of colors in the it
    pub fn prune(&mut self) {
        use OcTreeNode::*;

        // find child index with minimal color count in the child subtree
        fn argmin_color_count(tree: &OcTree) -> Option<usize> {
            tree.children
                .iter()
                .enumerate()
                .filter_map(|(index, node)| Some((index, node.info().min_color_count?)))
                .min_by_key(|(_, min_tail_tree)| *min_tail_tree)
                .map(|(index, _)| index)
        }

        // recursive prune helper
        fn prune_rec(mut tree: Box<OcTree>) -> OcTreeNode {
            match argmin_color_count(&tree) {
                None => Leaf(tree.removed),
                Some(index) => match tree.children[index].take() {
                    Empty => unreachable!("agrmin_color_count found and empty node"),
                    Leaf(leaf) => {
                        tree.removed += leaf;
                        if tree.children.iter().all(OcTreeNode::is_empty) {
                            Leaf(tree.removed)
                        } else {
                            Tree(tree)
                        }
                    }
                    Tree(child_tree) => {
                        let child = prune_rec(child_tree);
                        match child {
                            Leaf(leaf) if tree.children.iter().all(OcTreeNode::is_empty) => {
                                tree.removed += leaf;
                                Leaf(tree.removed)
                            }
                            _ => {
                                tree.node_update(index, |_| child);
                                Tree(tree)
                            }
                        }
                    }
                },
            }
        }

        if let Some(index) = argmin_color_count(self) {
            match self.children[index].take() {
                Empty => unreachable!("agrmin_color_count found and empty node"),
                Leaf(leaf) => self.removed += leaf,
                Tree(child_tree) => {
                    let child = prune_rec(child_tree);
                    self.node_update(index, |_| child);
                }
            }
        }
    }

    /// Render octree as graphviz digraph (for debugging)
    pub fn to_digraph<W: Write>(&self, mut out: W) -> std::io::Result<()> {
        pub fn to_digraph_rec<W: Write>(
            tree: &OcTree,
            parent: usize,
            next: &mut usize,
            out: &mut W,
        ) -> std::io::Result<()> {
            use OcTreeNode::*;
            for child in tree.children.iter() {
                match child {
                    Empty => continue,
                    Leaf(leaf) => {
                        let id = *next;
                        *next += 1;

                        let fg = leaf
                            .to_rgba()
                            .best_contrast(RGBA::new(255, 255, 255, 255), RGBA::new(0, 0, 0, 255));
                        writeln!(
                            out,
                            "  {} [style=filled, fontcolor=\"{}\" fillcolor=\"{}\", label=\"{}\"]",
                            id,
                            fg,
                            leaf.to_rgba(),
                            leaf.color_count
                        )?;
                        writeln!(out, "  {} -> {}", parent, id)?;
                    }
                    Tree(child) => {
                        let id = *next;
                        *next += 1;

                        writeln!(
                            out,
                            "  {} [label=\"{} {}\"]",
                            id,
                            child.info.leaf_count,
                            child.info.min_color_count.unwrap_or(0),
                        )?;
                        writeln!(out, "  {} -> {}", parent, id)?;
                        to_digraph_rec(child, id, next, out)?
                    }
                }
            }
            Ok(())
        }

        let mut next = 1;
        writeln!(out, "digraph OcTree {{")?;
        writeln!(out, "  rankdir=\"LR\"")?;
        writeln!(
            out,
            "  0 [label=\"{} {}\"]",
            self.info.leaf_count,
            self.info.min_color_count.unwrap_or(0),
        )?;
        to_digraph_rec(self, 0, &mut next, &mut out)?;
        writeln!(out, "}}")?;
        Ok(())
    }
}

/// Iterator which goes over all most significant bits of the color
/// concatenated together.
///
/// Example:
/// For RGB (90, 13, 157) in binary form
/// R 0 1 0 1 1 0 1 0
/// G 0 1 1 1 0 0 0 1
/// B 1 0 0 1 1 1 0 1
/// Output will be [0b001, 0b110, 0b010, 0b111, 0b101, 0b001, 0b100, 0b011]
struct OcTreePath {
    rgba: RGBA,
    state: u32,
    length: u8,
}

impl OcTreePath {
    pub fn new(rgba: RGBA) -> Self {
        let [r, g, b] = rgba.to_rgb();
        // pack RGB components into u32 value
        let state = ((r as u32) << 16) | ((g as u32) << 8) | b as u32;
        Self {
            rgba,
            state,
            length: 8,
        }
    }

    /// Convert octree path to a color
    pub fn rgba(&self) -> RGBA {
        self.rgba
    }
}

impl Iterator for OcTreePath {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.length == 0 {
            return None;
        }
        self.length -= 1;
        // - We should pick most significant bit from each component
        //   and concatenate into one value to get an index inside
        //   octree.
        // - Left shift all components and set least significant bits
        //   of all components to zero.
        // - Repeat until all bits of all components are zero
        let bits = self.state & 0x00808080;
        self.state = (self.state << 1) & 0x00fefefe;
        let value = ((bits >> 21) | (bits >> 14) | (bits >> 7)) & 0b111;
        Some(value as usize)
    }
}

/// 3-dimensional KDTree which is used to quickly find nearest (euclidean distance)
/// color from the palette.
///
/// Reference: [k-d tree](https://en.wikipedia.org/wiki/K-d_tree)
pub struct KDTree {
    nodes: Vec<KDNode>,
}

#[derive(Debug, Clone, Copy)]
struct KDNode {
    color: [u8; 3],
    color_index: usize,
    dim: usize,
    left: Option<usize>,
    right: Option<usize>,
}

impl KDTree {
    /// Create k-d tree from the list of colors
    pub fn new(colors: &[RGBA]) -> Self {
        fn build_rec(
            dim: usize,
            nodes: &mut Vec<KDNode>,
            colors: &mut [(usize, [u8; 3])],
        ) -> Option<usize> {
            match colors {
                [] => return None,
                [(color_index, color)] => {
                    nodes.push(KDNode {
                        color: *color,
                        color_index: *color_index,
                        dim,
                        left: None,
                        right: None,
                    });
                    return Some(nodes.len() - 1);
                }
                _ => (),
            }
            colors.sort_by_key(|(_, c)| c[dim]);
            let index = colors.len() / 2;
            let dim_next = (dim + 1) % 3;
            let left = build_rec(dim_next, nodes, &mut colors[..index]);
            let right = build_rec(dim_next, nodes, &mut colors[(index + 1)..]);
            let (color_index, color) = colors[index];
            nodes.push(KDNode {
                color,
                color_index,
                dim,
                left,
                right,
            });
            Some(nodes.len() - 1)
        }

        let mut nodes = Vec::new();
        let mut colors: Vec<_> = colors.iter().map(|c| c.to_rgb()).enumerate().collect();
        build_rec(0, &mut nodes, &mut colors);
        Self { nodes }
    }

    /// Find nearest neighbor color (euclidean distance) in the palette
    pub fn find(&self, color: RGBA) -> (usize, RGBA) {
        fn dist(rgb: [u8; 3], node: &KDNode) -> i32 {
            let [r0, g0, b0] = rgb;
            let [r1, g1, b1] = node.color;
            (r0 as i32 - r1 as i32).pow(2)
                + (g0 as i32 - g1 as i32).pow(2)
                + (b0 as i32 - b1 as i32).pow(2)
        }

        fn find_rec(nodes: &[KDNode], index: usize, target: [u8; 3]) -> (KDNode, i32) {
            let node = nodes[index];
            let node_dist = dist(target, &node);
            let (next, other) = if target[node.dim] < node.color[node.dim] {
                (node.left, node.right)
            } else {
                (node.right, node.left)
            };
            let (guess, guess_dist) = match next {
                None => (node, node_dist),
                Some(next_index) => {
                    let (guess, guess_dist) = find_rec(nodes, next_index, target);
                    if guess_dist >= node_dist {
                        (node, node_dist)
                    } else {
                        (guess, guess_dist)
                    }
                }
            };
            // check if the other branch is closer then best match we have found so far.
            let other_dist = (target[node.dim] as i32 - node.color[node.dim] as i32).pow(2);
            if other_dist >= guess_dist {
                return (guess, guess_dist);
            }
            match other {
                None => (guess, guess_dist),
                Some(other_index) => {
                    let (other, other_dist) = find_rec(nodes, other_index, target);
                    if other_dist < guess_dist {
                        (other, other_dist)
                    } else {
                        (guess, guess_dist)
                    }
                }
            }
        }

        let node = find_rec(&self.nodes, self.nodes.len() - 1, color.to_rgb()).0;
        let [r, g, b] = node.color;
        (node.color_index, RGBA::new(r, g, b, 255))
    }

    /// Render k-d tree as graphviz digraph (for debugging)
    pub fn to_digraph(&self, mut out: impl Write) -> std::io::Result<()> {
        fn to_digraph_rec(
            out: &mut impl Write,
            nodes: &[KDNode],
            index: usize,
        ) -> std::io::Result<()> {
            let node = nodes[index];
            let d = match node.dim {
                0 => "R",
                1 => "G",
                2 => "B",
                _ => unreachable!(),
            };
            let [r, g, b] = node.color;
            let color = RGBA::new(r, g, b, 255);
            let fg = color.best_contrast(RGBA::new(255, 255, 255, 255), RGBA::new(0, 0, 0, 255));
            writeln!(
                out,
                "  {} [style=filled, fontcolor=\"{}\" fillcolor=\"{}\", label=\"{} {} {:?}\"]",
                index, fg, color, d, node.color[node.dim], node.color,
            )?;
            if let Some(left) = node.left {
                writeln!(out, "  {} -> {} [color=green]", index, left)?;
                to_digraph_rec(out, nodes, left)?;
            }
            if let Some(right) = node.right {
                writeln!(out, "  {} -> {} [color=red]", index, right)?;
                to_digraph_rec(out, nodes, right)?;
            }
            Ok(())
        }

        writeln!(out, "digraph KDTree {{")?;
        writeln!(out, "  rankdir=\"LR\"")?;
        to_digraph_rec(&mut out, &self.nodes, self.nodes.len() - 1)?;
        writeln!(out, "}}")?;
        Ok(())
    }
}

/// Color palette which implements fast NNS with euclidean distance.
pub struct ColorPalette {
    colors: Vec<RGBA>,
    kdtree: KDTree,
}

impl ColorPalette {
    /// Create new palette for the list of colors
    pub fn new(colors: Vec<RGBA>) -> Option<Self> {
        if colors.is_empty() {
            None
        } else {
            let kdtree = KDTree::new(&colors);
            Some(Self { colors, kdtree })
        }
    }

    /// Extract palette from image using `OcTree`
    pub fn from_image(
        img: impl Surface<Item = RGBA>,
        palette_size: usize,
        bg: RGBA,
    ) -> Option<Self> {
        fn blend(bg: RGBA, color: RGBA) -> RGBA {
            if color.to_rgba()[3] < 255 {
                bg.blend_over(color)
            } else {
                color
            }
        }

        if img.is_empty() {
            return None;
        }
        let sample: u32 = (img.height() * img.width() / (palette_size * 100)) as u32;
        let mut octree: OcTree = if sample < 2 {
            img.iter().map(|c| blend(bg, *c)).collect()
        } else {
            let mut octree = OcTree::new();
            let mut rnd = Rnd::new();
            let mut colors = img.iter().copied();
            while let Some(color) = colors.nth((rnd.next_u32() % sample) as usize) {
                octree.insert(blend(bg, color));
            }
            octree
        };
        octree.prune_until(palette_size);
        Self::new(octree.build_palette())
    }

    // Number of color in the palette
    pub fn size(&self) -> usize {
        self.colors.len()
    }

    /// Get color by the index
    pub fn get(&self, index: usize) -> RGBA {
        self.colors[index]
    }

    /// List of colors available in the palette
    pub fn colors(&self) -> &[RGBA] {
        &self.colors
    }

    /// Find nearest color in the palette
    ///
    /// Returns index of the color and color itself
    pub fn find(&self, color: RGBA) -> (usize, RGBA) {
        self.kdtree.find(color)
    }

    /// Find nearest color in the palette by going over all colors
    ///
    /// This is a slower version of the find method, used only for testing
    /// find correctness and speed.
    pub fn find_naive(&self, color: RGBA) -> (usize, RGBA) {
        fn dist(c0: RGBA, c1: RGBA) -> i32 {
            let [r0, g0, b0] = c0.to_rgb();
            let [r1, g1, b1] = c1.to_rgb();
            (r0 as i32 - r1 as i32).pow(2)
                + (g0 as i32 - g1 as i32).pow(2)
                + (b0 as i32 - b1 as i32).pow(2)
        }
        let best_dist = dist(color, self.colors[0]);
        let (best_index, _) =
            (1..self.colors.len()).fold((0, best_dist), |(best_index, best_dist), index| {
                let dist = dist(color, self.colors[index]);
                if dist < best_dist {
                    (index, dist)
                } else {
                    (best_index, best_dist)
                }
            });
        (best_index, self.colors[best_index])
    }
}

#[cfg(test)]
mod tests {
    use crate::common::Random;

    use super::*;

    /// Convert path generated by OcTreePath back to RGBA color
    fn color_from_path(path: &Vec<usize>) -> RGBA {
        let mut r: u8 = 0;
        let mut g: u8 = 0;
        let mut b: u8 = 0;
        for index in 0..8 {
            r <<= 1;
            g <<= 1;
            b <<= 1;
            let bits = path.get(index).unwrap_or(&0);
            if bits & 0b100 != 0 {
                r |= 1;
            }
            if bits & 0b010 != 0 {
                g |= 1;
            }
            if bits & 0b001 != 0 {
                b |= 1;
            }
        }
        RGBA::new(r, g, b, 255)
    }

    #[test]
    fn test_octree_path() -> Result<(), Error> {
        let c0 = "#5a719d".parse::<RGBA>()?;
        let path: Vec<_> = OcTreePath::new(c0).collect();
        assert_eq!(path, vec![1, 6, 2, 7, 5, 1, 4, 3]);
        assert_eq!(c0, color_from_path(&path));

        let c1 = "#808080".parse::<RGBA>()?;
        let path: Vec<_> = OcTreePath::new(c1).collect();
        assert_eq!(path, vec![7, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(c1, color_from_path(&vec![7]));

        let c2 = "#d3869b".parse::<RGBA>()?;
        let path: Vec<_> = OcTreePath::new(c2).collect();
        assert_eq!(path, vec![7, 4, 0, 5, 1, 2, 7, 5]);
        assert_eq!(c2, color_from_path(&path));

        Ok(())
    }

    #[test]
    fn test_octree_info() {
        let mut tree = OcTree::new();
        tree.node_update(1, |_| {
            OcTreeNode::Leaf(OcTreeLeaf {
                red_acc: 1,
                green_acc: 2,
                blue_acc: 3,
                color_count: 4,
                index: 0,
            })
        });
        assert_eq!(
            tree.info,
            OcTreeInfo {
                leaf_count: 1,
                color_count: 4,
                min_color_count: Some(4),
            }
        );
    }

    #[test]
    fn test_octree() -> Result<(), Error> {
        let c0 = "#5a719d".parse::<RGBA>()?;
        let c1 = "#d3869b".parse::<RGBA>()?;

        let mut tree = OcTree::new();

        tree.insert(c0);
        tree.insert(c0);
        assert_eq!(
            tree.info(),
            OcTreeInfo {
                color_count: 2,
                leaf_count: 1,
                min_color_count: Some(2),
            }
        );
        assert_eq!(tree.find(c0), Some((0, c0)));

        tree.insert(c1);
        assert_eq!(
            tree.info(),
            OcTreeInfo {
                color_count: 3,
                leaf_count: 2,
                min_color_count: Some(1),
            }
        );
        assert_eq!(tree.find(c1), Some((0, c1)));

        Ok(())
    }

    #[test]
    pub fn test_palette() {
        // make sure that k-d tree can actually find nearest neighbor
        fn dist(c0: RGBA, c1: RGBA) -> i32 {
            let [r0, g0, b0] = c0.to_rgb();
            let [r1, g1, b1] = c1.to_rgb();
            (r0 as i32 - r1 as i32).pow(2)
                + (g0 as i32 - g1 as i32).pow(2)
                + (b0 as i32 - b1 as i32).pow(2)
        }

        let mut gen = RGBA::random_iter();
        let palette = ColorPalette::new((&mut gen).take(256).collect()).unwrap();
        let mut colors: Vec<_> = gen.take(65_536).collect();
        colors.extend(palette.colors().iter().copied());
        for (index, color) in colors.iter().enumerate() {
            let (_, find) = palette.find(*color);
            let (_, find_naive) = palette.find_naive(*color);
            if find != find_naive && dist(*color, find) != dist(*color, find_naive) {
                dbg!(dist(*color, find));
                dbg!(dist(*color, find_naive));
                panic!(
                    "failed to find colors[{}]={:?}: find_naive={:?} find={:?}",
                    index, color, find_naive, find
                );
            }
        }
    }

    #[test]
    fn test_image_serde() -> Result<(), Error> {
        let image_value = serde_json::json!({
            "data": "AAAAAAAAAAAAAAAAAAAAAP8AAAAAAP8AAAAAAAAA/wAAAP8AAAAAAAAA/////////wAAAAAA//8A////AP//AAAA//////////////8AAP8A/////////wD/AAD/AP8AAAAAAP8A/wAAAAAA//8A//8AAAAAAAAAAAAAAAAAAAAAAA==",
            "channels": 1,
            "size": [10, 13],
        });

        let image = Image::deserialize(image_value)?;
        println!(
            "[image] space invader: {:?}",
            image.ascii_view().debug(Size::new(5, 13))
        );

        let image_str = serde_json::to_string(&image)?;
        let image_serde: Image = serde_json::from_str(&image_str)?;
        println!(
            "[image] serde invader: {:?}",
            image_serde.ascii_view().debug(Size::new(5, 13))
        );

        assert_eq!(image.data(), image_serde.data());
        assert_eq!(image.shape(), image_serde.shape());

        Ok(())
    }
}
