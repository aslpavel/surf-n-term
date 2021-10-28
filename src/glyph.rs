use crate::{
    Color, ColorLinear, Face, Image, Size, Surface, SurfaceMut, SurfaceOwned, TerminalSize, RGBA,
};
pub use rasterize::FillRule;
use rasterize::{ActiveEdgeRasterizer, Align, BBox, Path, Rasterizer, Transform};
use std::{
    hash::{Hash, Hasher},
    io::Write,
    sync::Arc,
};

/// Glyph defined as an SVG path
#[derive(Debug, Clone)]
pub struct Glyph {
    /// Rasterize path representing the glyph
    path: Arc<Path>,
    /// View box
    view_box: BBox,
    /// Fill rule
    fill_rule: FillRule,
    /// Glyph size in cells
    size: Size,
    /// Hash that is used to determine equality of the paths
    hash: u64,
}

impl Glyph {
    pub fn new(path: Path, fill_rule: FillRule, view_box: Option<BBox>, size: Size) -> Self {
        let view_box = view_box
            .or_else(|| path.bbox(Transform::identity()))
            .unwrap_or(BBox::new((0.0, 0.0), (1.0, 1.0)));

        let mut hasher = GlyphHasher::new();
        path.write_svg_path(&mut hasher).unwrap();
        write!(&mut hasher, "{:?}", view_box).unwrap();
        let hash = hasher.finish();

        Self {
            path: Arc::new(path),
            view_box,
            hash,
            size,
            fill_rule,
        }
    }

    /// Rasterize glyph into an image with provied face.
    pub fn rasterize(&self, face: Face, term_size: TerminalSize) -> Image {
        let pixel_size = term_size.cells_in_pixels(self.size);
        let size = rasterize::Size {
            height: pixel_size.height,
            width: pixel_size.width,
        };
        let tr = Transform::fit_size(self.view_box, size, Align::Mid);

        let bg_rgba = face.bg.unwrap_or(RGBA::new(0, 0, 0, 0));
        let bg: ColorLinear = bg_rgba.into();
        let fg: ColorLinear = face.fg.unwrap_or(RGBA::new(255, 255, 255, 255)).into();

        let rasterizer = ActiveEdgeRasterizer::default();
        let mut surf = SurfaceOwned::new_with(size.height, size.width, |_, _| bg_rgba);
        let shape = surf.shape();
        let data = surf.data_mut();
        for pixel in rasterizer.mask_iter(&self.path, tr, size, self.fill_rule) {
            data[shape.offset(pixel.y, pixel.x)] = bg.lerp(fg, pixel.alpha).into();
        }

        Image::new(surf)
    }
}

impl PartialEq for Glyph {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash && self.fill_rule == other.fill_rule && self.size == other.size
    }
}

impl Eq for Glyph {}

impl Hash for Glyph {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash.hash(state);
        self.fill_rule.hash(state);
    }
}

struct GlyphHasher {
    hasher: fnv::FnvHasher,
}

impl GlyphHasher {
    fn new() -> Self {
        Self {
            hasher: Default::default(),
        }
    }

    fn finish(&self) -> u64 {
        self.hasher.finish()
    }
}

impl Write for GlyphHasher {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.hasher.write(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
