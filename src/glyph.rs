use crate::{Color, ColorLinear, Image, Size, Surface, SurfaceMut, SurfaceOwned};
pub use rasterize::FillRule;
use rasterize::{
    Align, BBox, Image as _, ImageMutRef, Path, Shape as ImageShape, SignedDifferenceRasterizer,
    Transform,
};
use std::{
    hash::{Hash, Hasher},
    io::Write,
    sync::Arc,
};

pub struct Glyph {
    /// Path scaled to fit in 1000x1000 grid
    path: Arc<Path>,
    /// Fill rule
    fill_rule: FillRule,
    /// Hash of the glyph
    hash: u64,
}

impl PartialEq for Glyph {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash && self.fill_rule == other.fill_rule
    }
}

impl Eq for Glyph {}

impl Hash for Glyph {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash.hash(state);
        self.fill_rule.hash(state);
    }
}

impl Glyph {
    pub fn new(mut path: Path, fill_rule: FillRule) -> Self {
        if let Some(src_bbox) = path.bbox(Transform::default()) {
            let dst_bbox = BBox::new((0.0, 0.0), (1000.0, 1000.0));
            let tr = Transform::fit_bbox(src_bbox, dst_bbox, Align::Mid);
            path.transform(tr);
        }

        let mut hasher = GlyphHasher::new();
        path.write_svg_path(&mut hasher)
            .expect("in memory write failed");
        let hash = hasher.finish();

        Self {
            path: Arc::new(path),
            hash,
            fill_rule,
        }
    }

    pub fn rasterize(&self, fg: impl Color, bg: impl Color, size: Size) -> Image {
        let mut surf = SurfaceOwned::new(size.height, size.width);
        let mut img = ImageMutRef::new(
            ImageShape {
                width: surf.width(),
                height: surf.height(),
                row_stride: surf.width(),
                col_stride: 1,
            },
            surf.data_mut(),
        );
        let rasterizer = SignedDifferenceRasterizer::default();
        let tr = Transform::fit_size(
            BBox::new((0.0, 0.0), (1000.0, 1000.0)),
            img.size(),
            Align::Mid,
        );
        self.path.mask(rasterizer, tr, self.fill_rule, &mut img);
        let fg: ColorLinear = fg.into();
        let bg: ColorLinear = bg.into();
        let img = surf.map(|_, _, t| bg.lerp(fg, *t).into());
        Image::new(img)
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
