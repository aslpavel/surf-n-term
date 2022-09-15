use super::{BoxConstraint, Layout, Tree, View, ViewContext};
use crate::{
    Cell, Error, Face, Image, Position, Size, Surface, SurfaceMut, SurfaceOwned, TerminalSurface,
    RGBA,
};
use rasterize::{
    BBox, FillRule, Image as RImage, LinColor, LineCap, LineJoin, PathBuilder, Point, Scalar,
    Scene, StrokeStyle, Transform,
};
use std::sync::Arc;

/// Create a frame with rounded corners and a border around a view
pub struct Frame<V> {
    view: V,
    color: RGBA,
    border_color: RGBA,
    border_width: Scalar,
    broder_radius: Scalar,
    // offset: Scalar,
}

type Fragments = Arc<[Cell; 9]>;
// const FRAGMENTS: Mutex<HashMap<(Size, RGBA, RGBA), Fragments>> = Mutex::new(HashMap::new());

impl<V> Frame<V> {
    /// Create new frame
    ///
    /// `border_width` is a fraction of the minimal dimension of of the cell and
    /// it will be rounded to the pixel size.
    /// `border_radius` is a fraction of the maximum possible radius.
    pub fn new(
        view: V,
        color: RGBA,
        border_color: RGBA,
        border_width: Scalar,
        border_radius: Scalar,
    ) -> Self {
        Self {
            view,
            color,
            border_color,
            border_width: border_width.clamp(0.0, 1.0),
            broder_radius: border_radius.clamp(0.0, 1.0),
        }
    }

    /// Generate image fragments of the frame
    fn fragments(&self, ctx: &ViewContext) -> Fragments {
        // size of a single cell
        let height = ctx.pixels_per_cell.height as Scalar;
        let width = ctx.pixels_per_cell.width as Scalar;

        // calculate border and radius size
        let total = height.min(width);
        let border = (total * self.border_width).round();
        let radius = (total - border / 2.0) * self.broder_radius;

        // offset from the left-top corner
        let offset = if radius < total / 2.0 {
            // try to keep in the middle of the cell if radius is small enough
            (total / 2.0).round() + border / 2.0
        } else {
            (total - radius).round() + border / 2.0
        };

        // draw 3x3 box which represent all the pieces that we need to render frame
        let size = Point::new(width * 3.0, height * 3.0);
        let path = Arc::new(
            PathBuilder::new()
                .move_to((offset, offset))
                .rbox(size - Point::new(offset, offset), (radius, radius))
                .build(),
        );

        // create and render scene
        let scene = Scene::group(vec![
            Scene::fill(
                path.clone(),
                Arc::new(LinColor::from(self.color)),
                FillRule::default(),
            ),
            Scene::stroke(
                path,
                Arc::new(LinColor::from(self.border_color)),
                StrokeStyle {
                    width: border,
                    line_join: LineJoin::Round,
                    line_cap: LineCap::Round,
                },
            ),
        ]);
        let image = scene.render(
            &rasterize::ActiveEdgeRasterizer::default(),
            Transform::identity(),
            Some(BBox::new((0.0, 0.0), size)),
            None,
        );

        // cut image into 9 pieces
        let h = ctx.pixels_per_cell.height;
        let w = ctx.pixels_per_cell.width;
        let mut fragments: [Cell; 9] = Default::default();
        for x in 0..3 {
            for y in 0..3 {
                let img = rimage_to_image(image.view(y, h * (y + 1) + 1, x, w * (x + 1) + 1));
                fragments[x + y * 3] = Cell::new_image(img);
            }
        }

        Arc::new(fragments)
    }
}

impl<V: View> View for Frame<V> {
    fn render<'a>(
        &self,
        ctx: &ViewContext,
        surf: &'a mut TerminalSurface<'a>,
        layout: &Tree<Layout>,
    ) -> Result<(), Error> {
        let mut surf = layout.apply_to(surf);

        // TODO: cache this!
        let fragments = self.fragments(&ctx);

        let empty = Cell::new(Face::new(None, Some(self.color), Default::default()), None);
        for col in 0..surf.width() {
            for row in 0..surf.height() {
                let xi = fragment_index(col, surf.width());
                let yi = fragment_index(row, surf.height());
                if let Some(cell) = surf.get_mut(row, col) {
                    if xi == 1 && yi == 1 {
                        *cell = empty.clone();
                    } else {
                        *cell = fragments[xi + yi * 3].clone();
                    }
                }
            }
        }

        self.view
            .render(ctx, &mut surf, layout.get(0).ok_or(Error::InvalidLayout)?)?;
        Ok(())
    }

    fn layout(&self, ctx: &ViewContext, ct: BoxConstraint) -> Tree<Layout> {
        let ct = BoxConstraint::new(
            Size {
                height: ct.min().height.saturating_sub(2),
                width: ct.min().width.saturating_sub(2),
            },
            Size {
                height: ct.max().height.saturating_sub(2),
                width: ct.max().width.saturating_sub(2),
            },
        );
        let mut layout = self.view.layout(ctx, ct);
        layout.set_pos(Position::new(1, 1));
        let size = Size {
            height: layout.size().height + 2,
            width: layout.size().width + 2,
        };
        Tree::new(Layout::new().with_size(size), vec![layout])
    }
}

/// Convert rasterized image into an Image
fn rimage_to_image(image: impl RImage<Pixel = LinColor>) -> Image {
    let data: Vec<RGBA> = image.iter().map(|c| RGBA::from(*c)).collect();
    Image::new(SurfaceOwned::from_vec(image.height(), image.width(), data))
}

/// Find an index given the size of the dimension and the offset
fn fragment_index(index: usize, size: usize) -> usize {
    if index == 0 {
        0
    } else if index + 1 < size {
        1
    } else {
        2
    }
}
