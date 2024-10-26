use super::{
    BoxConstraint, IntoView, Layout, Tree, TreeMut, View, ViewContext, ViewLayout, ViewLayoutStore,
    ViewMutLayout,
};
use crate::{
    common::LockExt, surface::ViewBounds, Cell, Error, Shape, Size, Surface, SurfaceMut,
    SurfaceMutView, SurfaceView, TerminalSurface,
};
use std::{
    ops::{Deref, DerefMut},
    sync::{Arc, RwLock},
};

struct OffscreenInner {
    ct: BoxConstraint,
    layout_store: ViewLayoutStore,
    shape: Shape,
    cells: Vec<Cell>,
}

/// View that holds offscreen surface, any view can be rendered into it,
/// the size of the buffer is automatically calculated based on layout
/// and constraints.
#[derive(Clone)]
pub struct Offscreen {
    inner: Arc<RwLock<OffscreenInner>>,
}

impl Default for Offscreen {
    fn default() -> Self {
        Self::new()
    }
}

impl Offscreen {
    /// Create new empty offscreen surface
    pub fn new() -> Self {
        let inner = OffscreenInner {
            ct: BoxConstraint::tight(Size::empty()),
            layout_store: ViewLayoutStore::new(),
            shape: Shape::from(Size::empty()),
            cells: Vec::default(),
        };
        Self {
            inner: Arc::new(RwLock::new(inner)),
        }
    }

    /// Acquire surface (inner read lock is held while it is alive)
    pub fn surf(&self) -> impl Surface<Item = Cell> + '_ {
        OffscreenSurface {
            guard: self.inner.read().expect("lock poisoned"),
        }
    }

    /// Acquire mutable surface (inner write lock is held while it is alive)
    pub fn surf_mut(&self) -> impl SurfaceMut<Item = Cell> + '_ {
        OffscreenSurface {
            guard: self.inner.write().expect("lock poisoned"),
        }
    }

    /// Create view restricted by rows and cols bounds
    pub fn view<RS, CS>(&self, rows: RS, cols: CS) -> OffscreenView
    where
        RS: ViewBounds,
        CS: ViewBounds,
    {
        OffscreenView {
            shape: self.inner.with(|inner| inner.shape).view(rows, cols),
            offscreen: self.clone(),
        }
    }

    /// Draw view on the internal surface, space is allocated based on constraints
    /// and layout of the provided view.
    pub fn draw_view(
        &self,
        ctx: &ViewContext,
        ct: BoxConstraint,
        view: impl IntoView,
    ) -> Result<(), Error> {
        let view = view.into_view();
        let mut inner = self.inner.write().expect("lock poisoned");
        inner.ct = ct;

        // layout
        let mut layout_store = std::mem::take(&mut inner.layout_store);
        layout_store.clear();
        let mut layout = ViewMutLayout::new(&mut layout_store, Layout::default());
        view.layout(ctx, ct, layout.view_mut())?;

        // allocate surface
        inner.cells.clear();
        let size = ct.clamp(layout.size());
        inner.cells.resize_with(size.area(), Cell::default);
        inner.shape = Shape::from(size);
        let surf = SurfaceMutView::new(inner.shape, &mut inner.cells);

        // render
        view.render(ctx, surf, layout.view())?;
        inner.layout_store = layout_store;

        Ok(())
    }
}

struct OffscreenSurface<G> {
    guard: G,
}

impl<G> Surface for OffscreenSurface<G>
where
    G: Deref<Target = OffscreenInner>,
{
    type Item = Cell;

    fn shape(&self) -> Shape {
        self.guard.shape
    }

    fn data(&self) -> &[Self::Item] {
        &self.guard.cells
    }
}

impl<G> SurfaceMut for OffscreenSurface<G>
where
    G: DerefMut<Target = OffscreenInner>,
{
    fn data_mut(&mut self) -> &mut [Self::Item] {
        &mut self.guard.cells
    }
}

impl IntoView for &Offscreen {
    type View = OffscreenView;

    fn into_view(self) -> Self::View {
        self.view(.., ..)
    }
}

pub struct OffscreenView {
    shape: Shape,
    offscreen: Offscreen,
}

impl View for OffscreenView {
    fn render(
        &self,
        ctx: &ViewContext,
        surf: TerminalSurface<'_>,
        layout: ViewLayout<'_>,
    ) -> Result<(), Error> {
        let src_surf = self.offscreen.surf();
        SurfaceView::new(self.shape, src_surf.data()).render(ctx, surf, layout)
    }

    fn layout(
        &self,
        ctx: &ViewContext,
        ct: BoxConstraint,
        layout: ViewMutLayout<'_>,
    ) -> Result<(), Error> {
        let src_surf = self.offscreen.surf();
        SurfaceView::new(self.shape, src_surf.data()).layout(ctx, ct, layout)
    }
}
