use super::{BoxConstraint, Layout, Tree, View, ViewContext};
use crate::{Error, TerminalSurface};

/// Widget that changes depending on constraints that it was given.
///
/// Build function is called during [View::layout] which generates a [View]
/// that is used for layout and rendering.
pub struct Dynamic<B> {
    build: B,
}

impl<B, V> Dynamic<B>
where
    B: Fn(&ViewContext, BoxConstraint) -> V + Send + Sync,
    V: View + 'static,
{
    pub fn new(build: B) -> Self {
        Self { build }
    }
}

impl<B, V> View for Dynamic<B>
where
    B: Fn(&ViewContext, BoxConstraint) -> V + Send + Sync,
    V: View + 'static,
{
    fn render(
        &self,
        ctx: &ViewContext,
        surf: TerminalSurface<'_>,
        layout: &Tree<Layout>,
    ) -> Result<(), Error> {
        let view = layout.data::<V>().ok_or(Error::InvalidLayout)?;
        view.render(ctx, surf, layout)?;
        Ok(())
    }

    fn layout(&self, ctx: &ViewContext, ct: BoxConstraint) -> Tree<Layout> {
        let view = (self.build)(ctx, ct);
        let mut layout = view.layout(ctx, ct);
        layout.set_data(view);
        layout
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        view::{tests::render, Align, Container},
        Position, Size, Surface, RGBA,
    };

    #[test]
    fn text_dynamic() -> Result<(), Error> {
        let green = "#00ff00".parse::<RGBA>()?;
        let red = "#ff0000".parse::<RGBA>()?;
        let ctx = &ViewContext::dummy();

        let view = &Dynamic::new(|_ctx, ct| {
            Container::new(if ct.max().width > ct.max().height * 2 {
                green
            } else {
                red
            })
            .with_horizontal(Align::Expand)
            .with_vertical(Align::Expand)
        });

        let green_size = Size::new(4, 10);
        println!("{:?}", view.debug(green_size));
        let surf = render(ctx, view, green_size)?;
        assert_eq!(surf.get(Position::origin()).unwrap().face().bg, Some(green));

        let red_size = Size::new(6, 10);
        println!("{:?}", view.debug(red_size));
        let surf = render(ctx, view, red_size)?;
        assert_eq!(surf.get(Position::origin()).unwrap().face().bg, Some(red));

        Ok(())
    }
}
