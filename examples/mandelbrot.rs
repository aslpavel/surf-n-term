use std::{
    cmp::min,
    ops::{Add, Mul, Range},
    time::Duration,
};
use surf_n_term::{
    view::ViewContext, Color, Error, Image, LinColor, Size, Surface, SurfaceOwned, SystemTerminal,
    Terminal, TerminalAction, TerminalCommand, TerminalEvent, TerminalSurfaceExt,
};

#[derive(Copy, Clone, Default)]
struct Complex {
    x: f64,
    y: f64,
}

impl Complex {
    fn sabs(self) -> f64 {
        self.x * self.x + self.y * self.y
    }
}

impl Mul for Complex {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        Complex {
            x: self.x * other.x - self.y * other.y,
            y: self.x * other.y + self.y * other.x,
        }
    }
}

impl Add for Complex {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Complex {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

fn mandelbrot_at(c: Complex, max_iter: usize) -> usize {
    (0..max_iter)
        .try_fold(Complex::default(), |z, i| {
            if z.sabs() >= 4.0 {
                Err(i)
            } else {
                Ok(z * z + c)
            }
        })
        .err()
        .unwrap_or(max_iter)
}

impl Default for ColorMap {
    fn default() -> Self {
        let colors = vec![
            LinColor::new(0.001462, 0.000466, 0.013866, 1.0),
            LinColor::new(0.039608, 0.03109, 0.133515, 1.0),
            LinColor::new(0.113094, 0.065492, 0.276784, 1.0),
            LinColor::new(0.211718, 0.061992, 0.418647, 1.0),
            LinColor::new(0.316654, 0.07169, 0.48538, 1.0),
            LinColor::new(0.414709, 0.110431, 0.504662, 1.0),
            LinColor::new(0.512831, 0.148179, 0.507648, 1.0),
            LinColor::new(0.613617, 0.181811, 0.498536, 1.0),
            LinColor::new(0.716387, 0.214982, 0.47529, 1.0),
            LinColor::new(0.816914, 0.255895, 0.436461, 1.0),
            LinColor::new(0.904281, 0.31961, 0.388137, 1.0),
            LinColor::new(0.960949, 0.418323, 0.35963, 1.0),
            LinColor::new(0.9867, 0.535582, 0.38221, 1.0),
            LinColor::new(0.996096, 0.653659, 0.446213, 1.0),
            LinColor::new(0.996898, 0.769591, 0.534892, 1.0),
            LinColor::new(0.99244, 0.88433, 0.640099, 1.0),
        ];
        Self { colors }
    }
}

struct ColorMap {
    colors: Vec<LinColor>,
}

impl ColorMap {
    fn lookup(&self, value: f64) -> LinColor {
        let offset = value.clamp(0.0, 1.0) * (self.colors.len() - 1) as f64;
        let index = offset.floor() as usize;
        let fract = offset.fract();
        let start = self.colors[index];
        let end = self.colors[min(index + 1, self.colors.len() - 1)];
        start.lerp(end, fract as f32)
    }
}

fn mandelbrot(xs: Range<f64>, ys: Range<f64>, size: Size, max_iter: usize) -> SurfaceOwned<usize> {
    let height = (size.height - 1) as f64;
    let width = (size.width - 1) as f64;
    SurfaceOwned::new_with(size, |pos| {
        let x = lerp(&xs, (pos.col as f64 + 0.5) / width);
        let y = lerp(&ys, (pos.row as f64 + 0.5) / height);
        mandelbrot_at(Complex { x, y }, max_iter)
    })
}

fn mandlebrot_imgs(
    mand: impl Surface<Item = usize>,
    colormap: &ColorMap,
    count: usize,
) -> Vec<Image> {
    let mut imgs = Vec::new();
    for iter in 0..count {
        let surf = SurfaceOwned::new_with(mand.size(), |pos| {
            let ratio = min(*mand.get(pos).unwrap(), iter) as f64 / count as f64;
            colormap.lookup(ratio).into()
        });
        imgs.push(Image::from(surf));
    }
    imgs
}

fn lerp(vs: &Range<f64>, t: f64) -> f64 {
    vs.start * (1.0 - t) + vs.end * t
}

fn main() -> Result<(), Error> {
    let f = TerminalEvent::Key("f".parse()?);
    let q = TerminalEvent::Key("q".parse()?);
    let ctrl_c = TerminalEvent::Key("ctrl+c".parse()?);
    let mut term = SystemTerminal::new()?;
    term.duplicate_output("/tmp/surf_n_term.log")?;

    // init
    term.execute_many([
        TerminalCommand::CursorSave,
        TerminalCommand::visible_cursor_set(false),
        TerminalCommand::altscreen_set(true),
    ])?;

    let mut imgs = Vec::new();
    let xs = -2.5..1.0;
    let ys = -1.0..1.0;
    let colormap = ColorMap::default();
    let mut size = Size {
        height: 58,
        width: 100,
    };
    let max_iter = 100;

    // loop
    let mut index = 0;
    let mut ascii = true;
    let delay = Duration::from_millis(16);
    term.run_render(move |term, event, mut view| -> Result<_, Error> {
        if imgs.len() < max_iter {
            let mand = mandelbrot(xs.clone(), ys.clone(), size, max_iter);
            imgs = mandlebrot_imgs(mand, &colormap, max_iter);
        };

        let ctx = ViewContext::new(term)?;
        if ascii {
            view.draw_view(&ctx, &imgs[index].ascii_view())?;
        } else {
            view.draw_view(&ctx, &imgs[index])?;
        }
        index = (index + 1) % max_iter;

        // process event
        if event.as_ref() == Some(&q) || event.as_ref() == Some(&ctrl_c) {
            Ok(TerminalAction::Quit(()))
        } else {
            if event.as_ref() == Some(&f) {
                ascii = !ascii;
                size = if ascii {
                    Size {
                        height: 58,
                        width: 100,
                    }
                } else {
                    Size {
                        height: 486,
                        width: 800,
                    }
                };
                imgs.clear();
            }
            Ok(TerminalAction::Sleep(delay))
        }
    })?;

    // clean up
    term.execute_many([
        TerminalCommand::altscreen_set(false),
        TerminalCommand::CursorRestore,
    ])?;
    term.poll(Some(Duration::from_millis(50)))?;

    Ok(())
}
