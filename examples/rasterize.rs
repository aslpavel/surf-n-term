use env_logger::Env;
use rasterize::{
    surf_to_ppm, timeit, Align, BBox, Curve, FillRule, LineCap, LineJoin, Path, Point, Scalar,
    Segment, StrokeStyle, Surface, Transform,
};
use std::{
    env, fmt,
    fs::File,
    io::{BufWriter, Read},
};

type Error = Box<dyn std::error::Error>;

#[derive(Debug)]
struct ArgsError(String);

impl ArgsError {
    fn new(err: impl Into<String>) -> Self {
        Self(err.into())
    }
}

impl fmt::Display for ArgsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for ArgsError {}

struct Args {
    input_file: String,
    output_file: String,
    outline: bool,
    width: Option<usize>,
    stroke: Option<Scalar>,
}

fn parse_args() -> Result<Args, Error> {
    let mut result = Args {
        input_file: String::new(),
        output_file: String::new(),
        outline: false,
        width: None,
        stroke: None,
    };
    let mut postional = 0;
    let mut args = env::args();
    let cmd = args.next().unwrap();
    while let Some(arg) = args.next() {
        match arg.as_ref() {
            "-w" => {
                let width = args
                    .next()
                    .ok_or_else(|| ArgsError::new("-w requires argument"))?;
                result.width = Some(width.parse()?);
            }
            "-s" => {
                let stroke = args
                    .next()
                    .ok_or_else(|| ArgsError::new("-s requres argument"))?;
                result.stroke = Some(stroke.parse()?);
            }
            "-o" => {
                result.outline = true;
            }
            _ => {
                postional += 1;
                match postional {
                    1 => result.input_file = arg,
                    2 => result.output_file = arg,
                    _ => return Err(ArgsError::new("unexpected positional argment").into()),
                }
            }
        }
    }
    if postional < 2 {
        eprintln!(
            "Very simple tool that accepts SVG path as an input and produces rasterized image"
        );
        eprintln!("\nUSAGE:");
        eprintln!(
            "    {} [-w <width>] [-s <stroke>] [-o] <file.path> <out.ppm>",
            cmd
        );
        eprintln!("\nARGS:");
        eprintln!("    -w <width>         width in pixels of the output image");
        eprintln!("    -s <stroke_width>  stroke path before rendering");
        eprintln!("    -o                 show outline with control points instead of filling");
        eprintln!("    <file.path>        file containing SVG path");
        eprintln!("    <out.ppm>          image rendered in the PPM format");
        std::process::exit(1);
    }
    Ok(result)
}

fn path_load(path: String) -> Result<Path, Error> {
    let mut contents = String::new();
    if path != "-" {
        let mut file = File::open(path)?;
        file.read_to_string(&mut contents)?;
    } else {
        std::io::stdin().read_to_string(&mut contents)?;
    }
    Ok(timeit("[parse]", || contents.parse())?)
}

fn outline(path: &Path) -> Path {
    let stroke_style = StrokeStyle {
        width: 2.0,
        line_join: LineJoin::Round,
        line_cap: LineCap::Round,
    };
    let control_style = StrokeStyle {
        width: 1.0,
        ..stroke_style
    };
    let control_radius = 3.0;
    let mut output = path.stroke(stroke_style);
    for subpath in path.subpaths().iter() {
        for segment in subpath.segments() {
            let control = match segment {
                Segment::Line(_) => Path::builder(),
                Segment::Quad(quad) => {
                    let [p0, p1, p2] = quad.points();
                    Path::builder()
                        .move_to(p0)
                        .line_to(p1)
                        .circle(control_radius)
                        .line_to(p2)
                }
                Segment::Cubic(cubic) => {
                    let [p0, p1, p2, p3] = cubic.points();
                    Path::builder()
                        .move_to(p0)
                        .line_to(p1)
                        .circle(control_radius)
                        .move_to(p3)
                        .line_to(p2)
                        .circle(control_radius)
                }
            };
            output.extend(
                Path::builder()
                    .move_to(segment.start())
                    .circle(control_radius)
                    .build(),
            );
            output.extend(control.build().stroke(control_style));
        }
        if (subpath.start() - subpath.end()).length() > control_radius {
            output.extend(
                Path::builder()
                    .move_to(subpath.end())
                    .circle(control_radius)
                    .build(),
            );
        }
    }
    output
}

fn main() -> Result<(), Error> {
    env_logger::from_env(Env::default().default_filter_or("debug")).init();
    let args = parse_args()?;

    let mut path = path_load(args.input_file)?;
    match args.width {
        Some(width) if width > 2 => {
            let src_bbox = path
                .bbox(Transform::default())
                .ok_or_else(|| ArgsError::new("path is empty"))?;
            let width = width as Scalar;
            let height = src_bbox.height() * width / src_bbox.width();
            let dst_bbox = BBox::new(Point::new(1.0, 1.0), Point::new(width - 1.0, height - 1.0));
            path.transform(Transform::fit(src_bbox, dst_bbox, Align::Mid));
        }
        _ => (),
    }
    log::info!("[path::segments_count] {}", path.segments_count());
    if let Some(stroke) = args.stroke {
        path = timeit("[stroke]", || {
            path.stroke(StrokeStyle {
                width: stroke,
                line_join: LineJoin::Round,
                line_cap: LineCap::Round,
            })
        });
        log::info!("[stroke::segments_count] {}", path.segments_count());
    }
    if args.outline {
        path = outline(&path);
    }

    let mask = timeit("[rasterize]", || {
        path.rasterize(Transform::default(), FillRule::NonZero)
    });
    log::info!(
        "[dimension] width: {} height: {}",
        mask.width(),
        mask.height()
    );

    if args.output_file != "-" {
        let mut image = BufWriter::new(File::create(args.output_file)?);
        timeit("[save:ppm]", || surf_to_ppm(&mask, &mut image))?;
    } else {
        timeit("[save:ppm]", || surf_to_ppm(&mask, std::io::stdout()))?;
    }

    Ok(())
}
