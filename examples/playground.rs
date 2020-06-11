#![allow(unused_imports, dead_code)]
use env_logger::Env;
use rasterize::{
    surf_to_png, timeit, Cubic, Curve, FillRule, Line, LineCap, LineJoin, Path, Quad, StrokeStyle,
    SubPath, Transform,
};
use std::{
    env,
    fs::File,
    io::{BufWriter, Read},
};

type Error = Box<dyn std::error::Error>;

fn path_load<P: AsRef<std::path::Path>>(path: P) -> Result<Path, Error> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(timeit("[parse]", || contents.parse())?)
}

// TODO: add support for "-" filenames
fn main() -> Result<(), Error> {
    env_logger::from_env(Env::default().default_filter_or("debug")).init();
    let path = path_load("./paths/squirrel.path")?;

    let stroke_style = StrokeStyle {
        width: 1.0,
        line_join: LineJoin::Miter(4.0),
        line_cap: LineCap::Round,
    };
    let stroke = path.stroke(stroke_style);
    println!("{}", stroke.to_string());

    Ok(())
}
