use env_logger::Env;
use rasterize::{surf_to_png, timeit, FillRule, Path, Transform};
use std::{
    env,
    fs::File,
    io::{BufWriter, Read},
};

type Error = Box<dyn std::error::Error>;

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

fn main() -> Result<(), Error> {
    env_logger::from_env(Env::default().default_filter_or("debug")).init();

    let mut args = env::args();
    if args.len() != 3 {
        eprintln!("Usage: rasterize <file.path> <out.png>");
        return Ok(());
    }
    let _cmd = args.next().unwrap();
    let path_filename = args.next().unwrap();
    let output_filename = args.next().unwrap();

    let tr = Transform::default();
    let path = path_load(path_filename)?;
    let mask = timeit("[rasterize]", || path.rasterize(tr, FillRule::NonZero));

    if output_filename != "-" {
        let mut image = BufWriter::new(File::create(output_filename)?);
        timeit("[save:png]", || surf_to_png(&mask, &mut image))?;
    } else {
        timeit("[save:png]", || surf_to_png(&mask, std::io::stdout()))?;
    }

    Ok(())
}
