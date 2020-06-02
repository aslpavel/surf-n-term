use rasterize::{surf_to_png, surf_to_ppm, timeit, FillRule, Path, Surface, Transform};
use std::str::FromStr;
pub type Error = Box<dyn std::error::Error>;

pub fn path_load<P: AsRef<std::path::Path>>(path: P) -> Result<Path, Error> {
    use std::{fs::File, io::Read};
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(timeit("[parse]", || Path::from_str(&contents))?)
}

pub const SQUIRREL: &str = "M12 1C9.79 1 8 2.31 8 3.92c0 1.94.5 3.03 0 6.08 0-4.5-2.77-6.34-4-6.34.05-.5-.48-.66-.48-.66s-.22.11-.3.34c-.27-.31-.56-.27-.56-.27l-.13.58S.7 4.29 .68 6.87c.2.33 1.53.6 2.47.43.89.05.67.79.47.99C2.78 9.13 2 8 1 8S0 9 1 9s1 1 3 1c-3.09 1.2 0 4 0 4H3c-1 0-1 1-1 1h6c3 0 5-1 5-3.47 0-.85-.43-1.79 -1-2.53-1.11-1.46.23-2.68 1-2 .77.68 3 1 3-2 0-2.21-1.79-4-4-4zM2.5 6 c-.28 0-.5-.22-.5-.5s.22-.5.5-.5.5.22.5.5-.22.5-.5.5z";

pub const VERIFIED: &str = "M7.67 14.72H8.38L10.1 13H12.5L13 12.5V10.08L14.74 8.36004V7.65004L13.03 5.93004V3.49004L12.53 3.00004H10.1L8.38 1.29004H7.67L6 3.00004H3.53L3 3.50004V5.93004L1.31 7.65004V8.36004L3 10.08V12.5L3.53 13H6L7.67 14.72ZM6.16 12H4V9.87004L3.88 9.52004L2.37 8.00004L3.85 6.49004L4 6.14004V4.00004H6.16L6.52 3.86004L8 2.35004L9.54 3.86004L9.89 4.00004H12V6.14004L12.17 6.49004L13.69 8.00004L12.14 9.52004L12 9.87004V12H9.89L9.51 12.15L8 13.66L6.52 12.14L6.16 12ZM6.73004 10.4799H7.44004L11.21 6.71L10.5 6L7.09004 9.41991L5.71 8.03984L5 8.74984L6.73004 10.4799Z";

pub const NOMOVE: &str = "M50,100 0,50 25,25Z L100,50 75,25Z";
pub const STAR: &str = "M50,0 21,90 98,35 2,35 79,90z M110,0 h90 v90 h-90 z M130,20 h50 v50 h-50 zM210,0  h90 v90 h-90 z M230,20 v50 h50 v-50 z";
pub const ARCS: &str = "M600,350 l 50,-25a80,60 -30 1,1 50,-25 l 50,-25a25,50 -30 0,1 50,-25 l 50,-25a25,75 -30 0,1 50,-25 l 50,-25a25,100 -30 0,1 50,-25 l 50,-25";
pub const TEST: &str = "M-20.0,-10.0 L-10.0,-20.0 L20.0,10.0 L10.0,20.0 Z";

fn main() -> Result<(), Error> {
    env_logger::init();

    // let path = Path::from_str(SQUIRREL)?;
    // let tr = Transform::default().scale(12.0, 12.0);

    let path = path_load("material-big.path")?;
    let tr = Transform::default();

    let mask = timeit("[rasterize]", || path.rasterize(tr, FillRule::EvenOdd)).unwrap();

    // let path = Path::from_str(VERIFIED)?;
    // let tr = Transform::default()
    //     .scale(12.0, 12.0)
    //     .rotate_around(0.523598, Point::new(-8.0, -8.0))
    //     .translate(-1.0, -1.0);
    // let mut mask = SurfaceOwned::new(300, 300);
    // for line in path.flatten(tr, FLATNESS, true) {
    //     rasterize_line(&mut mask, line);
    // }
    // coverage_to_mask(&mut mask, FillRule::EvenOdd);

    println!("{:?}", mask.shape());
    if false {
        let mut image = std::io::BufWriter::new(std::fs::File::create("rasterize.png")?);
        timeit("[save:png]", || surf_to_png(&mask, &mut image))?;
    } else {
        let mut image = std::io::BufWriter::new(std::fs::File::create("rasterize.ppm")?);
        timeit("[save:ppm]", || surf_to_ppm(&mask, &mut image))?;
    }

    Ok(())
}
