use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use std::{
    fs::File,
    io::{BufRead, BufReader, Read},
    path::Path,
    time::Duration,
};
use surf_n_term::{ColorPalette, Image, Surface, SurfaceOwned, RGBA};

/// This not really a ppm loader it is not doing proper validation or anything
fn load_ppm(path: impl AsRef<Path>) -> Result<Image, Box<dyn std::error::Error>> {
    let mut file = BufReader::new(File::open(path)?);

    // specially constructed ppm header to end with a new line
    let mut header = String::new();
    file.read_line(&mut header)?;
    let (height, width) = match header.trim().split(' ').collect::<Vec<_>>().as_slice() {
        ["P6", width, height, "255"] => (height.parse::<usize>()?, width.parse::<usize>()?),
        _ => panic!("bad ppm file header: \"{}\"", header),
    };

    // load colors
    let mut colors = Vec::with_capacity(height * width);
    let mut color = [0u8; 3];
    for _ in 0..height * width {
        file.read_exact(&mut color)?;
        let [r, g, b] = color;
        colors.push(RGBA::new(r, g, b, 255));
    }
    Ok(Image::new(SurfaceOwned::from_vec(height, width, colors)))
}

fn palette_benchmark(c: &mut Criterion) {
    let img = load_ppm("benches/flamingo.ppm").expect("failed to load flamingo.ppm");
    let img_colors: Vec<_> = img.iter().copied().collect();
    let p128 = ColorPalette::from_image(&img, 128).unwrap();
    let p256 = ColorPalette::from_image(&img, 256).unwrap();
    let p512 = ColorPalette::from_image(&img, 512).unwrap();

    let mut group = c.benchmark_group("palette");
    group.sampling_mode(SamplingMode::Flat);
    group.throughput(Throughput::Elements(img_colors.len() as u64));
    for palette in [p128, p256, p512].iter() {
        group.bench_with_input(
            BenchmarkId::new("kd-tree", palette.size()),
            palette,
            |b, p| {
                b.iter(|| {
                    for color in img_colors.iter() {
                        p.find(*color);
                    }
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("naive", palette.size()),
            palette,
            |b, p| {
                b.iter(|| {
                    for color in img_colors.iter() {
                        p.find_naive(*color);
                    }
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(30).warm_up_time(Duration::new(2, 0));
    targets = palette_benchmark,
);
criterion_main!(benches);
