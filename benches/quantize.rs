use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use std::{
    fs::File,
    io::{BufRead, BufReader, Read},
    path::Path,
};
use surf_n_term::{
    common::{clamp, Random},
    Color, LinColor, Size,
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
    Ok(Image::new(SurfaceOwned::from_vec(
        Size::new(height, width),
        colors,
    )))
}

fn palette_benchmark(c: &mut Criterion) {
    let img = load_ppm("benches/flamingo.ppm").expect("failed to load flamingo.ppm");
    let img_colors: Vec<_> = img.iter().copied().collect();
    let bg = RGBA::new(0, 0, 0, 255);
    let p128 = ColorPalette::from_image(&img, 128, bg).unwrap();
    let p256 = ColorPalette::from_image(&img, 256, bg).unwrap();
    let p512 = ColorPalette::from_image(&img, 512, bg).unwrap();

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

fn srgb_to_linear(color: RGBA) -> LinColor {
    fn s2l(x: f32) -> f32 {
        if x <= 0.04045 {
            x / 12.92
        } else {
            ((x + 0.055) / 1.055).powf(2.4)
        }
    }

    let [r, g, b, a] = color.to_rgba();
    let a = (a as f32) / 255.0;
    let r = s2l((r as f32) / 255.0) * a;
    let g = s2l((g as f32) / 255.0) * a;
    let b = s2l((b as f32) / 255.0) * a;
    LinColor::new(r, g, b, a)
}

fn linear_to_srgb(color: LinColor) -> RGBA {
    fn l2s(x: f32) -> f32 {
        if x <= 0.0031308 {
            x * 12.92
        } else {
            1.055 * x.powf(1.0 / 2.4) - 0.055
        }
    }

    let [r, g, b, a]: [f32; 4] = color.into();
    if a < std::f32::EPSILON {
        RGBA::new(0, 0, 0, 0)
    } else {
        let a = clamp(a, 0.0, 1.0);
        let r = (l2s(r / a) * 255.0).round() as u8;
        let g = (l2s(g / a) * 255.0).round() as u8;
        let b = (l2s(b / a) * 255.0).round() as u8;
        let a = (a * 255.0) as u8;
        RGBA::new(r, g, b, a)
    }
}

fn srgb_and_linear_benchmark(c: &mut Criterion) {
    let colors: Vec<_> = RGBA::random_iter().take(1024).collect();

    for color in colors.iter() {
        assert_eq!(*color, linear_to_srgb(srgb_to_linear(*color)));
    }

    let mut group = c.benchmark_group("srgb_and_linear");
    group.sampling_mode(SamplingMode::Flat);
    group.throughput(Throughput::Elements(1024 as u64));
    group.bench_function("naive", |b| {
        b.iter(|| {
            for color in colors.iter() {
                black_box(linear_to_srgb(black_box(srgb_to_linear(*color))));
            }
        })
    });
    group.bench_function("fast", |b| {
        b.iter(|| {
            for color in colors.iter() {
                black_box(RGBA::from(black_box(LinColor::from(*color))));
            }
        })
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default(); // .sample_size(30).warm_up_time(Duration::new(2, 0));
    targets = palette_benchmark, srgb_and_linear_benchmark,
);
criterion_main!(benches);
