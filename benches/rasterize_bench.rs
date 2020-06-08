use criterion::{criterion_group, criterion_main, Criterion};
use rasterize::{
    render::{Cubic, FLATNESS},
    FillRule, Path, SurfaceMut, Transform,
};
use std::{fs::File, io::Read, time::Duration};

fn curve_benchmark(c: &mut Criterion) {
    let path = Cubic::new((157.0, 67.0), (35.0, 200.0), (220.0, 260.0), (175.0, 45.0));
    c.bench_function("cubic extremities", |b| b.iter(|| path.extremities()));
}

fn path_benchmark(c: &mut Criterion) {
    let tr = Transform::default();
    // load path
    let mut path_str = String::new();
    let mut file = File::open("paths/material-big.path").expect("failed to open a path");
    file.read_to_string(&mut path_str)
        .expect("failed to read path");
    // parse path
    let path: Path = path_str.parse().unwrap();
    // rasterize path
    let mut surf = path.rasterize(tr, FillRule::EvenOdd);

    c.bench_function("path parse", |b| {
        b.iter_with_large_drop(|| path_str.parse::<Path>())
    });

    c.bench_function("path flatten", |b| {
        b.iter(|| path.flatten(tr, FLATNESS, true).count())
    });

    c.bench_function("path bbox", |b| b.iter(|| path.bbox(tr)));

    c.bench_function("path rasterize", |b| {
        b.iter_with_large_drop(|| path.rasterize(tr, FillRule::EvenOdd))
    });

    c.bench_function("path rasterize to", |b| {
        b.iter(|| {
            surf.clear();
            path.rasterize_to(tr, FillRule::EvenOdd, &mut surf);
        })
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10).warm_up_time(Duration::new(1, 0));
    targets = curve_benchmark, path_benchmark
);
criterion_main!(benches);
