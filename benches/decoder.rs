use criterion::{Bencher, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use surf_n_term::{
    common::Rnd,
    decoder::{Base64Decoder, Decoder, TTYEventDecoder},
    encoder::Base64Encoder,
};

const SMALL_SET: &str = "\x1bOR\x1b[15~\x1b[97;15R\x1b[4;3104;1482t\x1b[8;101;202t\x1b[<0;24;14M\x1b[<65;142;30M\u{1F431}ABC\x1b_Gi=127;OK\x1b\\";

fn small_decoder_benchmark(c: &mut Criterion) {
    let mut decoder = TTYEventDecoder::new();
    let mut cursor = Cursor::new(Vec::new());
    cursor.get_mut().write_all(SMALL_SET.as_ref()).unwrap();
    let count = decoder.decode_into(&mut cursor, &mut Vec::new()).unwrap();

    let mut group = c.benchmark_group("small_decoder");
    group
        .throughput(Throughput::Elements(count as u64))
        .bench_function("decode", |b| {
            b.iter(|| {
                cursor.seek(SeekFrom::Start(0)).unwrap();
                while let Some(_event) = decoder.decode(&mut cursor).unwrap() {}
            })
        });
    group.finish();
}

fn random_vec(size: usize) -> Vec<u8> {
    let mut vec = Vec::with_capacity(size);
    let mut rnd = Rnd::new();
    while vec.len() < size {
        vec.write_all(&rnd.next_u8x4())
            .expect("in memory write failed")
    }
    vec
}

fn base64_encode(b: &mut Bencher, &size: &usize) {
    let data = random_vec(size);
    let mut out = Vec::with_capacity(size * 2);
    b.iter(|| {
        out.clear();
        let mut encoder = Base64Encoder::new(&mut out);
        encoder.write_all(&data).unwrap();
        encoder.finish().unwrap();
    })
}

fn base64_decode(b: &mut Bencher, &size: &usize) {
    let mut data = Vec::with_capacity(size);
    let mut encoder = Base64Encoder::new(&mut data);
    encoder.write_all(&random_vec(size)).unwrap();
    encoder.finish().unwrap();
    let mut out = Vec::with_capacity(size);
    b.iter(|| {
        out.clear();
        let mut decoder = Base64Decoder::new(data.as_slice());
        decoder.read_to_end(&mut out).unwrap();
    })
}

const SIZES: [usize; 4] = [128, 512, 3 * 1024, 1024 * 1024];

fn base64_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("base64 encode");
    for size in SIZES {
        group
            .throughput(Throughput::Bytes(size as u64))
            .bench_with_input(BenchmarkId::new("stream", size), &size, base64_encode);
    }
    group.finish();

    let mut group = c.benchmark_group("base64 decode");
    for size in SIZES {
        group
            .throughput(Throughput::Bytes(size as u64))
            .bench_with_input(BenchmarkId::new("stream", size), &size, base64_decode);
    }
    group.finish();
}

criterion_group!(benches, small_decoder_benchmark, base64_benchmark);
criterion_main!(benches);
