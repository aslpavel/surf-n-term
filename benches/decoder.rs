use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use std::io::{Cursor, Seek, SeekFrom, Write};
use surf_n_term::decoder::{Decoder, TTYDecoder};

const SMALL_SET: &str = "\x1bOR\x1b[15~\x1b[97;15R\x1b[4;3104;1482t\x1b[8;101;202t\x1b[<0;24;14M\x1b[<65;142;30M\u{1F431}ABC\x1b_Gi=127;OK\x1b\\";

pub fn small_decoder_benchmark(c: &mut Criterion) {
    let mut decoder = TTYDecoder::new();
    let mut cursor = Cursor::new(Vec::new());
    cursor.get_mut().write_all(SMALL_SET.as_ref()).unwrap();
    let count = decoder.decode_into(&mut cursor, &mut Vec::new()).unwrap();

    let mut group = c.benchmark_group("small");
    group.throughput(Throughput::Elements(count as u64));
    group.bench_function("decode", |b| {
        b.iter(|| {
            cursor.seek(SeekFrom::Start(0)).unwrap();
            while let Some(_event) = decoder.decode(&mut cursor).unwrap() {}
        })
    });
    group.finish();
}

criterion_group!(benches, small_decoder_benchmark);
criterion_main!(benches);
