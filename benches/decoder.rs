use criterion::{criterion_group, criterion_main, Criterion};
use std::io::{Cursor, Seek, SeekFrom, Write};
use tty_surface::{Decoder, TTYDecoder};

const SMALL_SET: &str = "\x1bOR\x1b[15~\x1b[97;15R\x1b[4;3104;1482t\x1b[8;101;202t\x1b[<0;24;14M\x1b[<65;142;30M\u{1F431}ABC";

pub fn small_decoder_benchmark(c: &mut Criterion) {
    c.bench_function("smal set", |b| {
        let mut decoder = TTYDecoder::new();
        let mut cursor = Cursor::new(Vec::new());
        cursor.write_all(SMALL_SET.as_ref()).unwrap();

        b.iter(move || {
            cursor.seek(SeekFrom::Start(0)).unwrap();
            while let Some(_event) = decoder.decode(&mut cursor).unwrap() {}
        })
    });
}

criterion_group!(benches, small_decoder_benchmark);
criterion_main!(benches);
