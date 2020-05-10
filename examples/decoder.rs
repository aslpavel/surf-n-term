use std::{
    boxed::Box,
    error::Error,
    io::{Cursor, Seek, SeekFrom, Write},
};
use surf_n_term::{Decoder, decoder::TTYDecoder};

const SMALL_SET: &str = "\x1bOR\x1b[15~\x1b[97;15R\x1b[4;3104;1482t\x1b[8;101;202t\x1b[<0;24;14M\x1b[<65;142;30M\u{1F431}ABC";

fn main() -> Result<(), Box<dyn Error>> {
    let mut decoder = TTYDecoder::new();
    let mut cursor = Cursor::new(Vec::new());
    cursor.write_all(SMALL_SET.as_ref())?;

    for _ in 0..2 {
        cursor.seek(SeekFrom::Start(0)).unwrap();
        while let Some(event) = decoder.decode(&mut cursor).unwrap() {
            println!("{:?}", event);
        }
        println!();
    }

    Ok(())
}
