use rasterize::RGBA;
use surf_n_term::{ColorPalette, Size, SurfaceMut, SurfaceOwned};

type Error = Box<dyn std::error::Error + Sync + Send + 'static>;

fn load_png(path: impl AsRef<std::path::Path>) -> Result<SurfaceOwned<RGBA>, Error> {
    let decoder = png::Decoder::new(std::fs::File::open(path)?);
    let mut reader = decoder.read_info()?;
    let mut buffer = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buffer)?;
    buffer.truncate(info.buffer_size());
    let size = Size {
        width: info.width as usize,
        height: info.height as usize,
    };
    let mut surf = SurfaceOwned::new(size);
    match info.color_type {
        png::ColorType::Rgb => {
            surf.fill_with(|pos, _| {
                let offset = (pos.row * size.width + pos.col) * 3;
                let rgb = &buffer[offset..offset + 3];
                RGBA::new(rgb[0], rgb[1], rgb[2], 255)
            });
            Ok(surf)
        }
        png::ColorType::Rgba => {
            surf.fill_with(|pos, _| {
                let offset = (pos.row * size.width + pos.col) * 4;
                let rgba = &buffer[offset..offset + 4];
                RGBA::new(rgba[0], rgba[1], rgba[2], rgba[3])
            });
            Ok(surf)
        }
        color_type => return Err(format!("Unsupported color type: {color_type:?}").into()),
    }
}

fn main() -> Result<(), Error> {
    let palette = match std::env::args().collect::<Vec<_>>().as_slice() {
        [_, palette_size, image_path] => {
            let palette_size = palette_size.parse()?;
            let image = load_png(image_path)?;
            ColorPalette::from_image(image, palette_size, RGBA::default())
                .ok_or_else(|| "Empty image".to_owned())?
        }
        args => {
            eprintln!("Usage: {} <palette_size> <impage.png>", args[0]);
            return Ok(());
        }
    };
    for color in palette.colors() {
        println!("{color:?}");
    }

    Ok(())
}
