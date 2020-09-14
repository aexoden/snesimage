use std::error::Error;

use log::info;
use rand::Rng;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::rect::Point;

extern crate image;

pub mod config;
pub mod util;

/*
Ideas:
   - Initially quantize the source image to the SNES color space, since that can't be exceeded anyway.
   - Separate the image into N groups of tiles, and then optimize those sets of tiles.
   - Figure out how to combine the above with dithering.
   - Potentially limit the degree of change, so the palette doesn't end up all over the place anytime the p value is still > 0.
*/

struct OptimizedImage {
    original: image::RgbaImage,
    tile_palettes: Vec<u8>,
    palette: Vec<image::Rgba<u8>>,
    palette_map: Vec<u8>,
    palette_size: usize,
    palette_count: usize,
}

fn color_distance(color1: &image::Rgba<u8>, color2: &image::Rgba<u8>) -> f64 {
    ((color1[0] as f64 - color2[0] as f64).powf(2.0)
        + (color1[1] as f64 - color2[1] as f64).powf(2.0)
        + (color1[2] as f64 - color2[2] as f64).powf(2.0))
    .sqrt()
}

impl OptimizedImage {
    pub fn new(source: &image::RgbaImage, palette_count: usize, palette_size: usize) -> Self {
        OptimizedImage {
            original: source.clone(),
            tile_palettes: vec![0; 32 * 32],
            palette: vec![image::Rgba([0, 0, 0, 0]); palette_count * palette_size],
            palette_map: vec![0; (source.width() * source.height()) as usize],
            palette_size,
            palette_count,
        }
    }

    pub fn randomize(&mut self) {
        for i in 0..self.palette.len() {
            let r = rand::thread_rng().gen_range(0, 32);
            let g = rand::thread_rng().gen_range(0, 32);
            let b = rand::thread_rng().gen_range(0, 32);
            self.palette[i] = image::Rgba([r * 8 + r / 4, g * 8 + g / 4, b * 8 + b / 4, 255])
        }

        self.optimize();
    }

    pub fn optimize(&mut self) {
        for tile_x in 0..(self.original.width() / 8) {
            for tile_y in 0..(self.original.height() / 8) {
                self.optimize_tile(tile_x as usize * 8, tile_y as usize * 8);
            }
        }
    }

    fn optimize_tile(&mut self, tile_x: usize, tile_y: usize) {
        let mut best_error = f64::MAX;
        let mut best_colors = vec![0; 8 * 8];

        for palette in 0..self.palette_count {
            let mut error = 0.0;
            let mut colors = vec![0; 8 * 8];

            for x in 0..8 {
                for y in 0..8 {
                    let original_pixel = self
                        .original
                        .get_pixel((tile_x + x) as u32, (tile_y + y) as u32);
                    let mut best_delta = f64::MAX;
                    let mut best_color = 0;

                    for color_index in 0..self.palette_size {
                        let color = self.palette[palette * self.palette_size + color_index];
                        let delta = color_distance(original_pixel, &color);

                        if delta < best_delta {
                            best_delta = delta;
                            best_color = color_index;
                        }
                    }

                    colors[y * 8 + x] = best_color;
                    error += best_delta;
                }
            }

            if error < best_error {
                best_error = error;
                best_colors = colors;
            }
        }

        for x in 0..8 {
            for y in 0..8 {
                self.palette_map[(tile_y + y) * self.original.width() as usize + (tile_x + x)] =
                    best_colors[y * 8 + x] as u8;
            }
        }
    }

    pub fn error(&self) -> f64 {
        let rgba = self.as_rgbaimage();

        rgba.enumerate_pixels()
            .map(|(x, y, pixel)| {
                let other = self.original.get_pixel(x, y);

                if other[3] == 0 {
                    0.0
                } else {
                    ((pixel[0] as f64 - other[0] as f64).powf(2.0)
                        + (pixel[1] as f64 - other[1] as f64).powf(2.0)
                        + (pixel[2] as f64 - other[2] as f64).powf(2.0))
                    .sqrt()
                }
            })
            .sum()
    }

    pub fn update_palette(&mut self, p: f64) {
        let index = rand::thread_rng().gen_range(0, self.palette.len());
        let current_error = self.error();
        let current_value = self.palette[index];

        let r = rand::thread_rng().gen_range(0, 32);
        let g = rand::thread_rng().gen_range(0, 32);
        let b = rand::thread_rng().gen_range(0, 32);
        self.palette[index] = image::Rgba([r * 8 + r / 4, g * 8 + g / 4, b * 8 + b / 4, 255]);

        self.optimize();

        if rand::thread_rng().gen_range(0.0, 1.0) > p && self.error() > current_error {
            self.palette[index] = current_value;
            self.optimize();
        }
    }

    pub fn as_rgbaimage(&self) -> image::RgbaImage {
        let mut image = image::RgbaImage::new(self.original.width(), self.original.height());

        for (x, y, pixel) in self.original.enumerate_pixels() {
            let tile_x = x / 8;
            let tile_y = y / 8;
            let palette_index = self.tile_palettes[(tile_y * 32 + tile_x) as usize];
            let color_index = palette_index as usize * self.palette_size
                + self.palette_map[(y * self.original.width() + x) as usize] as usize;

            if pixel[3] == 0 {
                image.put_pixel(x, y, image::Rgba([0, 0, 0, 0]));
            } else {
                image.put_pixel(x, y, self.palette[color_index]);
            }
        }

        image
    }
}

pub fn run(config: config::Config) -> Result<(), Box<dyn Error>> {
    println!("SNES Image Optimizer");
    println!("Source Image: {}", config.source_filename);

    let source_image = image::open(config.source_filename)?.to_rgba();
    let mut target_image = OptimizedImage::new(&source_image, 3, 15);
    target_image.randomize();

    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;

    let window = video_subsystem
        .window("snesimage", 512, 256)
        .position_centered()
        .build()?;

    let mut canvas = window.into_canvas().build()?;

    canvas.set_draw_color(Color::RGB(0, 0, 0));
    canvas.clear();
    canvas.present();

    let mut finished = false;
    let mut event_pump = sdl_context.event_pump()?;
    let mut p = 1.0;
    let mut last_error = f64::MAX;

    while !finished {
        target_image.update_palette(p);
        target_image.optimize();

        let error = target_image.error();

        if (error - last_error).abs() > f64::EPSILON {
            info!("p: {:0.5}  Error: {}", p, target_image.error());
            last_error = error;
        }

        canvas.set_draw_color(Color::RGB(0, 0, 0));
        canvas.clear();
        render_image(&source_image, &mut canvas, 0, 0);
        render_image(&target_image.as_rgbaimage(), &mut canvas, 256, 0);

        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => {
                    finished = true;
                }
                _ => {}
            }
        }

        canvas.present();
        p -= 0.0001;

        if p < 0.0 {
            p = 0.0;
        }
    }

    Ok(())
}

fn render_image(
    image: &image::RgbaImage,
    canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
    base_x: usize,
    base_y: usize,
) {
    for (x, y, pixel) in image.enumerate_pixels() {
        canvas.set_draw_color(Color::RGB(pixel[0], pixel[1], pixel[2]));
        canvas
            .draw_point(Point::new(
                x as i32 + base_x as i32,
                y as i32 + base_y as i32,
            ))
            .unwrap();
    }
}
