use std::collections::HashMap;
use std::error::Error;

use rand::Rng;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::rect::Point;

extern crate image;

pub mod config;

/*
 Ideas:
    - Initially quantize the source image to the SNES color space, since that can't be exceeded anyway.
    - Separate the image into N groups of tiles, and then optimize those sets of tiles.
    - Figure out how to combine the above with dithering.
    - If continuing with stochastic search, at least ensure all entries in palette map are touched routinely. Focus
      on actual optimization rather than a completely random process.
    - Potentially limit the degree of change, so the palette doesn't end up all over the place anytime the p value is still > 0.
 */

struct OptimizedImage {
    original: image::RgbaImage,
    tile_palettes: Vec<u8>,
    palette: Vec<image::Rgba<u8>>,
    palette_map: HashMap<image::Rgba<u8>, Vec<u8>>,
    palette_size: usize,
    palette_count: usize,
}

fn color_distance(color1: &image::Rgba<u8>, color2: &image::Rgba<u8>) -> f64 {
    ((color1[0] as f64 - color2[0] as f64).powf(2.0) + (color1[1] as f64 - color2[1] as f64).powf(2.0) + (color1[2] as f64 - color2[2] as f64).powf(2.0)).sqrt()
}

impl OptimizedImage {
    pub fn new(source: &image::RgbaImage, palette_count: usize, palette_size: usize) -> Self {
        let palette_map = source.enumerate_pixels().map(|(_, _, pixel)| {
            (image::Rgba([pixel[0], pixel[1], pixel[2], pixel[3]]), vec![0; palette_count])
        }).collect();

        OptimizedImage {
            original: source.clone(),
            tile_palettes: vec![0; 32 * 32],
            palette: vec![image::Rgba([0, 0, 0, 0]); palette_count * palette_size],
            palette_map,
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

        for i in 0..self.tile_palettes.len() {
            self.tile_palettes[i] = rand::thread_rng().gen_range(0, self.palette_count) as u8;
        }

        self.palette_map = self.palette_map.iter().map(|(key, _)| {
            let palettes = (0..self.palette_count).map(|_| {
                rand::thread_rng().gen_range(0, self.palette_size) as u8
            }).collect();

            (*key, palettes)
        }).collect();
    }

    pub fn error(&self) -> f64 {
        let mut error = 0.0;
        let rgba = self.as_rgbaimage();

        rgba.enumerate_pixels().map(|(x, y, pixel)| {
            let other = self.original.get_pixel(x, y);

            if other[3] == 0 {
                0.0
            } else {
                ((pixel[0] as f64 - other[0] as f64).powf(2.0) + (pixel[1] as f64 - other[1] as f64).powf(2.0) + (pixel[2] as f64 - other[2] as f64).powf(2.0)).sqrt()
            }
        }).sum()
    }

    pub fn update_palette(&mut self, p: f64) {
        let index = rand::thread_rng().gen_range(0, self.palette.len());
        let current_error = self.error();
        let current_value = self.palette[index];

        let r = rand::thread_rng().gen_range(0, 32);
        let g = rand::thread_rng().gen_range(0, 32);
        let b = rand::thread_rng().gen_range(0, 32);
        self.palette[index] = image::Rgba([r * 8 + r / 4, g * 8 + g / 4, b * 8 + b / 4, 255]);

        if rand::thread_rng().gen_range(0.0, 1.0) > p && self.error() > current_error {
            self.palette[index] = current_value;
        }
    }

    pub fn update_tile_palettes(&mut self, p: f64) {
        let index = rand::thread_rng().gen_range(0, self.tile_palettes.len());
        let current_error = self.error();
        let current_value = self.tile_palettes[index];

        self.tile_palettes[index] = rand::thread_rng().gen_range(0, self.palette_count) as u8;

        if rand::thread_rng().gen_range(0.0, 1.0) > p && self.error() > current_error {
            self.tile_palettes[index] = current_value;
        }
    }

    pub fn update_palette_map(&mut self, p: f64) {
        let x = rand::thread_rng().gen_range(0, self.original.width());
        let y = rand::thread_rng().gen_range(0, self.original.height());
        let index = rand::thread_rng().gen_range(0, self.palette_count);
        let key = self.original.get_pixel(x, y);
        let current_value = self.palette_map[key][index];

        let current_error = color_distance(key, &self.palette[index as usize * self.palette_size + current_value as usize]);

        let new_value = rand::thread_rng().gen_range(0, self.palette_size) as u8;
        let new_error = color_distance(key, &self.palette[index as usize * self.palette_size + new_value as usize]);

        if rand::thread_rng().gen_range(0.0, 1.0) < p || new_error < current_error {
            self.palette_map.get_mut(key).unwrap()[index] = new_value;
        }
    }

    pub fn as_rgbaimage(&self) -> image::RgbaImage {
        let mut image = image::RgbaImage::new(self.original.width(), self.original.height());

        for (x, y, pixel) in self.original.enumerate_pixels() {
            let tile_x = x / 8;
            let tile_y = y / 8;
            let palette_index = self.tile_palettes[(tile_y * 32 + tile_x) as usize];
            let color_index = palette_index as usize * self.palette_size + self.palette_map[&image::Rgba([pixel[0], pixel[1], pixel[2], pixel[3]])][palette_index as usize] as usize;

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
    let mut cycle = 0;
    let mut p = 1.0;

    while !finished {
        if cycle % 256 == 0 {
            target_image.update_palette(p);
        }
        if cycle % 16 == 0 {
            target_image.update_tile_palettes(p);
        }
        target_image.update_palette_map(p);

        println!("p: {:0.5}  Error: {}", p, target_image.error());

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
        cycle += 1;
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
