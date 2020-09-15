use std::error::Error;

use log::info;
use rand::Rng;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::rect::Point;
use sdl2::rect::Rect;

extern crate image;

pub mod config;
pub mod util;

/*
Ideas:
   - Separate the image into N groups of tiles, and then optimize those sets of tiles.
   - Figure out how to combine the above with dithering.
   - Before output, sort the colors in the palette to group like colors.
*/

struct OptimizedImage {
    original: image::RgbaImage,
    tile_palettes: Vec<u8>,
    palette: Palette,
    palette_map: Vec<u8>,
}

impl OptimizedImage {
    pub fn new(source: &image::RgbaImage, palette_count: usize, palette_size: usize) -> Self {
        OptimizedImage {
            original: source.clone(),
            tile_palettes: vec![0; 32 * 32],
            palette: Palette::new(palette_count, palette_size),
            palette_map: vec![0; (source.width() * source.height()) as usize],
        }
    }

    pub fn randomize(&mut self) {
        self.palette.randomize();
        self.optimize();
    }

    pub fn optimize(&mut self) {
        for tile_x in 0..(self.original.width() / 8) {
            for tile_y in 0..(self.original.height() / 8) {
                self.optimize_tile(tile_x as usize, tile_y as usize);
            }
        }
    }

    fn optimize_tile(&mut self, tile_x: usize, tile_y: usize) {
        let mut best_error = f64::MAX;
        let mut best_palette = 0;
        let mut best_colors = vec![0; 8 * 8];

        for palette in 0..self.palette.sub_count {
            let mut error = 0.0;
            let mut colors = vec![0; 8 * 8];

            for x in 0..8 {
                for y in 0..8 {
                    let original_pixel = self
                        .original
                        .get_pixel((tile_x * 8 + x) as u32, (tile_y * 8 + y) as u32);

                    if original_pixel[3] > 0 {
                        let mut best_delta = f64::MAX;
                        let mut best_color = 0;

                        for color_index in 0..self.palette.sub_size {
                            let color =
                                &self.palette.palette[palette * self.palette.sub_size + color_index];
                            let delta = color_distance(original_pixel, &color.as_rgba());

                            if delta < best_delta {
                                best_delta = delta;
                                best_color = color_index;
                            }
                        }

                        colors[y * 8 + x] = best_color;
                        error += best_delta;
                    }
                }
            }

            if error < best_error {
                best_error = error;
                best_colors = colors;
                best_palette = palette;
            }
        }

        self.tile_palettes[tile_y * (self.original.width() / 8) as usize + tile_x] =
            best_palette as u8;

        for x in 0..8 {
            for y in 0..8 {
                self.palette_map
                    [(tile_y * 8 + y) * self.original.width() as usize + (tile_x * 8 + x)] =
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
                    color_distance(pixel, other)
                }
            })
            .sum()
    }

    pub fn randomize_unused_colors(&mut self) {
        let mut counts = vec![0; self.palette.palette.len()];

        for x in 0..self.original.width() {
            for y in 0..self.original.height() {
                let tile_x = x / 8;
                let tile_y = y / 8;

                let index = self.tile_palettes[(tile_y * (self.original.width() / 8) + tile_x) as usize] + self.palette_map[(y * self.original.width() + x) as usize];
                counts[index as usize] += 1;
            }
        }

        for (index, count) in counts.iter().enumerate() {
            if *count == 0 {
                self.palette.randomize_single(index);
            }
        }
    }

    pub fn update_palette(&mut self, p: f64) {
        let index = rand::thread_rng().gen_range(0, self.palette.palette.len());
        let current_error = self.error();
        let current_value = self.palette.palette[index].clone();

        let channel = rand::thread_rng().gen_range(0, 3);

        let delta: i8 = if current_value.data[channel] == 0 {
            1
        } else if current_value.data[channel] == 31 {
            -1
        } else {
            match rand::thread_rng().gen_range(0, 2) {
                0 => -1,
                1 => 1,
                _ => unreachable!(),
            }
        };

        let mut new_value = current_value.clone();
        new_value.data[channel] = (new_value.data[channel] as i8 + delta) as u8;

        self.palette.palette[index] = new_value;

        self.optimize();

        if rand::thread_rng().gen_range(0.0, 1.0) > p && self.error() > current_error {
            self.palette.palette[index] = current_value;
            self.optimize();
        }

        if rand::thread_rng().gen_range(0.0, 1.0) < 0.01 {
            self.randomize_unused_colors();
            self.optimize();
        }
    }

    pub fn as_rgbaimage(&self) -> image::RgbaImage {
        let mut image = image::RgbaImage::new(self.original.width(), self.original.height());

        for (x, y, pixel) in self.original.enumerate_pixels() {
            let tile_x = x / 8;
            let tile_y = y / 8;
            let palette_index = self.tile_palettes[(tile_y * 32 + tile_x) as usize];
            let color_index = palette_index as usize * self.palette.sub_size
                + self.palette_map[(y * self.original.width() + x) as usize] as usize;

            if pixel[3] == 0 {
                image.put_pixel(x, y, image::Rgba([0, 0, 0, 0]));
            } else {
                image.put_pixel(x, y, self.palette.palette[color_index].as_rgba());
            }
        }

        image
    }
}

#[derive(Clone)]
struct SnesColor {
    data: Vec<u8>,
}

impl SnesColor {
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        SnesColor {
            data: vec![r, g, b],
        }
    }

    pub fn as_rgba(&self) -> image::Rgba<u8> {
        image::Rgba([
            self.data[0] * 8 + self.data[0] / 4,
            self.data[1] * 8 + self.data[1] / 4,
            self.data[2] * 8 + self.data[2] / 4,
            255,
        ])
    }

    pub fn as_sdl_rgb(&self) -> Color {
        Color::RGB(
            self.data[0] * 8 + self.data[0] / 4,
            self.data[1] * 8 + self.data[1] / 4,
            self.data[2] * 8 + self.data[2] / 4,
        )
    }
}

struct Palette {
    palette: Vec<SnesColor>,
    sub_size: usize,
    sub_count: usize,
}

impl Palette {
    pub fn new(sub_count: usize, sub_size: usize) -> Self {
        Palette {
            palette: vec![SnesColor::new(0, 0, 0); sub_count * sub_size],
            sub_count,
            sub_size,
        }
    }

    pub fn randomize(&mut self) {
        for i in 0..self.palette.len() {
            self.randomize_single(i);
        }
    }

    pub fn randomize_single(&mut self, index: usize) {
        let r = rand::thread_rng().gen_range(0, 32);
        let g = rand::thread_rng().gen_range(0, 32);
        let b = rand::thread_rng().gen_range(0, 32);
        self.palette[index] = SnesColor::new(r, g, b);
    }

    pub fn render(
        &self,
        canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
        base_x: usize,
        base_y: usize,
    ) {
        for palette_index in 0..self.sub_count {
            for color_index in 0..self.sub_size {
                let x = base_x + (color_index + 1) * 8;
                let y = base_y + palette_index * 8;
                let color = &self.palette[palette_index * self.sub_size + color_index];
                canvas.set_draw_color(color.as_sdl_rgb());
                canvas
                    .fill_rect(Rect::new(x as i32, y as i32, 8, 8))
                    .unwrap();
            }
        }
    }
}

pub fn run(config: config::Config) -> Result<(), Box<dyn Error>> {
    println!("SNES Image Optimizer");
    println!("Source Image: {}", config.source_filename);

    let source_image = image::open(config.source_filename)?.to_rgba();
    let mut target_image = OptimizedImage::new(
        &source_image,
        config.subpalette_count,
        config.subpalette_size,
    );
    target_image.randomize();

    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;

    let window = video_subsystem
        .window("snesimage", 640, 256)
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
        target_image.palette.render(&mut canvas, 512, 0);

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
        p -= config.p_delta;

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

fn color_distance(color1: &image::Rgba<u8>, color2: &image::Rgba<u8>) -> f64 {
    let red_mean = (color1[0] as f64 + color2[0] as f64) / 2.0;
    let r = color1[0] as f64 - color2[0] as f64;
    let g = color1[1] as f64 - color2[1] as f64;
    let b = color1[2] as f64 - color2[2] as f64;

    ((((512.0 + red_mean) * r * r) / 256.0) + 4.0 * g * g + (((767.0 - red_mean) * b * b) / 256.0)).sqrt()
}
