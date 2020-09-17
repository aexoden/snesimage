use std::error::Error;
use std::fs::File;
use std::io::prelude::*;

use cogset::{Euclid, Kmeans};
use log::info;
use rand::Rng;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::mouse::MouseButton;
use sdl2::pixels::Color;
use sdl2::rect::Point;
use sdl2::rect::Rect;

extern crate image;
extern crate json;

pub mod config;
pub mod util;

/*
Ideas:
   - Before output, sort the colors in the palette to group like colors.
   - Do k-means clustering in a perceptually uniform color space.
*/

struct OptimizedImage {
    original: image::RgbaImage,
    tile_palettes: Vec<u8>,
    palette: Palette,
    palette_map: Vec<u8>,
    dither: bool,
}

impl OptimizedImage {
    pub fn new(
        source: &image::RgbaImage,
        palette_count: usize,
        palette_size: usize,
        dither: bool,
    ) -> Self {
        OptimizedImage {
            original: source.clone(),
            tile_palettes: vec![0; 32 * 32],
            palette: Palette::new(palette_count, palette_size),
            palette_map: vec![0; (source.width() * source.height()) as usize],
            dither,
        }
    }

    fn width_in_tiles(&self) -> usize {
        (self.original.width() / 8) as usize
    }

    fn height_in_tiles(&self) -> usize {
        (self.original.height() / 8) as usize
    }

    pub fn initialize_tiles(&mut self) {
        if self.palette.sub_count == 1 {
            self.recalculate_palette(0);
            self.optimize();
            return;
        }

        let mut means: Vec<Euclid<[f64; 3]>> = vec![];
        let mut map: Vec<usize> = vec![];

        for tile_x in 0..self.width_in_tiles() {
            for tile_y in 0..self.height_in_tiles() {
                let mut sum = vec![0; 3];
                let index = tile_y * self.width_in_tiles() + tile_x;

                for x in 0..8 {
                    for y in 0..8 {
                        let color = self
                            .original
                            .get_pixel((tile_x * 8 + x) as u32, (tile_y * 8 + y) as u32);

                        if color[3] > 0 {
                            for i in 0..sum.len() {
                                sum[i] += color[i] as usize;
                            }
                        }
                    }
                }

                if sum[0] + sum[1] + sum[2] > 0 {
                    means.push(Euclid([
                        sum[0] as f64 / 64.0,
                        sum[1] as f64 / 64.0,
                        sum[2] as f64 / 64.0,
                    ]));

                    map.push(index);
                }
            }
        }

        let kmeans = Kmeans::new(&means, self.palette.sub_count);
        info!("Finished assigning initial tiles");

        for (index, (mean, tiles)) in kmeans.clusters().iter().enumerate() {
            for tile_index in tiles {
                self.tile_palettes[map[*tile_index]] = index as u8;
            }

            let color = SnesColor::new(
                (mean.0[0] / 8.0).round() as u8,
                (mean.0[1] / 8.0).round() as u8,
                (mean.0[2] / 8.0).round() as u8,
            );

            info!(
                "{} tiles using color: {}, {}, {}",
                tiles.len(),
                color.data[0],
                color.data[1],
                color.data[2]
            );

            for i in 0..self.palette.sub_size {
                self.palette.palette[index * self.palette.sub_size + i] = color.clone();
            }
        }

        self.optimize();
    }

    fn recalculate_palette(&mut self, palette: usize) {
        let mut pixels: Vec<Euclid<[f64; 3]>> = vec![];

        for (tile, tile_palette) in self.tile_palettes.iter().enumerate() {
            if *tile_palette as usize == palette {
                let tile_x = tile % self.width_in_tiles();
                let tile_y = tile / self.width_in_tiles();

                for x in 0..8 {
                    for y in 0..8 {
                        let color = self
                            .original
                            .get_pixel((tile_x * 8 + x) as u32, (tile_y * 8 + y) as u32);

                        if color[3] > 0 {
                            pixels.push(Euclid([
                                color[0] as f64,
                                color[1] as f64,
                                color[2] as f64,
                            ]));
                        }
                    }
                }
            }
        }

        let kmeans = Kmeans::new(&pixels, self.palette.sub_size);

        for (index, (value, _)) in kmeans.clusters().iter().enumerate() {
            let color = SnesColor::new(
                (value.0[0] / 8.0).round() as u8,
                (value.0[1] / 8.0).round() as u8,
                (value.0[2] / 8.0).round() as u8,
            );

            self.palette.palette[palette * self.palette.sub_size + index] = color;
        }
    }

    pub fn recalculate_palettes(&mut self) {
        for palette in 0..self.palette.sub_count {
            self.recalculate_palette(palette);
        }

        self.optimize();
    }

    fn get_palette_index(&self, x: usize, y: usize) -> usize {
        let tile_x = x / 8;
        let tile_y = y / 8;
        let tile_index = tile_x + tile_y * self.width_in_tiles();

        self.tile_palettes[tile_index] as usize
    }

    pub fn optimize(&mut self) {
        let dither_weights = if self.dither {
            [7.0 / 16.0, 3.0 / 16.0, 5.0 / 16.0, 1.0 / 16.0]
        } else {
            [0.0, 0.0, 0.0, 0.0]
        };

        let error_multiplier = 0.8;
        let mut error =
            vec![vec![0.0; 3]; self.original.width() as usize * self.original.height() as usize];

        for y in 0..self.original.height() {
            for x in 0..self.original.width() {
                let pixel_index = (y * self.original.width() + x) as usize;
                let original_color = self.original.get_pixel(x, y);
                let palette = self.get_palette_index(x as usize, y as usize);

                let target_color_values = [
                    original_color[0] as f64 + error[pixel_index][0],
                    original_color[1] as f64 + error[pixel_index][1],
                    original_color[2] as f64 + error[pixel_index][2],
                ];

                let color_index = self
                    .palette
                    .get_closest_color_index(palette, &target_color_values);

                self.palette_map[pixel_index] = if original_color[3] > 0 {
                    color_index as u8
                } else {
                    0
                };

                let new_color =
                    self.palette.palette[palette * self.palette.sub_size + color_index].as_rgba();

                let pixel_error = if original_color[3] > 0 {
                    [
                        target_color_values[0] - new_color[0] as f64,
                        target_color_values[1] - new_color[1] as f64,
                        target_color_values[2] - new_color[2] as f64,
                    ]
                } else {
                    [
                        error[pixel_index][0],
                        error[pixel_index][1],
                        error[pixel_index][2],
                    ]
                };

                for (i, value) in pixel_error.iter().enumerate() {
                    if x + 1 < self.original.width() {
                        error[pixel_index + 1][i] = value * error_multiplier * dither_weights[0];
                    }

                    if y + 1 < self.original.height() {
                        if x > 0 {
                            error[pixel_index + self.original.width() as usize - 1][i] =
                                value * error_multiplier * dither_weights[1];
                        }

                        error[pixel_index + self.original.width() as usize][i] =
                            value * error_multiplier * dither_weights[2];

                        if x + 1 < self.original.width() {
                            error[pixel_index + self.original.width() as usize + 1][i] =
                                value * error_multiplier * dither_weights[3];
                        }
                    }
                }
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

                let index = self.tile_palettes
                    [tile_y as usize * self.width_in_tiles() + tile_x as usize]
                    as usize
                    * self.palette.sub_size
                    + self.palette_map[(y * self.original.width() + x) as usize] as usize;
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

        if p < 0.000001 && rand::thread_rng().gen_range(0.0, 1.0) < 0.001 {
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

    pub fn as_json(&self) -> String {
        let mut palette = vec![];

        for palette_index in 0..self.palette.sub_count {
            for i in 0..16 {
                if i == 0 {
                    palette.push(0);
                } else if i <= self.palette.sub_size {
                    palette.push(
                        self.palette.palette[palette_index * self.palette.sub_size + i - 1]
                            .as_u16(),
                    );
                } else {
                    palette.push(0);
                }
            }
        }

        let mut tiles = vec![];
        let mut tile_palettes = vec![];

        for tile_y in 0..self.height_in_tiles() {
            for tile_x in 0..self.width_in_tiles() {
                let tile_index = tile_y * self.width_in_tiles() + tile_x;
                let mut tile = vec![];

                for y in 0..8 {
                    for x in 0..8 {
                        let index =
                            (tile_y * 8 + y) * self.original.width() as usize + (tile_x * 8 + x);
                        tile.push(self.palette_map[index]);
                    }
                }

                tiles.push(tile);
                tile_palettes.push(self.tile_palettes[tile_index]);
            }
        }

        json::stringify_pretty(
            json::object! {
                palette: palette,
                tiles: tiles,
                tile_palettes: tile_palettes,
            },
            4,
        )
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

    pub fn as_u16(&self) -> u16 {
        self.data[0] as u16 + ((self.data[1] as u16) << 5) + ((self.data[2] as u16) << 10)
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

    pub fn randomize_single(&mut self, index: usize) {
        let r = rand::thread_rng().gen_range(0, 32);
        let g = rand::thread_rng().gen_range(0, 32);
        let b = rand::thread_rng().gen_range(0, 32);
        self.palette[index] = SnesColor::new(r, g, b);
    }

    pub fn get_closest_color_index(&self, palette: usize, target_color: &[f64]) -> usize {
        let mut best_index = 0;
        let mut best_error = f64::MAX;

        let target_color = image::Rgba([
            target_color[0].min(255.0).max(0.0).round() as u8,
            target_color[1].min(255.0).max(0.0).round() as u8,
            target_color[2].min(255.0).max(0.0).round() as u8,
            255,
        ]);

        for index in 0..self.sub_size {
            let color = self.palette[palette * self.sub_size + index].as_rgba();
            let error = color_distance(&color, &target_color);

            if error < best_error {
                best_error = error;
                best_index = index;
            }
        }

        best_index
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

#[derive(PartialEq)]
enum Phase {
    TileAssignment,
    Clustering,
    Optimization,
}

pub fn run(config: config::Config) -> Result<(), Box<dyn Error>> {
    info!("Using source image: {}", config.source_filename);

    let source_image = image::open(config.source_filename)?.to_rgba();
    let mut target_image = OptimizedImage::new(
        &source_image,
        config.subpalette_count,
        config.subpalette_size,
        config.dither,
    );

    target_image.initialize_tiles();

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
    let mut phase = Phase::TileAssignment;
    let mut event_pump = sdl_context.event_pump()?;
    let mut p = 1.0;
    let mut last_error = f64::MAX;

    while !finished {
        if let Phase::Optimization = phase {
            target_image.update_palette(p);
            target_image.optimize();

            let error = target_image.error();

            if (error - last_error).abs() > f64::EPSILON {
                info!("p: {:0.5}  Error: {}", p, target_image.error());
                last_error = error;
            }

            p -= config.p_delta;

            if p < 0.0 {
                p = 0.0;
            }
        }

        canvas.set_draw_color(Color::RGB(0, 0, 0));
        canvas.clear();
        render_image(&source_image, &mut canvas, 0, 0, phase == Phase::TileAssignment);
        render_image(
            &target_image.as_rgbaimage(),
            &mut canvas,
            256,
            0,
            phase == Phase::TileAssignment,
        );
        target_image.palette.render(&mut canvas, 512, 0);

        canvas.set_draw_color(Color::RGB(0, 128, 0));
        canvas.fill_rect(Rect::new(520, 232, 52, 16))?;

        canvas.set_draw_color(Color::RGB(0, 0, 255));
        canvas.fill_rect(Rect::new(580, 232, 52, 16))?;

        for event in event_pump.poll_iter() {
            match event {
                Event::MouseButtonUp {
                    x,
                    y,
                    mouse_btn: MouseButton::Left,
                    ..
                } => {
                    if x >= 520 && y >= 232 && x < (520 + 52) && y < (232 + 16) {
                        match phase {
                            Phase::TileAssignment => {
                                info!("Generating initial palettes");
                                phase = Phase::Clustering;
                                target_image.recalculate_palettes();
                            },
                            Phase::Clustering => {
                                info!("Beginning optimization");
                                phase = Phase::Optimization;
                            },
                            _ => {},
                        }
                    }

                    if x >= 580 && y >= 232 && x < (580 + 52) && y < (232 + 16) {
                        info!("Writing output to {}", config.target_filename);
                        let mut file = File::create(&config.target_filename)?;
                        file.write_all(target_image.as_json().as_bytes())?;
                    }

                    if x < 512 {
                        let tile_x = x % 256 / 8;
                        let tile_y = y / 8;
                        let index =
                            tile_y as usize * target_image.width_in_tiles() + tile_x as usize;

                        target_image.tile_palettes[index] =
                            (target_image.tile_palettes[index] + 1) % config.subpalette_count as u8;

                        if phase != Phase::TileAssignment {
                            target_image.recalculate_palettes();
                        }
                    }
                }
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
    }

    Ok(())
}

fn render_image(
    image: &image::RgbaImage,
    canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
    base_x: usize,
    base_y: usize,
    grid: bool,
) {
    for (x, y, pixel) in image.enumerate_pixels() {
        if !grid || (x % 8 > 0 && y % 8 > 0) {
            canvas.set_draw_color(Color::RGB(pixel[0], pixel[1], pixel[2]));
        } else {
            canvas.set_draw_color(Color::RGB(
                pixel[0] / 5 * 4,
                pixel[1] / 5 * 4,
                pixel[2] / 5 * 4,
            ));
        }

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

    ((((512.0 + red_mean) * r * r) / 256.0) + 4.0 * g * g + (((767.0 - red_mean) * b * b) / 256.0))
        .sqrt()
}
