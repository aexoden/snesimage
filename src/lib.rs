use std::error::Error;
use std::fs::File;
use std::io::prelude::*;

use cached::proc_macro::cached;
use cogset::{Euclid, Kmeans};
use log::info;
use scarlet::color::{Color, RGBColor};
use scarlet::colors::cielabcolor::CIELABColor;
use scarlet::coord::Coord;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::mouse::MouseButton;
use sdl2::pixels::Color as SDLColor;
use sdl2::rect::Point;
use sdl2::rect::Rect;

extern crate image;
extern crate json;

pub mod config;
pub mod util;

/*
Ideas:
   - Before output, sort the colors in the palette to group like colors.
   - Use an actual GUI library so the user doesn't have to deal with unlabeled buttons.
*/

struct OptimizedImage {
    original: image::RgbaImage,
    tile_palettes: Vec<u8>,
    palette: Palette,
    palette_map: Vec<u8>,
    dither: bool,
    perceptual_palettes: bool,
    perceptual_optimization: bool,
}

impl OptimizedImage {
    pub fn new(
        source: &image::RgbaImage,
        palette_count: usize,
        palette_size: usize,
        dither: bool,
        perceptual_palettes: bool,
        perceptual_optimization: bool,
    ) -> Self {
        OptimizedImage {
            original: source.clone(),
            tile_palettes: vec![0; 32 * 32],
            palette: Palette::new(palette_count, palette_size),
            palette_map: vec![0; (source.width() * source.height()) as usize],
            dither,
            perceptual_palettes,
            perceptual_optimization,
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
                let mut sum = vec![0.0; 3];
                let mut count = 0;
                let index = tile_y * self.width_in_tiles() + tile_x;

                for x in 0..8 {
                    for y in 0..8 {
                        let color = self
                            .original
                            .get_pixel((tile_x * 8 + x) as u32, (tile_y * 8 + y) as u32);

                        if color[3] > 0 {
                            if self.perceptual_palettes {
                                let lab_color: CIELABColor =
                                    RGBColor::from((color[0], color[1], color[2])).convert();
                                sum[0] += lab_color.l;
                                sum[1] += lab_color.a;
                                sum[2] += lab_color.b;
                            } else {
                                for i in 0..sum.len() {
                                    sum[i] += color[i] as f64;
                                }
                            }

                            count += 1;
                        }
                    }
                }

                if sum[0] + sum[1] + sum[2] > 0.0 {
                    means.push(Euclid([
                        sum[0] / count as f64,
                        sum[1] / count as f64,
                        sum[2] / count as f64,
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

            let color = if self.perceptual_palettes {
                let rgb_color: RGBColor = CIELABColor::from(Coord {
                    x: mean.0[0],
                    y: mean.0[1],
                    z: mean.0[2],
                })
                .convert();

                SnesColor::new(
                    rgb_color.int_r() / 8,
                    rgb_color.int_g() / 8,
                    rgb_color.int_b() / 8,
                )
            } else {
                SnesColor::new(
                    (mean.0[0] / 8.0).round() as u8,
                    (mean.0[1] / 8.0).round() as u8,
                    (mean.0[2] / 8.0).round() as u8,
                )
            };

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

    pub fn optimize_palette_entry_channel(&mut self, palette: usize, index: usize, channel: usize) {
        let original_color = self.palette.palette[palette * self.palette.sub_size + index].clone();
        let mut best_value = original_color.data[channel];
        let mut best_error = self.error();

        for value in 0..32 {
            self.palette.palette[palette * self.palette.sub_size + index].data[channel] = value;
            self.optimize();

            let error = self.error();

            if error < best_error {
                best_error = error;
                best_value = value;
            }
        }

        if original_color.data[channel] != best_value {
            let mut new_color = original_color.clone();
            new_color.data[channel] = best_value;
            info!("Setting color ({}, {}) from ({}, {}, {}) to ({}, {}, {})", palette, index, original_color.data[0], original_color.data[1], original_color.data[2], new_color.data[0], new_color.data[1], new_color.data[2]);
        }

        self.palette.palette[palette * self.palette.sub_size + index].data[channel] = best_value;
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
                            if self.perceptual_palettes {
                                let lab_color: CIELABColor =
                                    RGBColor::from((color[0], color[1], color[2])).convert();

                                pixels.push(Euclid([lab_color.l, lab_color.a, lab_color.b]));
                            } else {
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
        }

        let kmeans = Kmeans::new(&pixels, self.palette.sub_size);

        for (index, (value, _)) in kmeans.clusters().iter().enumerate() {
            let rgb_color: RGBColor = CIELABColor::from(Coord {
                x: value.0[0],
                y: value.0[1],
                z: value.0[2],
            })
            .convert();

            let color = if self.perceptual_palettes {
                SnesColor::new(
                    rgb_color.int_r() / 8,
                    rgb_color.int_g() / 8,
                    rgb_color.int_b() / 8,
                )
            } else {
                SnesColor::new(
                    (value.0[0] / 8.0).round() as u8,
                    (value.0[1] / 8.0).round() as u8,
                    (value.0[2] / 8.0).round() as u8,
                )
            };

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

                let color_index = self.palette.get_closest_color_index(
                    palette,
                    &target_color_values,
                    self.perceptual_optimization,
                );

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
                        error[pixel_index + 1][i] += value * error_multiplier * dither_weights[0];
                    }

                    if y + 1 < self.original.height() {
                        if x > 0 {
                            error[pixel_index + self.original.width() as usize - 1][i] +=
                                value * error_multiplier * dither_weights[1];
                        }

                        error[pixel_index + self.original.width() as usize][i] +=
                            value * error_multiplier * dither_weights[2];

                        if x + 1 < self.original.width() {
                            error[pixel_index + self.original.width() as usize + 1][i] +=
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
                } else if self.perceptual_optimization {
                    color_distance_cielab(*pixel, *other)
                } else {
                    color_distance_red_mean(*pixel, *other)
                }
            })
            .sum()
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
                        if self
                            .original
                            .get_pixel((tile_x * 8 + x) as u32, (tile_y * 8 + y) as u32)[3]
                            == 0
                        {
                            tile.push(0);
                        } else {
                            tile.push(self.palette_map[index] + 1);
                        }
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

#[derive(Clone, PartialEq)]
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

    pub fn as_sdl_rgb(&self) -> SDLColor {
        SDLColor::RGB(
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

    pub fn get_closest_color_index(
        &self,
        palette: usize,
        target_color: &[f64],
        cielab: bool,
    ) -> usize {
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
            let error = if cielab {
                color_distance_cielab(color, target_color)
            } else {
                color_distance_red_mean(color, target_color)
            };

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
        config.perceptual_palettes,
        config.perceptual_optimization,
    );

    target_image.initialize_tiles();

    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;

    let window = video_subsystem
        .window("snesimage", 640, 256)
        .position_centered()
        .build()?;

    let mut canvas = window.into_canvas().build()?;

    canvas.set_draw_color(SDLColor::RGB(0, 0, 0));
    canvas.clear();
    canvas.present();

    let mut finished = false;
    let mut phase = Phase::TileAssignment;
    let mut event_pump = sdl_context.event_pump()?;
    let mut last_error = f64::MAX;

    let mut palette = 0;
    let mut palette_index = 0;
    let mut channel = 0;

    while !finished {
        if let Phase::Optimization = phase {
            target_image.optimize_palette_entry_channel(palette, palette_index, channel);
            target_image.optimize();

            let error = target_image.error();

            if (error - last_error).abs() > f64::EPSILON {
                info!("Current Error: {}", target_image.error());
                last_error = error;
            }

            channel += 1;

            if channel == 3 {
                channel = 0;
                palette_index += 1;

                if palette_index == config.subpalette_size {
                    palette_index = 0;
                    palette += 1;

                    if palette == config.subpalette_count {
                        palette = 0;
                    }
                }
            }
        }

        canvas.set_draw_color(SDLColor::RGB(0, 0, 0));
        canvas.clear();
        render_image(
            &source_image,
            &mut canvas,
            0,
            0,
            phase == Phase::TileAssignment,
        );
        render_image(
            &target_image.as_rgbaimage(),
            &mut canvas,
            256,
            0,
            phase == Phase::TileAssignment,
        );
        target_image.palette.render(&mut canvas, 512, 0);

        canvas.set_draw_color(SDLColor::RGB(0, 128, 0));
        canvas.fill_rect(Rect::new(520, 232, 52, 16))?;

        canvas.set_draw_color(SDLColor::RGB(0, 0, 255));
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
                            }
                            Phase::Clustering => {
                                info!("Beginning optimization");
                                phase = Phase::Optimization;
                            }
                            _ => {}
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
            canvas.set_draw_color(SDLColor::RGB(pixel[0], pixel[1], pixel[2]));
        } else {
            canvas.set_draw_color(SDLColor::RGB(
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

fn color_distance_red_mean(color1: image::Rgba<u8>, color2: image::Rgba<u8>) -> f64 {
    let red_mean = (color1[0] as f64 + color2[0] as f64) / 2.0;
    let r = color1[0] as f64 - color2[0] as f64;
    let g = color1[1] as f64 - color2[1] as f64;
    let b = color1[2] as f64 - color2[2] as f64;

    ((((512.0 + red_mean) * r * r) / 256.0) + 4.0 * g * g + (((767.0 - red_mean) * b * b) / 256.0))
        .sqrt()
}

#[cached]
fn color_distance_cielab(color1: image::Rgba<u8>, color2: image::Rgba<u8>) -> f64 {
    let color1 = RGBColor::from((color1[0], color1[1], color1[2]));
    let color2 = RGBColor::from((color2[0], color2[1], color2[2]));

    color1.distance(&color2)
}
