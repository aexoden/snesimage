use std::error::Error;
use std::fs::File;
use std::io::prelude::*;

use cached::proc_macro::cached;
use cogset::{Euclid, Kmeans};
use log::info;
use palette::{ColorDifference, FromColor, IntoColor, Lab, Srgb};
use rand::distributions::{Distribution, Uniform};
use rgb::FromSlice;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::mouse::MouseButton;
use sdl2::pixels::Color as SDLColor;
use sdl2::rect::Point;
use sdl2::rect::Rect;
use serde_json::json;
use ssimulacra2::{compute_frame_ssimulacra2, ColorPrimaries, Rgb, TransferCharacteristic};

pub mod config;
pub mod util;

/*
Ideas:
   - Before output, sort the colors in the palette to group like colors.
   - Use an actual GUI library so the user doesn't have to deal with unlabeled buttons.
*/

const WIDTH: usize = 256;
const HEIGHT: usize = 256;
const NES_COLOR_COUNT: usize = 56;

struct OptimizedImage {
    width: usize,
    height: usize,
    original: Vec<rgb::RGBA8>,
    tile_palettes: Vec<u8>,
    palette: Palette,
    palette_map: Vec<u8>,
    dither: bool,
    perceptual_palettes: bool,
    nes: bool,
}

impl OptimizedImage {
    pub fn new(
        source: &image::RgbaImage,
        palette_count: usize,
        palette_size: usize,
        dither: bool,
        perceptual_palettes: bool,
        nes: bool,
    ) -> Self {
        OptimizedImage {
            width: source.width() as usize,
            height: source.height() as usize,
            original: source.clone().into_raw().as_rgba().to_vec(),
            tile_palettes: vec![0; 32 * 32],
            palette: Palette::new(palette_count, palette_size),
            palette_map: vec![0; (source.width() * source.height()) as usize],
            dither,
            perceptual_palettes,
            nes,
        }
    }

    fn width_in_tiles(&self) -> usize {
        (self.width / 8) as usize
    }

    fn height_in_tiles(&self) -> usize {
        (self.height / 8) as usize
    }

    pub fn get_original_pixel(&self, x: usize, y: usize) -> rgb::RGBA8 {
        self.original[y * self.width + x]
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
                        let color = self.get_original_pixel(tile_x * 8 + x, tile_y * 8 + y);

                        if color.a > 0 {
                            if self.perceptual_palettes {
                                let lab_color: Lab = Srgb::<u8>::new(color.r, color.g, color.b)
                                    .into_format()
                                    .into_color();
                                sum[0] += lab_color.l;
                                sum[1] += lab_color.a;
                                sum[2] += lab_color.b;
                            } else {
                                sum[0] += color.r as f32;
                                sum[1] += color.g as f32;
                                sum[2] += color.b as f32;
                            }

                            count += 1;
                        }
                    }
                }

                if sum[0] + sum[1] + sum[2] > 0.0 {
                    means.push(Euclid([
                        sum[0] as f64 / count as f64,
                        sum[1] as f64 / count as f64,
                        sum[2] as f64 / count as f64,
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
                let rgb_color: Srgb<u8> =
                    Srgb::from_format(Srgb::from_color(Lab::new(mean.0[0], mean.0[1], mean.0[2])));

                if self.nes {
                    SnesColor::new_nes_only(
                        rgb_color.red / 8,
                        rgb_color.green / 8,
                        rgb_color.blue / 8,
                        self.perceptual_palettes,
                    )
                } else {
                    SnesColor::new(rgb_color.red / 8, rgb_color.green / 8, rgb_color.blue / 8)
                }
            } else {
                if self.nes {
                    SnesColor::new_nes_only(
                        (mean.0[0] / 8.0).round() as u8,
                        (mean.0[1] / 8.0).round() as u8,
                        (mean.0[2] / 8.0).round() as u8,
                        self.perceptual_palettes,
                    )
                } else {
                    SnesColor::new(
                        (mean.0[0] / 8.0).round() as u8,
                        (mean.0[1] / 8.0).round() as u8,
                        (mean.0[2] / 8.0).round() as u8,
                    )
                }
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

    pub fn optimize_palette_entry_random(&mut self, palette: usize, index: usize) {
        let original_color = self.palette.palette[palette * self.palette.sub_size + index].clone();

        let mut best_color = original_color.clone();
        let mut best_error = self.error();

        let mut rng = rand::thread_rng();
        let die = Uniform::from(0..32);

        for _ in 0..64 {
            let red = die.sample(&mut rng);
            let green = die.sample(&mut rng);
            let blue = die.sample(&mut rng);

            self.palette.palette[palette * self.palette.sub_size + index] =
                SnesColor::new(red, green, blue);
            self.optimize();

            let error = self.error();

            if error < best_error {
                best_error = error;
                best_color = SnesColor::new(red, green, blue);
            }
        }

        if original_color != best_color {
            info!(
                "Setting color ({}, {}) from ({}, {}, {}) to ({}, {}, {})",
                palette,
                index,
                original_color.data[0],
                original_color.data[1],
                original_color.data[2],
                best_color.data[0],
                best_color.data[1],
                best_color.data[2]
            );
        }

        self.palette.palette[palette * self.palette.sub_size + index] = best_color;
        self.optimize()
    }

    pub fn optimize_palette_entry_nes(&mut self, palette: usize, index: usize) {
        let original_color = self.palette.palette[palette * self.palette.sub_size + index].clone();

        let mut best_index = 0;
        let mut best_error = f64::MAX;

        for nes_index in 0..NES_COLOR_COUNT {
            self.palette.palette[palette * self.palette.sub_size + index] =
                get_nes_color(nes_index);
            self.optimize();

            let error = self.error();

            if error < best_error {
                best_error = error;
                best_index = nes_index;
            }
        }

        let new_color = get_nes_color(best_index);

        if original_color != new_color {
            info!(
                "Setting color ({}, {}) from ({}, {}, {}) to ({}, {}, {})",
                palette,
                index,
                original_color.data[0],
                original_color.data[1],
                original_color.data[2],
                new_color.data[0],
                new_color.data[1],
                new_color.data[2]
            );
        }

        self.palette.palette[palette * self.palette.sub_size + index] = new_color;
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
            info!(
                "Setting color ({}, {}) from ({}, {}, {}) to ({}, {}, {})",
                palette,
                index,
                original_color.data[0],
                original_color.data[1],
                original_color.data[2],
                new_color.data[0],
                new_color.data[1],
                new_color.data[2]
            );
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
                        let color = self.get_original_pixel(tile_x * 8 + x, tile_y * 8 + y);

                        if color.a > 0 {
                            if self.perceptual_palettes {
                                let lab_color: Lab = Srgb::<u8>::new(color.r, color.g, color.b)
                                    .into_format()
                                    .into_color();

                                pixels.push(Euclid([
                                    lab_color.l.into(),
                                    lab_color.a.into(),
                                    lab_color.b.into(),
                                ]));
                            } else {
                                pixels.push(Euclid([
                                    color.r as f64,
                                    color.g as f64,
                                    color.b as f64,
                                ]));
                            }
                        }
                    }
                }
            }
        }

        let kmeans = Kmeans::new(&pixels, self.palette.sub_size);

        for (index, (value, _)) in kmeans.clusters().iter().enumerate() {
            let rgb_color: Srgb<u8> = Srgb::from_format(Srgb::from_color(Lab::new(
                value.0[0], value.0[1], value.0[2],
            )));

            let color = if self.perceptual_palettes {
                if self.nes {
                    SnesColor::new_nes_only(
                        rgb_color.red / 8,
                        rgb_color.green / 8,
                        rgb_color.blue / 8,
                        self.perceptual_palettes,
                    )
                } else {
                    SnesColor::new(rgb_color.red / 8, rgb_color.green / 8, rgb_color.blue / 8)
                }
            } else {
                if self.nes {
                    SnesColor::new_nes_only(
                        (value.0[0] / 8.0).round() as u8,
                        (value.0[1] / 8.0).round() as u8,
                        (value.0[2] / 8.0).round() as u8,
                        self.perceptual_palettes,
                    )
                } else {
                    SnesColor::new(
                        (value.0[0] / 8.0).round() as u8,
                        (value.0[1] / 8.0).round() as u8,
                        (value.0[2] / 8.0).round() as u8,
                    )
                }
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
        let mut error = vec![vec![0.0; 3]; self.width * self.height];

        for y in 0..self.height {
            for x in 0..self.width {
                let pixel_index = y * self.width + x;
                let original_color = self.get_original_pixel(x, y);
                let palette = self.get_palette_index(x as usize, y as usize);

                let target_color_values = [
                    original_color.r as f64 + error[pixel_index][0],
                    original_color.g as f64 + error[pixel_index][1],
                    original_color.b as f64 + error[pixel_index][2],
                ];

                let color_index = self.palette.get_closest_color_index(
                    palette,
                    &target_color_values,
                    self.perceptual_palettes,
                );

                self.palette_map[pixel_index] = if original_color.a > 0 {
                    color_index as u8
                } else {
                    0
                };

                let new_color =
                    self.palette.palette[palette * self.palette.sub_size + color_index].as_rgba();

                let pixel_error = if original_color.a > 0 {
                    [
                        target_color_values[0] - new_color.r as f64,
                        target_color_values[1] - new_color.g as f64,
                        target_color_values[2] - new_color.b as f64,
                    ]
                } else {
                    [
                        error[pixel_index][0],
                        error[pixel_index][1],
                        error[pixel_index][2],
                    ]
                };

                for (i, value) in pixel_error.iter().enumerate() {
                    if x + 1 < self.width {
                        error[pixel_index + 1][i] += value * error_multiplier * dither_weights[0];
                    }

                    if y + 1 < self.height {
                        if x > 0 {
                            error[pixel_index + self.width - 1][i] +=
                                value * error_multiplier * dither_weights[1];
                        }

                        error[pixel_index + self.width][i] +=
                            value * error_multiplier * dither_weights[2];

                        if x + 1 < self.width {
                            error[pixel_index + self.width + 1][i] +=
                                value * error_multiplier * dither_weights[3];
                        }
                    }
                }
            }
        }
    }

    pub fn error(&self) -> f64 {
        let rgba = self.as_rgba();

        let src_data = self
            .original
            .iter()
            .map(|value| {
                [
                    value.r as f32 / 255.0,
                    value.g as f32 / 255.0,
                    value.b as f32 / 255.0,
                ]
            })
            .collect::<Vec<_>>();

        let src = Rgb::new(
            src_data,
            self.width,
            self.height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap();

        let dst_data = rgba
            .iter()
            .map(|value| {
                [
                    value.r as f32 / 255.0,
                    value.g as f32 / 255.0,
                    value.b as f32 / 255.0,
                ]
            })
            .collect::<Vec<_>>();

        let dst = Rgb::new(
            dst_data,
            self.width,
            self.height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap();

        100.0 - compute_frame_ssimulacra2(src, dst).unwrap()
    }

    pub fn as_rgba(&self) -> Vec<rgb::RGBA8> {
        let mut image = vec![
            rgb::RGBA8 {
                r: 0,
                g: 0,
                b: 0,
                a: 0
            };
            self.width * self.height
        ];

        for y in 0..self.height {
            for x in 0..self.width {
                let tile_x = x / 8;
                let tile_y = y / 8;
                let palette_index = self.tile_palettes[tile_y * 32 + tile_x];
                let color_index = palette_index as usize * self.palette.sub_size
                    + self.palette_map[y * self.width + x] as usize;
                let pixel = self.get_original_pixel(x, y);

                if pixel.a > 0 {
                    image[y * self.width + x] = self.palette.palette[color_index].as_rgba();
                }
            }
        }

        image
    }

    pub fn as_json(&self) -> serde_json::Value {
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
                        let index = (tile_y * 8 + y) * self.width + (tile_x * 8 + x);
                        if self.get_original_pixel(tile_x * 8 + x, tile_y * 8 + y).a == 0 {
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

        json!({
            "palette": palette,
            "tiles": tiles,
            "tile_palettes": tile_palettes,
        })
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

    pub fn new_nes_only(r: u8, g: u8, b: u8, cielab: bool) -> Self {
        let color = SnesColor::new(r, g, b).as_rgba();

        let mut best_color = get_nes_color(0);
        let mut best_error = f64::MAX;

        for index in 0..NES_COLOR_COUNT {
            let error = if cielab {
                color_distance_cielab(color, get_nes_color(index).as_rgba())
            } else {
                color_distance_red_mean(color, get_nes_color(index).as_rgba())
            };

            if error < best_error {
                best_color = get_nes_color(index);
                best_error = error;
            }
        }

        best_color
    }

    pub fn as_rgba(&self) -> rgb::RGBA8 {
        rgb::RGBA8 {
            r: self.data[0] * 8 + self.data[0] / 4,
            g: self.data[1] * 8 + self.data[1] / 4,
            b: self.data[2] * 8 + self.data[2] / 4,
            a: 255,
        }
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

fn get_nes_color(index: usize) -> SnesColor {
    match index {
        0 => SnesColor::new(13, 13, 13),
        1 => SnesColor::new(0, 2, 16),
        2 => SnesColor::new(3, 0, 17),
        3 => SnesColor::new(7, 0, 15),
        4 => SnesColor::new(10, 0, 10),
        5 => SnesColor::new(11, 0, 3),
        6 => SnesColor::new(9, 2, 0),
        7 => SnesColor::new(7, 3, 0),
        8 => SnesColor::new(4, 6, 0),
        9 => SnesColor::new(0, 7, 0),
        10 => SnesColor::new(0, 8, 0),
        11 => SnesColor::new(0, 7, 4),
        12 => SnesColor::new(0, 5, 10),
        13 => SnesColor::new(0, 0, 0),
        14 => SnesColor::new(23, 23, 23),
        15 => SnesColor::new(3, 10, 24),
        16 => SnesColor::new(9, 6, 28),
        17 => SnesColor::new(14, 4, 26),
        18 => SnesColor::new(18, 3, 21),
        19 => SnesColor::new(19, 5, 11),
        20 => SnesColor::new(19, 6, 0),
        21 => SnesColor::new(15, 9, 0),
        22 => SnesColor::new(11, 12, 0),
        23 => SnesColor::new(4, 14, 0),
        24 => SnesColor::new(0, 15, 0),
        25 => SnesColor::new(0, 14, 8),
        26 => SnesColor::new(0, 13, 17),
        27 => SnesColor::new(0, 0, 0),
        28 => SnesColor::new(31, 31, 31),
        29 => SnesColor::new(13, 20, 31),
        30 => SnesColor::new(17, 19, 31),
        31 => SnesColor::new(22, 16, 31),
        32 => SnesColor::new(27, 14, 31),
        33 => SnesColor::new(28, 14, 23),
        34 => SnesColor::new(28, 17, 13),
        35 => SnesColor::new(26, 19, 5),
        36 => SnesColor::new(22, 21, 1),
        37 => SnesColor::new(15, 24, 2),
        38 => SnesColor::new(10, 25, 8),
        39 => SnesColor::new(8, 25, 16),
        40 => SnesColor::new(8, 24, 24),
        41 => SnesColor::new(9, 9, 9),
        42 => SnesColor::new(31, 31, 31),
        43 => SnesColor::new(25, 29, 31),
        44 => SnesColor::new(27, 27, 31),
        45 => SnesColor::new(29, 27, 31),
        46 => SnesColor::new(31, 26, 31),
        47 => SnesColor::new(31, 26, 30),
        48 => SnesColor::new(31, 27, 25),
        49 => SnesColor::new(31, 28, 22),
        50 => SnesColor::new(30, 30, 21),
        51 => SnesColor::new(27, 31, 21),
        52 => SnesColor::new(25, 31, 23),
        53 => SnesColor::new(24, 31, 26),
        54 => SnesColor::new(24, 30, 30),
        55 => SnesColor::new(23, 24, 23),
        _ => SnesColor::new(0, 0, 0),
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

        let target_color = rgb::RGBA8 {
            r: target_color[0].min(255.0).max(0.0).round() as u8,
            g: target_color[1].min(255.0).max(0.0).round() as u8,
            b: target_color[2].min(255.0).max(0.0).round() as u8,
            a: 255,
        };

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

    let source_image = image::open(config.source_filename)?.into_rgba8();

    if source_image.width() as usize != WIDTH && source_image.height() as usize != HEIGHT {
        return Err("Image size must be 256x256".into());
    }

    let mut target_image = OptimizedImage::new(
        &source_image,
        config.subpalette_count,
        config.subpalette_size,
        config.dither,
        config.perceptual_palettes,
        config.nes,
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
    let mut step = 0;

    let mut palette = 0;
    let mut palette_index = 0;
    let mut channel = 0;

    while !finished {
        if let Phase::Optimization = phase {
            let random = step % 5 < 4;

            if config.nes {
                target_image.optimize_palette_entry_nes(palette, palette_index);
            } else if random {
                target_image.optimize_palette_entry_random(palette, palette_index);
            } else {
                target_image.optimize_palette_entry_channel(palette, palette_index, channel);
            }

            target_image.optimize();

            let error = target_image.error();

            if (error - last_error).abs() > f64::EPSILON {
                info!("Current Error: {}", error);
                last_error = error;
            }

            channel += 1;

            if channel == 3 || random {
                channel = 0;
                palette_index += 1;

                if palette_index == config.subpalette_size {
                    palette_index = 0;
                    palette += 1;

                    if palette == config.subpalette_count {
                        palette = 0;
                        step += 1;
                    }
                }
            }
        }

        canvas.set_draw_color(SDLColor::RGB(0, 0, 0));
        canvas.clear();
        render_image(
            &source_image.clone().into_raw().as_rgba().to_vec(),
            source_image.width() as usize,
            source_image.height() as usize,
            &mut canvas,
            0,
            0,
            phase == Phase::TileAssignment,
        );
        render_image(
            &target_image.as_rgba(),
            target_image.width,
            target_image.height,
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
                        file.write_all(target_image.as_json().to_string().as_bytes())?;
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
    image: &Vec<rgb::RGBA8>,
    width: usize,
    height: usize,
    canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
    base_x: usize,
    base_y: usize,
    grid: bool,
) {
    for y in 0..height {
        for x in 0..width {
            let pixel = image[y * width + x];

            if !grid || (x % 8 > 0 && y % 8 > 0) {
                canvas.set_draw_color(SDLColor::RGB(pixel.r, pixel.g, pixel.b));
            } else {
                canvas.set_draw_color(SDLColor::RGB(
                    pixel.r / 5 * 4,
                    pixel.g / 5 * 4,
                    pixel.b / 5 * 4,
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
}

fn color_distance_red_mean(color1: rgb::RGBA8, color2: rgb::RGBA8) -> f64 {
    let red_mean = (color1.r as f64 + color2.r as f64) / 2.0;
    let r = color1.r as f64 - color2.r as f64;
    let g = color1.g as f64 - color2.g as f64;
    let b = color1.b as f64 - color2.b as f64;

    ((((512.0 + red_mean) * r * r) / 256.0) + 4.0 * g * g + (((767.0 - red_mean) * b * b) / 256.0))
        .sqrt()
}

#[cached]
fn color_distance_cielab(color1: rgb::RGBA8, color2: rgb::RGBA8) -> f64 {
    let color1: Lab = Srgb::new(color1.r, color1.g, color1.b)
        .into_format()
        .into_color();
    let color2: Lab = Srgb::new(color2.r, color2.g, color2.b)
        .into_format()
        .into_color();

    color1.get_color_difference(&color2).into()
}
