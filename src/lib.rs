use std::error::Error;

use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::rect::Point;

extern crate image;

pub mod config;

pub fn run(config: config::Config) -> Result<(), Box<dyn Error>> {
    println!("SNES Image Optimizer");
    println!("Source Image: {}", config.source_filename);

    let source_image = image::open(config.source_filename)?.to_rgba();

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

    while !finished {
        canvas.set_draw_color(Color::RGB(0, 0, 0));
        canvas.clear();
        render_image(&source_image, &mut canvas, 0, 0);

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
