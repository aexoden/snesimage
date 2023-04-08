use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Config {
    // Filename for the image to optimize.
    pub source_filename: String,

    // Filename for the generated JSON output.
    pub target_filename: String,

    // Number of separate subpalettes to use.
    #[arg(short = 'c', long, default_value = "1")]
    pub subpalette_count: usize,

    // Number of colors within each subpalette (not including the transparent color).
    #[arg(short = 's', long, default_value = "7")]
    pub subpalette_size: usize,

    // Whether to dither the output.
    #[arg(short, long)]
    pub dither: bool,

    // Whether to use more expensive CIELAB-based-computations for color comparisons.
    #[arg(long)]
    pub perceptual_palettes: bool,

    // Enables a special mode that uses only colors similar to those available on the NES.
    #[arg(long)]
    pub nes: bool,
}
