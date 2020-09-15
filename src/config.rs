use clap::Clap;

#[derive(Clap)]
#[clap(version = "0.1.0", author = "Jason Lynch <jason@calindora.com>")]
pub struct Config {
    // Filename for the image to optimize.
    pub source_filename: String,

    // Filename for the generated output.
    //pub target_filename: String,

    // Number of separate subpalettes to use.
    #[clap(short = "c", long, default_value = "1")]
    pub subpalette_count: usize,

    // Number of colors within each subpalette (not including the transparent color).
    #[clap(short = "s", long, default_value = "7")]
    pub subpalette_size: usize,

    // Rate of decreasing the probability of taking a worse palette.
    #[clap(short, long, default_value = "0.00001")]
    pub p_delta: f64,
}
