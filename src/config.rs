use clap::Clap;

#[derive(Clap)]
#[clap(version = "0.1.0", author = "Jason Lynch <jason@calindora.com>")]
pub struct Config {
    // Filename for the image to optimize.
    pub source_filename: String,
    // Filename for the generated output.
    //pub target_filename: String,
}
