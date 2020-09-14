use std::process;

use clap::Clap;

use snesimage::config;

fn main() {
    let config = config::Config::parse();

    snesimage::run(config).unwrap_or_else(|err| {
        eprintln!("Error running application: {}", err);
        process::exit(1)
    });
}
