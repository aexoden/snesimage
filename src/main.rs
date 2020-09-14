use std::process;

use clap::Clap;
use log::error;

use snesimage::config;
use snesimage::util;

fn main() {
    util::setup_logger().unwrap_or_else(|err| {
        eprintln!("FATAL: Could not initialize logger: {}", err);
    });

    let config = config::Config::parse();

    snesimage::run(config).unwrap_or_else(|err| {
        error!("Error running application: {}", err);
        process::exit(1)
    });
}
