extern crate cbindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let output_file = PathBuf::from(&crate_dir)
        .join("include")
        .join("numrs.h");

    // Create include directory if it doesn't exist
    std::fs::create_dir_all(
        PathBuf::from(&crate_dir).join("include")
    ).expect("Failed to create include directory");

    // Only generate if header doesn't exist or FORCE_CBINDGEN is set
    // This avoids macro expansion errors during normal builds
    if !output_file.exists() || env::var("FORCE_CBINDGEN").is_ok() {
        let config = cbindgen::Config::from_file("cbindgen.toml")
            .expect("Unable to find cbindgen.toml");

        println!("cargo:warning=Starting cbindgen generation...");
        match cbindgen::Builder::new()
            .with_crate(&crate_dir)
            .with_config(config)
            .with_language(cbindgen::Language::C)
            .with_cpp_compat(true)
            .generate()
        {
            Ok(bindings) => {
                bindings.write_to_file(&output_file);
                println!("cargo:warning=Generated C header at {:?}", output_file);
            }
            Err(e) => {
                panic!("cbindgen failed: {}", e);
            }
        }
    }
    
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
}
