pub mod array;
pub mod tensor;
pub mod ops;
pub mod nn;
pub mod train;

// Re-export specific items if needed, but C API usually just relies on the extern "C" functions
// capable of being called regardless of Rust module structure.

#[no_mangle]
pub extern "C" fn numrs_version() -> *const i8 {
    "0.1.0\0".as_ptr() as *const i8
}

/// Print startup log
#[no_mangle]
pub extern "C" fn numrs_print_startup_log() {
    numrs::print_startup_log();
}
pub mod optim;
