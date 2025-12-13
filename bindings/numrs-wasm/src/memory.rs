//! WASM Memory Management
//!
//! Manejo de memoria optimizado para arrays grandes (>10K elementos)
//! Usa memory.grow() cuando sea necesario

use wasm_bindgen::prelude::*;

/// Información de memoria WASM
#[wasm_bindgen]
pub struct WasmMemoryInfo {
    initial_pages: u32,
    current_pages: u32,
    max_pages: Option<u32>,
}

#[wasm_bindgen]
impl WasmMemoryInfo {
    #[wasm_bindgen(getter)]
    pub fn initial_pages(&self) -> u32 {
        self.initial_pages
    }

    #[wasm_bindgen(getter)]
    pub fn current_pages(&self) -> u32 {
        self.current_pages
    }

    #[wasm_bindgen(getter)]
    pub fn max_pages(&self) -> Option<u32> {
        self.max_pages
    }

    /// Tamaño actual en MB
    #[wasm_bindgen(getter)]
    pub fn size_mb(&self) -> f64 {
        (self.current_pages as f64 * 64.0) / 1024.0
    }
}

// pub fn get_memory_info() -> WasmMemoryInfo { ... }
// pub fn check_memory_for_array(num_elements: usize, bytes_per_element: usize) -> bool { ... }
