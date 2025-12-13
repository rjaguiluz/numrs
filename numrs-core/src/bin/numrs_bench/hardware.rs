//! Hardware detection utilities

use sysinfo::System;
use crate::types::HardwareInfo;

pub fn detect_hardware() -> HardwareInfo {
    let mut sys = System::new_all();
    sys.refresh_all();
    
    // CPU detection
    let cpu = if let Some(cpu) = sys.cpus().first() {
        cpu.brand().trim().to_string()
    } else {
        "Unknown CPU".to_string()
    };
    
    // RAM detection (convert bytes to GB)
    let ram_gb = sys.total_memory() / 1024 / 1024 / 1024;
    
    // OS detection
    let os = format!(
        "{} {} ({})",
        System::name().unwrap_or_else(|| "Unknown OS".to_string()),
        System::os_version().unwrap_or_else(|| "".to_string()),
        System::kernel_version().unwrap_or_else(|| "".to_string())
    );
    
    // GPU detection (basic - platform specific)
    let gpu = detect_gpu();
    
    HardwareInfo {
        cpu,
        gpu,
        ram_gb,
        os,
    }
}

fn detect_gpu() -> String {
    // Try to detect GPU using platform-specific methods
    #[cfg(target_os = "windows")]
    {
        // On Windows, try to get GPU info from wgpu
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapters = instance.enumerate_adapters(wgpu::Backends::all());
        if let Some(adapter) = adapters.first() {
            return adapter.get_info().name;
        }
    }
    
    #[cfg(target_os = "linux")]
    {
        // On Linux, try lspci or similar
        if let Ok(output) = std::process::Command::new("lspci")
            .args(&["-nn"])
            .output()
        {
            if let Ok(stdout) = String::from_utf8(output.stdout) {
                for line in stdout.lines() {
                    if line.contains("VGA") || line.contains("3D") {
                        if let Some(gpu_name) = line.split(':').nth(2) {
                            return gpu_name.trim().to_string();
                        }
                    }
                }
            }
        }
    }
    
    #[cfg(target_os = "macos")]
    {
        // On macOS, use system_profiler
        if let Ok(output) = std::process::Command::new("system_profiler")
            .args(&["SPDisplaysDataType"])
            .output()
        {
            if let Ok(stdout) = String::from_utf8(output.stdout) {
                for line in stdout.lines() {
                    if line.contains("Chipset Model:") {
                        if let Some(gpu_name) = line.split(':').nth(1) {
                            return gpu_name.trim().to_string();
                        }
                    }
                }
            }
        }
    }
    
    "Unknown GPU".to_string()
}

pub fn generate_benchmark_filename(hw_info: &HardwareInfo) -> String {
    // Sanitize CPU name for filename
    let cpu_clean = hw_info.cpu
        .replace("(R)", "")
        .replace("(TM)", "")
        .replace(" CPU", "")
        .replace(" @", "")
        .split_whitespace()
        .take(3) // Take first 3 words (e.g., "Intel Core i7")
        .collect::<Vec<_>>()
        .join("_");
    
    // Sanitize GPU name for filename
    let gpu_clean = hw_info.gpu
        .replace("(R)", "")
        .replace("(TM)", "")
        .replace("NVIDIA ", "")
        .replace("GeForce ", "")
        .replace("AMD ", "")
        .replace("Radeon ", "")
        .replace("Intel ", "")
        .replace("(", "")
        .replace(")", "")
        .split_whitespace()
        .filter(|s| !s.is_empty())
        .take(2) // Take first 2 words (e.g., "RTX 3080")
        .collect::<Vec<_>>()
        .join("_");
    
    format!("BENCHMARK_{}_{}.md", cpu_clean, gpu_clean)
}
