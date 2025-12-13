//! numrs-ops: Operation inventory and validation tool (AUTO-SCANNING)
//!
//! This tool automatically scans the codebase and generates a comprehensive table showing:
//! - All operations (detected by scanning src/ops/)
//! - Backend support for each operation (detected from source code)
//! - Dependencies between operations (extracted from imports)
//! - Implementation status (auto-detected)
//!
//! Usage: cargo run --bin numrs-ops [--format table|json|csv]

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::fs;
use anyhow::Result;

#[derive(Debug, Clone, PartialEq)]
enum OperationStatus {
    Implemented,
    Planned,
    NotStarted,
}

#[derive(Debug, Clone)]
struct Operation {
    name: String,
    category: String,
    subcategory: Option<String>,
    status: OperationStatus,
    backends: BackendSupport,
    dependencies: Vec<String>,
    fast_api: bool, // Available in ops::*
}

#[derive(Debug, Clone)]
struct BackendSupport {
    cpu_scalar: bool,
    cpu_simd: bool,
    blas: bool,
    webgpu: bool,
    cuda: bool,
    metal: bool,
}

impl BackendSupport {
    fn none() -> Self {
        Self {
            cpu_scalar: false,
            cpu_simd: false,
            blas: false,
            webgpu: false,
            cuda: false,
            metal: false,
        }
    }
}

/// Scan the src/ops directory to find all implemented operations
fn scan_operations(ops_dir: &Path) -> Result<Vec<Operation>> {
    let mut operations = Vec::new();
    
    // Scan all categories in src/ops
    for category in &["elementwise", "reduction", "linalg", "shape", "stats", "random"] {
        let category_path = ops_dir.join(category);
        if !category_path.exists() {
            continue;
        }
        
        scan_category(&category_path, category, &mut operations)?;
    }
    
    // Add planned operations that might not exist yet
    add_planned_operations(&mut operations);
    
    Ok(operations)
}

/// Scan a category directory for operations
fn scan_category(category_path: &Path, category: &str, operations: &mut Vec<Operation>) -> Result<()> {
    for entry in fs::read_dir(category_path)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_file() && path.extension().map(|e| e == "rs").unwrap_or(false) {
            let filename = path.file_stem().unwrap().to_str().unwrap();
            
            // Skip mod.rs
            if filename == "mod" {
                continue;
            }
            
            // Parse the operation from the file
            if let Ok(op) = parse_operation_file(&path, category, None) {
                operations.push(op);
            }
        } else if path.is_dir() {
            // Handle subcategories (e.g., elementwise/binary, elementwise/unary)
            let subcategory = path.file_name().unwrap().to_str().unwrap();
            
            for sub_entry in fs::read_dir(&path)? {
                let sub_entry = sub_entry?;
                let sub_path = sub_entry.path();
                
                if sub_path.is_file() && sub_path.extension().map(|e| e == "rs").unwrap_or(false) {
                    let filename = sub_path.file_stem().unwrap().to_str().unwrap();
                    
                    if filename == "mod" {
                        continue;
                    }
                    
                    if let Ok(op) = parse_operation_file(&sub_path, category, Some(subcategory)) {
                        operations.push(op);
                    }
                }
            }
        }
    }
    
    Ok(())
}

/// Parse an operation file to extract metadata
fn parse_operation_file(path: &Path, category: &str, subcategory: Option<&str>) -> Result<Operation> {
    let content = fs::read_to_string(path)?;
    let filename = path.file_stem().unwrap().to_str().unwrap();
    
    // Determine backend support by analyzing the code
    let backends = detect_backend_support(&content);
    
    // Extract dependencies from use statements
    let dependencies = extract_dependencies(&content);
    
    // Check if it's in the fast API (public function with #[inline(always)])
    let fast_api = content.contains("#[inline(always)]") || 
                    (content.contains("pub fn ") && !content.contains("pub(crate)"));
    
    Ok(Operation {
        name: filename.to_string(),
        category: category.to_string(),
        subcategory: subcategory.map(|s| s.to_string()),
        status: OperationStatus::Implemented,
        backends,
        dependencies,
        fast_api,
    })
}

/// Detect backend support by analyzing code patterns
fn detect_backend_support(content: &str) -> BackendSupport {
    let mut support = BackendSupport::none();
    
    // Check for dispatch table usage (full backend support)
    if content.contains("get_dispatch_table()") {
        support.cpu_scalar = true;
        
        // Check which kinds are used
        if content.contains("ElementwiseKind") || content.contains("ReductionKind") {
            // These use SIMD backends
            support.cpu_simd = true;
            support.webgpu = true;
            
            // Metal supports elementwise operations (like WebGPU)
            #[cfg(target_os = "macos")]
            {
                support.metal = true;
            }
            
            // If not on macOS, still detect Metal support from validation
            if !cfg!(target_os = "macos") {
                // Check if dispatch system would enable Metal
                support.metal = true; // Will be available on macOS
            }
        }
        
        // Check for BLAS support
        if content.contains("matmul") || content.contains("table.matmul") {
            support.blas = true;
            
            // Metal also supports matmul
            support.metal = true;
        }
        
        // Check for dot product
        if content.contains("table.dot") || content.contains("dot(") {
            support.blas = true;
        }
    } else {
        // CPU-only operations (reshape, transpose, concat, etc.)
        support.cpu_scalar = true;
    }
    
    // Check for specific backend mentions
    if content.contains("ReductionKind::Sum") {
        support.blas = true; // sum supports BLAS
    }
    
    support
}

/// Extract dependencies from use statements
fn extract_dependencies(content: &str) -> Vec<String> {
    let mut deps = HashSet::new();
    
    // Look for use crate::ops::... statements
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("use crate::ops::") {
            // Extract function names from use statement
            // Handle both single imports: use crate::ops::mean;
            // And grouped imports: use crate::ops::{mean, sum};
            
            if let Some(after_ops) = trimmed.strip_prefix("use crate::ops::") {
                let cleaned = after_ops.trim_end_matches(';').trim();
                
                // Check if it's a grouped import with braces
                if cleaned.starts_with('{') && cleaned.ends_with('}') {
                    // Extract multiple imports: {mean, sum, max}
                    let inner = &cleaned[1..cleaned.len()-1];
                    for func in inner.split(',') {
                        let func_name = func.trim();
                        if !func_name.is_empty() && !func_name.contains("::") {
                            deps.insert(func_name.to_string());
                        }
                    }
                } else {
                    // Single import: mean or mean::something
                    // Only take the first part before ::
                    let func_name = cleaned.split("::").next().unwrap_or("").trim();
                    if !func_name.is_empty() && func_name != "{" && !func_name.contains("*") {
                        deps.insert(func_name.to_string());
                    }
                }
            }
        }
        
        // Also check for direct ops:: calls in the code (more careful parsing)
        if trimmed.contains("ops::") && !trimmed.starts_with("use ") && !trimmed.starts_with("//") {
            // Find all occurrences of ops::
            let mut rest = trimmed;
            while let Some(pos) = rest.find("ops::") {
                rest = &rest[pos + 5..]; // Skip "ops::"
                
                // Extract the function name (alphanumeric + underscore only)
                let func_name: String = rest.chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                
                if !func_name.is_empty() && func_name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    deps.insert(func_name);
                }
            }
        }
    }
    
    deps.into_iter().collect()
}

/// Add known planned operations that don't exist yet
fn add_planned_operations(operations: &mut Vec<Operation>) {
    let existing: HashSet<String> = operations.iter().map(|op| op.name.clone()).collect();
    
    // List of planned operations
    let planned = vec![
        ("reduction", None, "argmin"),
        ("reduction", None, "std"),
        ("linalg", None, "outer"),
        ("linalg", None, "mv"),
        ("linalg", None, "vv"),
        ("linalg", None, "mm"),
        ("linalg", Some("advanced"), "inverse"),
        ("linalg", Some("advanced"), "det"),
        ("linalg", Some("advanced"), "solve"),
        ("linalg", Some("advanced"), "cholesky"),
        ("linalg", Some("advanced"), "qr"),
        ("linalg", Some("advanced"), "svd"),
        ("linalg", Some("advanced"), "eig"),
        ("shape", None, "permute"),
        ("shape", None, "flatten"),
        ("shape", None, "squeeze"),
        ("shape", None, "unsqueeze"),
        ("shape", None, "broadcast_to"),
        ("shape", None, "expand_dims"),
        ("shape", None, "stack"),
        ("shape", None, "tile"),
        ("shape", None, "repeat"),
        ("shape", None, "slice"),
        ("shape", None, "take"),
        ("shape", None, "gather"),
        ("stats", None, "log_softmax"),
        ("random", None, "rand"),
        ("random", None, "randn"),
        ("random", None, "randint"),
        ("random", None, "seed"),
    ];
    
    for (category, subcategory, name) in planned {
        if !existing.contains(name) {
            operations.push(Operation {
                name: name.to_string(),
                category: category.to_string(),
                subcategory: subcategory.map(|s| s.to_string()),
                status: OperationStatus::Planned,
                backends: BackendSupport::none(),
                dependencies: infer_dependencies(name),
                fast_api: false,
            });
        }
    }
}

/// Infer dependencies for planned operations based on known patterns
fn infer_dependencies(op_name: &str) -> Vec<String> {
    match op_name {
        "std" => vec!["variance".to_string(), "sqrt".to_string()],
        "argmin" => vec![],
        "outer" => vec!["mul".to_string(), "reshape".to_string()],
        "mv" => vec!["matmul".to_string(), "reshape".to_string()],
        "vv" => vec!["dot".to_string()],
        "mm" => vec!["matmul".to_string()],
        "flatten" => vec!["reshape".to_string()],
        "broadcast_to" => vec!["reshape".to_string()],
        "expand_dims" => vec!["unsqueeze".to_string()],
        "stack" => vec!["expand_dims".to_string(), "concat".to_string()],
        "slice" => vec!["reshape".to_string()],
        "log_softmax" => vec!["softmax".to_string(), "log".to_string()],
        "randn" => vec!["rand".to_string()],
        "randint" => vec!["rand".to_string()],
        _ => vec![],
    }
}

fn build_operation_registry() -> Vec<Operation> {
    // Automatically scan the src/ops directory
    let ops_dir = Path::new("src/ops");
    
    match scan_operations(ops_dir) {
        Ok(ops) => ops,
        Err(e) => {
            eprintln!("Error scanning operations: {}", e);
            eprintln!("Falling back to empty registry");
            vec![]
        }
    }
}

fn print_table(ops: &[Operation]) -> String {
    let mut output = String::new();
    
    output.push_str("# numrs Operations Check (AUTO-SCANNED)\n\n");
    output.push_str(&format!("*Generated: {}*\n\n", chrono::Local::now().format("%Y-%m-%d %H:%M:%S")));
    
    // Summary statistics
    let total = ops.len();
    let implemented = ops.iter().filter(|o| o.status == OperationStatus::Implemented).count();
    let planned = ops.iter().filter(|o| o.status == OperationStatus::Planned).count();
    let not_started = ops.iter().filter(|o| o.status == OperationStatus::NotStarted).count();
    let with_fast_api = ops.iter().filter(|o| o.fast_api).count();
    
    output.push_str("## Summary\n\n");
    output.push_str(&format!("- **Total operations**: {}\n", total));
    output.push_str(&format!("- ✅ **Implemented**: {} ({:.1}%)\n", implemented, (implemented as f32 / total as f32) * 100.0));
    output.push_str(&format!("- 🔄 **Planned**: {} ({:.1}%)\n", planned, (planned as f32 / total as f32) * 100.0));
    output.push_str(&format!("- ⭕ **Not started**: {} ({:.1}%)\n", not_started, (not_started as f32 / total as f32) * 100.0));
    output.push_str(&format!("- 🚀 **Fast API**: {} operations\n\n", with_fast_api));
    
    // Group by category
    let mut by_category: HashMap<String, Vec<&Operation>> = HashMap::new();
    for op in ops {
        by_category.entry(op.category.clone()).or_insert_with(Vec::new).push(op);
    }
    
    let category_order = vec!["elementwise", "reduction", "linalg", "shape", "stats", "random"];
    
    for category in category_order {
        if let Some(category_ops) = by_category.get(category) {
            output.push_str(&format!("## {} ({} ops)\n\n", category.to_uppercase(), category_ops.len()));
            
            // Print table header with backend columns
            output.push_str("| Operation | Status | CPU-Scalar | CPU-SIMD | BLAS | WebGPU | CUDA | Metal | Fast API | Dependencies |\n");
            output.push_str("|-----------|--------|------------|----------|------|--------|------|-------|----------|--------------|\n");
            
            // Sort by subcategory and name
            let mut sorted_ops = category_ops.clone();
            sorted_ops.sort_by(|a, b| {
                match (&a.subcategory, &b.subcategory) {
                    (Some(sub_a), Some(sub_b)) => sub_a.cmp(sub_b).then(a.name.cmp(&b.name)),
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    (None, None) => a.name.cmp(&b.name),
                }
            });
            
            for op in sorted_ops {
                let status_icon = match op.status {
                    OperationStatus::Implemented => "✅",
                    OperationStatus::Planned => "🔄",
                    OperationStatus::NotStarted => "⭕",
                };
                
                let full_path = if let Some(sub) = &op.subcategory {
                    format!("{}/{}/{}", op.category, sub, op.name)
                } else {
                    format!("{}/{}", op.category, op.name)
                };
                
                let cpu_scalar = if op.backends.cpu_scalar { "✓" } else { "" };
                let cpu_simd = if op.backends.cpu_simd { "✓" } else { "" };
                let blas = if op.backends.blas { "✓" } else { "" };
                let webgpu = if op.backends.webgpu { "✓" } else { "" };
                let cuda = if op.backends.cuda { "✓" } else { "" };
                let metal = if op.backends.metal { "✓" } else { "" };
                let fast_api = if op.fast_api { "🚀" } else { "" };
                
                let deps = if op.dependencies.is_empty() {
                    "-".to_string()
                } else {
                    op.dependencies.join(", ")
                };
                
                output.push_str(&format!("| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                    full_path,
                    status_icon,
                    cpu_scalar,
                    cpu_simd,
                    blas,
                    webgpu,
                    cuda,
                    metal,
                    fast_api,
                    deps
                ));
            }
            
            output.push('\n');
        }
    }
    
    // Backend coverage summary
    output.push_str("## Backend Coverage Summary\n\n");
    
    let cpu_scalar_count = ops.iter().filter(|o| o.backends.cpu_scalar).count();
    let cpu_simd_count = ops.iter().filter(|o| o.backends.cpu_simd).count();
    let blas_count = ops.iter().filter(|o| o.backends.blas).count();
    let webgpu_count = ops.iter().filter(|o| o.backends.webgpu).count();
    let cuda_count = ops.iter().filter(|o| o.backends.cuda).count();
    let metal_count = ops.iter().filter(|o| o.backends.metal).count();
    
    output.push_str("| Backend | Operations | Coverage (of implemented) |\n");
    output.push_str("|---------|------------|---------------------------|\n");
    output.push_str(&format!("| CPU-Scalar | {} | {:.1}% |\n", cpu_scalar_count, (cpu_scalar_count as f32 / implemented as f32) * 100.0));
    output.push_str(&format!("| CPU-SIMD | {} | {:.1}% |\n", cpu_simd_count, (cpu_simd_count as f32 / implemented as f32) * 100.0));
    output.push_str(&format!("| BLAS | {} | {:.1}% |\n", blas_count, (blas_count as f32 / implemented as f32) * 100.0));
    output.push_str(&format!("| WebGPU | {} | {:.1}% |\n", webgpu_count, (webgpu_count as f32 / implemented as f32) * 100.0));
    output.push_str(&format!("| CUDA | {} | {:.1}% |\n", cuda_count, (cuda_count as f32 / implemented as f32) * 100.0));
    output.push_str(&format!("| Metal | {} | {:.1}% |\n", metal_count, (metal_count as f32 / implemented as f32) * 100.0));
    
    // Print dependency graph
    output.push_str("\n## Dependency Analysis\n\n");
    
    let ops_with_deps: Vec<_> = ops.iter()
        .filter(|o| !o.dependencies.is_empty())
        .collect();
    
    if !ops_with_deps.is_empty() {
        output.push_str("### Operations with dependencies:\n\n");
        for op in ops_with_deps {
            let mut missing_deps = Vec::new();
            for dep in &op.dependencies {
                let dep_op = ops.iter().find(|o| o.name == *dep);
                if let Some(dep_op) = dep_op {
                    if dep_op.status != OperationStatus::Implemented {
                        missing_deps.push(dep.as_str());
                    }
                } else {
                    missing_deps.push(dep.as_str());
                }
            }
            
            let full_path = if let Some(sub) = &op.subcategory {
                format!("{}/{}/{}", op.category, sub, op.name)
            } else {
                format!("{}/{}", op.category, op.name)
            };
            
            if !missing_deps.is_empty() {
                output.push_str(&format!("- `{}`: ⚠️ Missing deps: {}\n", full_path, missing_deps.join(", ")));
            } else {
                output.push_str(&format!("- `{}`: ✅ All deps satisfied ({})\n", full_path, op.dependencies.join(", ")));
            }
        }
    }
    
    output.push_str("\n---\n\n");
    output.push_str("*Auto-generated by `cargo run --bin numrs-ops` (scans src/ops/)*\n");
    
    output
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let format = args.get(1).map(|s| s.as_str()).unwrap_or("table");
    
    eprintln!("🔍 Scanning src/ops/ for implemented operations...");
    let ops = build_operation_registry();
    eprintln!("✅ Found {} operations", ops.len());
    
    match format {
        "table" | _ => {
            let table_output = print_table(&ops);
            
            // Write to OPS_CHECK.md file
            let output_path = "OPS_CHECK.md";
            std::fs::write(output_path, &table_output)
                .expect("Failed to write OPS_CHECK.md");
            
            // Also print to stdout
            print!("{}", table_output);
            
            eprintln!("\n✅ Generated: {}", output_path);
        }
    }
    
    Ok(())
}
