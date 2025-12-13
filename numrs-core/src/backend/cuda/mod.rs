#[derive(Debug, Clone)]
pub struct CudaBackend {}

impl CudaBackend {
    pub fn new() -> Self { Self {} }
    // execute() removed - CUDA not implemented, use ops::fast::* with dispatch system
}
