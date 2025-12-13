use crate::llo::ElementwiseKind;

/// Generate a simple WGSL elementwise kernel for Add/Mul
pub fn elementwise_wgsl(kind: ElementwiseKind, _inputs: Vec<usize>, output_shape: Vec<usize>) -> String {
    let op = match kind {
        ElementwiseKind::Add => "a + b",
        ElementwiseKind::Mul => "a * b",
        ElementwiseKind::Sub => "a - b",
        ElementwiseKind::Div => "a / b",
        ElementwiseKind::Sqrt => "sqrt(a)",
        ElementwiseKind::Sin => "sin(a)",
        ElementwiseKind::Cos => "cos(a)",
        ElementwiseKind::Pow => "pow(a, b)",
        ElementwiseKind::Abs => "abs(a)",
        ElementwiseKind::Neg => "-a",
        ElementwiseKind::Exp => "exp(a)",
        ElementwiseKind::Log => "log(a)",
        ElementwiseKind::Tan => "tan(a)",
        ElementwiseKind::Asin => "asin(a)",
        ElementwiseKind::Acos => "acos(a)",
        ElementwiseKind::Atan => "atan(a)",
        ElementwiseKind::Relu => "max(a, 0.0)",
        ElementwiseKind::LeakyRelu => "select(0.01 * a, a, a > 0.0)",
        ElementwiseKind::Sigmoid => "1.0 / (1.0 + exp(-a))",
        ElementwiseKind::Tanh => "tanh(a)",
        ElementwiseKind::Softplus => "log(1.0 + exp(a))",
    };

    format!(r#"// WGSL elementwise kernel (prototype)
fn compute(a: f32, b: f32) -> f32 {{
    {}
}}
// output shape: {:?}
"#, op, output_shape)
}

pub fn reduction_wgsl(_axis: Option<usize>, _inputs: Vec<usize>, _output_shape: Vec<usize>) -> String {
    "// WGSL reduction kernel template (prototype)\nfn compute_reduce(a: array<f32>) -> f32 { /* ... */ }\n".to_string()
}
