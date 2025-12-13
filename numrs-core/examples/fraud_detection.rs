use anyhow::Result;
use numrs::autograd::nn::{Linear, ReLU, Sequential, Sigmoid};
use numrs::autograd::{LossFunction, Module, Tensor};
use numrs::ops::model::{infer, load_onnx};
use numrs::ops::{add, log, mean, mul, neg, sub};
use numrs::{Array, Dataset, TrainerBuilder};
use rand::prelude::*;
use std::collections::HashMap;
use std::path::Path;

// =========================================================================================
// 1. DATA GENERATION
// =========================================================================================

/// Genera datos sintéticos para fraud detection
/// Features: [Amount, Hour, RiskScore]
/// Label: 1.0 (Fraud) | 0.0 (Legit)
fn get_fraud_data(n: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut rng = thread_rng();
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);

    for _ in 0..n {
        // Feature 1: Monto normalizado (0.0 = $0, 1.0 = $10,000)
        let amount: f32 = rng.gen_range(0.0..1.0);
        // Feature 2: Hora del día normalizada (0.0 = 00:00, 1.0 = 23:59)
        let hour: f32 = rng.gen_range(0.0..1.0);
        // Feature 3: Score de Riesgo externo (0.0 = Seguro, 1.0 = Peligroso)
        let risk: f32 = rng.gen_range(0.0..1.0);

        // LÓGICA DE NEGOCIO (Ground Truth):
        // 1. Si el riesgo es extremo (>0.95) -> FRAUDE SEGURO
        // 2. Si el riesgo es alto (>0.8) Y el monto es alto (>0.5) -> FRAUDE
        let is_fraud = (risk > 0.8 && amount > 0.5) || (risk > 0.95);

        let label = if is_fraud { 1.0 } else { 0.0 };

        x.push(vec![amount, hour, risk]);
        y.push(label);
    }
    (x, y)
}

// =========================================================================================
// 2. CUSTOM LOSS FUNCTION
// =========================================================================================

/// Binary Cross Entropy Loss personalizada
struct BCELoss;

impl LossFunction for BCELoss {
    fn compute(&self, preds: &Tensor, targets: &Tensor) -> Result<Tensor> {
        // preds: Probabilidades [0..1] (Output de Sigmoid)
        // targets: Etiquetas [0 o 1]

        // NumRs doesn't support automatic broadcasting yet, so we must manually
        // create constants with the same shape as inputs [BatchSize, 1].
        // 1. Estabilidad Numérica (Epsilon)
        // 1. Estabilidad Numérica (Epsilon)
        // Now using automatic broadcasting and f64->f32 casting!
        let epsilon = Tensor::new(Array::new(vec![1], vec![1e-7]), false);

        // preds_safe = preds + epsilon
        let preds_safe = preds.add(&epsilon)?;

        // 2. Primer Término: - y * log(p)
        let log_p = preds_safe.log()?;
        let term1 = targets.mul(&log_p)?; // y * log(p)

        // 3. Segundo Término: - (1-y) * log(1-p)
        // "1" Scalar Tensor
        let one = Tensor::new(Array::new(vec![1], vec![1.0]), false);

        // "-1" Scalar Tensor (for subtraction via add + mul(-1))
        let neg_one = Tensor::new(Array::new(vec![1], vec![-1.0]), false);

        // one_minus_y = 1 + (y * -1)
        let neg_targets = targets.mul(&neg_one)?;
        let one_minus_y = one.add(&neg_targets)?;

        // one_minus_p = 1 + (p * -1)
        let neg_preds = preds_safe.mul(&neg_one)?;
        let one_minus_p = one.add(&neg_preds)?;

        // Add epsilon to (1-p)
        let one_minus_p_safe = one_minus_p.add(&epsilon)?;
        let term2 = one_minus_y.mul(&one_minus_p_safe.log()?)?; // (1-y)*log(1-p)

        // 4. Loss = - mean(term1 + term2)
        //    Loss = mean(term1 + term2) * -1
        let total = term1.add(&term2)?;
        let mean_val = total.mean()?; // mean() is a method on Tensor
        mean_val.mul(&neg_one)
    }
}

// =========================================================================================
// MAIN
// =========================================================================================

fn main() -> Result<()> {
    // FORCE CPU BACKEND FOR EXAMPLE STABILITY
    //std::env::set_var("NUMRS_BACKEND", "cpu");

    println!("═══════════════════════════════════════════════════════════");
    println!("  🚨  NumRs Example: Fraud Detection (Tutorial 06)");
    println!("═══════════════════════════════════════════════════════════\n");

    // ---------------------------------------------------------
    // STEP 1: PREPARE DATA
    // ---------------------------------------------------------
    println!("📊 Step 1: Generating Dataset (Synthetic Fraud Data)...");

    // Train: 2000 samples
    let (train_x_raw, train_y_raw) = get_fraud_data(2000);
    // Transforma Vec<f32> a Vec<Vec<f32>> compatible con Dataset
    // (Ya viene como Vec<Vec> desde get_fraud_data, solo necesitamos envolver targets)
    let train_y_vec: Vec<Vec<f32>> = train_y_raw.iter().map(|&y| vec![y]).collect();

    // Test: 500 samples
    let (test_x_raw, test_y_raw) = get_fraud_data(500);
    let test_y_vec: Vec<Vec<f32>> = test_y_raw.iter().map(|&y| vec![y]).collect();

    let train_dataset = Dataset::new(train_x_raw, train_y_vec, 32); // Batch size 32
    let test_dataset = Dataset::new(test_x_raw, test_y_vec, 32);

    println!("   Train samples: {}", train_dataset.inputs.len());
    println!("   Test samples:  {}", test_dataset.inputs.len());

    // ---------------------------------------------------------
    // STEP 2: DEFINE MODEL
    // ---------------------------------------------------------
    println!("\n🧠 Step 2: Defining MLP Architecture...");

    // Input 3 -> L1(16) -> L2(8) -> Out(1) -> Sigmoid
    let model = Sequential::new(vec![
        Box::new(Linear::new(3, 16)?),
        Box::new(ReLU),
        Box::new(Linear::new(16, 8)?),
        Box::new(ReLU),
        Box::new(Linear::new(8, 1)?),
        Box::new(Sigmoid), // Convierte logits a probabilidad [0, 1]
    ]);

    println!("   Architecture: Linear(3->16) -> ReLU -> Linear(16->8) -> ReLU -> Linear(8->1) -> Sigmoid");

    // ---------------------------------------------------------
    // STEP 3: TRAIN
    // ---------------------------------------------------------
    println!("\n🏋️  Step 3: Training with Custom BCELoss...");

    let mut trainer = TrainerBuilder::new(model)
        .learning_rate(0.01)
        .build_adam(Box::new(BCELoss)); // Using build_adam as build_adamw is not exposed yet

    let history = trainer.fit(&train_dataset, Some(&test_dataset), 10, true)?;

    let final_loss = history.last().unwrap().0.loss;
    println!("   ✓ Training finished. Final Loss: {:.4}", final_loss);

    // ---------------------------------------------------------
    // STEP 4: EXPORT TO ONNX
    // ---------------------------------------------------------
    println!("\n📦 Step 4: Exporting to ONNX...");

    // Trace with dummy input
    let dummy_input = Tensor::new(Array::zeros(vec![1, 3]), false);

    // Ensure eval mode (though MLP has no dropout/bn here, good practice)
    trainer.model_mut().eval();
    let output = trainer.model().forward(&dummy_input)?;

    let model_path = "fraud_model.onnx.json";
    numrs::ops::export::export_to_onnx(&output, model_path)?;
    println!("   -> Saved to '{}'", model_path);

    // ---------------------------------------------------------
    // STEP 5: PRODUCTION INFERENCE
    // ---------------------------------------------------------
    println!("\n🚀 Step 5: Simulating Production Inference...\n");

    if !Path::new(model_path).exists() {
        return Err(anyhow::anyhow!("Model file not found"));
    }

    // 1. Load Model (Simulate server startup)
    let onnx_model = load_onnx(model_path)?;
    println!("   [Server] Model loaded successfully.");

    // 2. Simulate incoming requests
    let cases = vec![
        // Amount, Hour, Risk (Expected)
        (0.1, 0.5, 0.1, "Legit"),  // Low risk, low amount
        (0.9, 0.2, 0.96, "Fraud"), // High risk (>0.95 condition)
        (0.6, 0.8, 0.85, "Fraud"), // High risk (>0.8) AND high amount (>0.5)
        (0.2, 0.9, 0.85, "Legit"), // High risk BUT low amount (Should be legit per logic)
    ];

    println!("   Processing transactions:");
    println!("   ---------------------------------------------------------------");
    println!(
        "   | {:<6} | {:<6} | {:<6} | {:<8} | {:<8} | {:<6} |",
        "Amount", "Hour", "Risk", "Prob", "Pred", "Match?"
    );
    println!("   ---------------------------------------------------------------");

    let input_name = &onnx_model.graph.inputs[0].name;

    for (amt, hr, rsk, label) in cases {
        // Prepare input map
        let input_arr = Array::new(vec![1, 3], vec![amt, hr, rsk]);
        let mut inputs = HashMap::new();
        inputs.insert(input_name.clone(), input_arr);

        // Run Inference
        let results = infer(&onnx_model, inputs)?;
        let output_tensor = results
            .get("tensor_0")
            .or_else(|| results.get("output"))
            .unwrap(); // Output name might vary, check graph usually

        let prob = output_tensor.data[0];
        let pred = if prob > 0.5 { "Fraud" } else { "Legit" };
        let match_ok = if pred == label { "✅" } else { "❌" };

        println!(
            "   | {:<6.2} | {:<6.2} | {:<6.2} | {:<8.4} | {:<8} |   {}    |",
            amt, hr, rsk, prob, pred, match_ok
        );
    }
    println!("   ---------------------------------------------------------------");

    Ok(())
}
