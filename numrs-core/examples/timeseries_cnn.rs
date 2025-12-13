//! End-to-End Example: Time-Series Forecasting with CNN & ONNX
//!
//! Demonstrates:
//! 1. 1D CNN for time-series regression.
//! 2. Custom Data Generation & Normalization.
//! 3. Training with `Trainer` API (Standard NumRs pattern).
//! 4. Automatic ONNX Export of CNN models.
//!
//! Task: Predict the next value in a noisy sine wave + trend series.

use anyhow::Result;
use numrs::array::Array;
use numrs::autograd::nn::{BatchNorm1d, Conv1d, Dropout, Flatten, Sequential};
use numrs::autograd::{Linear, LossFunction, MSELoss, Module, ReLU, Tensor};
use numrs::{Dataset, TrainerBuilder};
use std::cell::RefCell;
use std::fs;
use std::rc::Rc;

// =========================================================================================
// PASO 1: Data Generation
// =========================================================================================

/// Genera datos sintéticos estacionarios (Sine wave + Noise)
/// Retorna (Inputs, Targets) como Vec<Vec<f32>> normalizados.
fn generate_synthetic_timeseries(
    num_samples: usize,
    seq_len: usize,
    offset: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut inputs = Vec::with_capacity(num_samples);
    let mut targets = Vec::with_capacity(num_samples);

    // Configuración de la serie
    let mut val = 0.0; // Centered at 0

    // Sample points with offset
    let total_points = num_samples + seq_len;
    let mut raw_data = Vec::with_capacity(total_points);

    for k in 0..total_points {
        let t = (offset + k) as f32;
        // Seasonality (Sine Wave)
        // Period = 2PI/0.1 = ~62 steps
        val = (t * 0.1).sin() * 10.0;

        // Noise
        let noise = ((k * 12345 + 6789) % 100) as f32 / 20.0 - 2.5; // [-2.5, 2.5]
        val += noise;

        raw_data.push(val);
    }

    // Normalization Stats
    let center = 0.0;
    let scale = 12.5;

    for i in 0..num_samples {
        let start_idx = i;

        let mut window = Vec::with_capacity(seq_len);
        for j in 0..seq_len {
            let norm_val = (raw_data[start_idx + j] - center) / scale;
            window.push(norm_val);
        }

        let target_val = (raw_data[start_idx + seq_len] - center) / scale;

        inputs.push(window);
        targets.push(vec![target_val]);
    }

    (inputs, targets)
}

#[derive(Clone)]
struct ForecastCNN {
    conv1: Conv1d,
    bn1: BatchNorm1d,
    dropout1: Dropout,

    conv2: Conv1d,
    bn2: BatchNorm1d,
    dropout2: Dropout,

    head: Sequential,

    relu: ReLU,
    seq_len: usize,
}

impl ForecastCNN {
    pub fn new(seq_len: usize) -> Result<Self> {
        // Layer 1: Conv(1 -> 16) | Kernel=3
        let conv1 = Conv1d::new(1, 16, 3, 1, 0)?;
        let bn1 = BatchNorm1d::new(16)?;

        // Layer 2: Conv(16 -> 32)
        let conv2 = Conv1d::new(16, 32, 3, 1, 0)?;
        let bn2 = BatchNorm1d::new(32)?;

        // Output len calculation (with padding=0)
        // L_out = (L_in - K)/S + 1
        // L1: (50 - 3)/1 + 1 = 48
        // L2: (48 - 3)/1 + 1 = 46
        let output_len = seq_len - 4;

        // Head: Flatten -> Linear(FeatureSize -> 64) -> ReLU -> Linear(64 -> 1)
        let flat_features = 32 * output_len;

        let head = Sequential::new(vec![
            Box::new(Flatten::new(1, 2)) as Box<dyn Module>,
            Box::new(Linear::new(flat_features, 64)?) as Box<dyn Module>,
            Box::new(ReLU) as Box<dyn Module>,
            Box::new(Linear::new(64, 1)?) as Box<dyn Module>,
        ]);

        Ok(Self {
            conv1,
            bn1,
            dropout1: Dropout::new(0.1),
            conv2,
            bn2,
            dropout2: Dropout::new(0.1),
            head,
            relu: ReLU,
            seq_len,
        })
    }
}

impl Module for ForecastCNN {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Input comes from Dataset as [Batch, SeqLen] (2D)
        // We need to reshape to [Batch, 1, SeqLen] for Conv1D

        let batch_size = input.shape()[0];
        let reshaped_input = input.reshape(vec![batch_size, 1, self.seq_len])?;

        // Block 1
        let x = self.conv1.forward(&reshaped_input)?;
        let x = self.bn1.forward(&x)?;
        let x = self.relu.forward(&x)?;
        let x = self.dropout1.forward(&x)?;

        // Block 2
        let x = self.conv2.forward(&x)?;
        let x = self.bn2.forward(&x)?;
        let x = self.relu.forward(&x)?;
        let x = self.dropout2.forward(&x)?;

        // Head
        let x = self.head.forward(&x)?;

        Ok(x)
    }

    fn parameters(&self) -> Vec<Rc<RefCell<Tensor>>> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn2.parameters());
        params.extend(self.head.parameters());
        params
    }

    fn train(&mut self) {
        self.conv1.train();
        self.bn1.train();
        self.dropout1.train();
        self.conv2.train();
        self.bn2.train();
        self.dropout2.train();
        self.head.train();
    }

    fn eval(&mut self) {
        self.conv1.eval();
        self.bn1.eval();
        self.dropout1.eval();
        self.conv2.eval();
        self.bn2.eval();
        self.dropout2.eval();
        self.head.eval();
    }

    fn box_clone(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

// =========================================================================================
// MAIN
// =========================================================================================
fn main() -> Result<()> {
    // FORCE CPU BACKEND TO AVOID MIXED-DEVICE ISSUES (Temporary Fix)
    std::env::set_var("NUMRS_BACKEND", "cpu");

    println!("═══════════════════════════════════════════════════════════");
    println!("  📈  NumRs: Time-Series CNN Forecasting (ONNX Ready)");
    println!("═══════════════════════════════════════════════════════════\n");

    // Hyperparams
    let seq_len = 128; // Increased from 50 to capture ~2 full cycles (Period ~63)
    let batch_size = 32;
    let epochs = 50;
    let lr = 0.001;

    // ========================================================================
    // PASO 1: Generación de Datos
    // ========================================================================
    println!("📊 PASO 1: Generando dataset (Noisy Sine Wave)\n");

    // Train: 0..1000
    let (train_x, train_y) = generate_synthetic_timeseries(5000, seq_len, 0);
    // Validation set (offset by 200 time units)
    let (test_x, test_y) = generate_synthetic_timeseries(100, seq_len, 200);

    println!("  Training samples: {}", train_x.len());
    println!("  Test samples:     {}", test_x.len());
    println!("  Sequence Len:     {}", seq_len);

    let train_dataset = Dataset::new(train_x, train_y, batch_size);
    let test_dataset = Dataset::new(test_x, test_y, batch_size);

    // ========================================================================
    // PASO 2: Arquitectura
    // ========================================================================
    println!("\n🧠 PASO 2: Definiendo arquitectura CNN\n");

    let model = ForecastCNN::new(seq_len)?;

    println!("  Arquitectura:");
    println!("    Block 1: Conv1d(1->16) -> BN -> ReLU -> Dropout");
    println!("    Block 2: Conv1d(16->32) -> BN -> ReLU -> Dropout");
    println!("    Head:    Flatten -> Linear -> ReLU -> Linear(1)");

    // ========================================================================
    // PASO 3: Entrenamiento
    // ========================================================================
    println!("\n🎯 PASO 3: Iniciando entrenamiento\n");

    let mut trainer = TrainerBuilder::new(model)
        .learning_rate(lr)
        .build_adam(Box::new(MSELoss));

    let history = trainer.fit(&train_dataset, Some(&test_dataset), epochs, true)?;
    let final_loss = history.last().unwrap().0.loss;

    println!(
        "\n  ✓ Entrenamiento completado. Loss final: {:.4}\n",
        final_loss
    );

    // ========================================================================
    // PASO 4: Visualización de Resultados
    // ========================================================================
    println!("🔍 PASO 4: Validación Visual\n");

    // Get predictions for visualization
    let (batch_x, batch_y) = test_dataset.get_batch(0)?;
    let preds = trainer.model().forward(&batch_x)?;

    let center = 0.0;
    let scale = 12.5;

    println!("  Verificando predicciones (primeros 5 ejemplos):");
    println!("  (Valores des-normalizados para interpretación)");

    for i in 0..5 {
        let pred_norm = preds.data.data[i];
        let target_norm = batch_y.data.data[i];

        let pred_real = pred_norm * scale + center;
        let target_real = target_norm * scale + center;
        let diff = (pred_real - target_real).abs();

        println!(
            "    Ex {}: Pred={:.2} | Target={:.2} | Diff={:.2}",
            i, pred_real, target_real, diff
        );
    }

    // ========================================================================
    // PASO 5: Exportar a ONNX
    // ========================================================================
    println!("\n📦 PASO 5: Exportando a ONNX\n");

    // Dummy input [1, 30] -> Reshaped inside to [1, 1, 30]
    let (dummy_x, _) = test_dataset.get_batch(0)?;
    // Take just one example to trace graph
    let dummy_input = Tensor::new(
        Array::new(vec![1, seq_len], dummy_x.data.data[0..seq_len].to_vec()),
        false, // Back to false: Exporter will treat as Input, Reshape will be skipped in graph
    );

    let model_path = "timeseries_cnn.onnx.json";

    // Run forward again to ensure clean graph captured
    let output = trainer.model().forward(&dummy_input)?;

    numrs::ops::export::export_to_onnx(&output, model_path)?;

    println!("  ✅ Modelo exportado exitosamente!");
    println!("     Archivo: {}", model_path);

    fs::write(
        "timeseries_cnn.metadata.txt",
        format!("CNN Forecasting\nLoss: {:.4}", final_loss),
    )?;

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  ✅ Demo Finalizado");
    println!("═══════════════════════════════════════════════════════════\n");

    Ok(())
}
