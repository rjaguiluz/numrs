import { Tensor, nn, TrainerBuilder, Dataset } from '../ts/native_loader';
import { Array as NumRsArray } from '../ts/array';
import * as fs from 'fs';

const { Conv1d, BatchNorm1d, Sequential, Flatten, Linear, ReLU, Dropout } = nn;

// --- Data Generation ---
// Generate synthetic sine wave data
function generateSyntheticTimeSeries(numSamples: number, seqLen: number, offset: number) {
    const inputs: number[][] = [];
    const targets: number[][] = [];

    const totalPoints = numSamples + seqLen;
    const rawData = new Float32Array(totalPoints);

    for (let k = 0; k < totalPoints; k++) {
        const t = offset + k;
        // Sine wave with period ~63 + noise
        let val = Math.sin(t * 0.1) * 10.0;
        const noise = ((k * 12345 + 6789) % 100) / 20.0 - 2.5;
        val += noise;
        rawData[k] = val;
    }

    const center = 0.0;
    const scale = 12.5;

    for (let i = 0; i < numSamples; i++) {
        const window: number[] = [];
        for (let j = 0; j < seqLen; j++) {
            window.push((rawData[i + j] - center) / scale);
        }
        const targetVal = (rawData[i + seqLen] - center) / scale;

        inputs.push(window);
        targets.push([targetVal]);
    }
    return { inputs, targets };
}

async function main() {
    console.log("═══════════════════════════════════════════════════════════");
    console.log("  📈  NumRs: Time-Series CNN Forecasting (Node.js)");
    console.log("═══════════════════════════════════════════════════════════\n");

    const SEQ_LEN = 128;
    const BATCH_SIZE = 32;
    const EPOCHS = 50;

    // 1. Data Generation
    console.log("📊 PASO 1: Generando dataset (Noisy Sine Wave)\n");
    const trainData = generateSyntheticTimeSeries(5000, SEQ_LEN, 0);
    const testData = generateSyntheticTimeSeries(100, SEQ_LEN, 200);

    const trainDataset = new Dataset(trainData.inputs, trainData.targets, BATCH_SIZE);
    const testDataset = new Dataset(testData.inputs, testData.targets, BATCH_SIZE);

    console.log(`  Training batches: ${trainDataset.numBatches}`);

    // 2. Define Architecture
    console.log("\n🧠 PASO 2: Definiendo arquitectura CNN\n");
    // Model: Unsqueeze(1) -> Conv -> BN -> ReLU -> Dropout -> ...
    const model = new Sequential();

    // IMPORTANT: Custom reshape layer we added to bindings!
    // Input [N, L] -> [N, 1, L]
    model.addUnsqueeze(1);

    // Block 1: 1 -> 16
    model.addConv1D(new Conv1d(1, 16, 3, 1, 0));
    model.addBatchNorm1D(new BatchNorm1d(16));
    model.addRelu(new ReLU());
    model.addDropout(new Dropout(0.1));

    // Block 2: 16 -> 32
    model.addConv1D(new Conv1d(16, 32, 3, 1, 0));
    model.addBatchNorm1D(new BatchNorm1d(32));
    model.addRelu(new ReLU());
    model.addDropout(new Dropout(0.1));

    // Head calculation:
    // L_in = 128
    // L1 = 128 - 3 + 1 = 126
    // L2 = 126 - 3 + 1 = 124
    // Output channels = 32
    // Flat features = 32 * 124 = 3968
    const flatFeatures = 32 * 124;

    model.addFlatten(new Flatten(1, 2)); // Flatten C and L
    model.addLinear(new Linear(flatFeatures, 64));
    model.addRelu(new ReLU());
    model.addLinear(new Linear(64, 1));

    console.log("  Arquitectura configurada.");

    // 3. Training
    console.log("\n🎯 PASO 3: Iniciando entrenamiento\n");
    const builder = new TrainerBuilder(model);
    builder.learningRate(0.001);
    const trainer = builder.build("adam", "mse");

    const history = trainer.fit(trainDataset, testDataset, EPOCHS, true);

    const last = history[history.length - 1];
    console.log(`\n  ✓ Entrenamiento completado. Loss final: ${last.trainLoss.toFixed(4)}`);

    // 4. Visualization
    console.log("\n🔍 PASO 4: Validación Visual (Primer batch de test)\n");
    // Manual forward on random sample
    // We can't easily get batch from Dataset in JS (no getter exposed in NativeDataset yet).
    // So we reconstruct a tensor from testData manually.
    const sampleInputRaw = testData.inputs[0];
    const sampleTargetRaw = testData.targets[0][0];

    const f32Input = new Float32Array(sampleInputRaw);
    // Create [1, 128]
    const inputTensor = new Tensor(new NumRsArray(f32Input, [1, SEQ_LEN]), [1, SEQ_LEN], false);

    // Forward
    const predTensor = model.forward(inputTensor);
    const predVal = predTensor.data.data[0];

    // Denormalize
    const scale = 12.5; const center = 0.0;
    const realPred = predVal * scale + center;
    const realTarget = sampleTargetRaw * scale + center;

    console.log(`  Sample 0:`);
    console.log(`    Predicted (Norm): ${predVal.toFixed(4)}`);
    console.log(`    Target    (Norm): ${sampleTargetRaw.toFixed(4)}`);
    console.log(`    Predicted (Real): ${realPred.toFixed(4)}`);
    console.log(`    Target    (Real): ${realTarget.toFixed(4)}`);
    console.log(`    Diff:             ${Math.abs(realPred - realTarget).toFixed(4)}`);

    // 5. Export
    console.log("\n📦 PASO 5: Exportando a ONNX\n");
    try {
        model.saveOnnx(inputTensor, "timeseries_cnn.onnx");
        console.log("  ✅ Modelo exportado a timeseries_cnn.onnx");
        if (fs.existsSync("timeseries_cnn.onnx")) {
            fs.unlinkSync("timeseries_cnn.onnx");
        }
    } catch (e) {
        console.error("  Error exporting:", e);
    }
}

main().catch(console.error);
