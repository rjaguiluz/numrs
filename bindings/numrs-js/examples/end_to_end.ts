import { Tensor, nn, TrainerBuilder, Dataset } from '../ts/native_loader';
import { Array as NumRsArray } from '../ts/array';

const { Linear, Sequential, ReLU } = nn;

async function main() {
    console.log("═══════════════════════════════════════════════════════════");
    console.log("  🎯 NumRs End-to-End (Node.js): Training → ONNX → Inference");
    console.log("═══════════════════════════════════════════════════════════\n");

    // ========================================================================
    // PASO 1: Preparar dataset
    // ========================================================================
    console.log("📊 PASO 1: Preparando dataset (Clasificación 2D Simulado)\n");

    const trainInputs: number[][] = [];
    const trainTargets: number[][] = [];

    // Generar datos sintéticos
    for (let i = 0; i < 40; i++) {
        const x = i * 0.05;
        const y = i * 0.03;

        trainInputs.push([x, y]);

        // Target classification logic
        if (x + y < 1.0) {
            trainTargets.push([1.0, 0.0]); // Class 0
        } else {
            trainTargets.push([0.0, 1.0]); // Class 1
        }
    }

    // Dataset wrapping
    const dataset = new Dataset(trainInputs, trainTargets, 8);
    console.log(`  ✓ Generados ${trainInputs.length} ejemplos`);

    // ========================================================================
    // PASO 2: Crear y entrenar modelo
    // ========================================================================
    console.log("\n🧠 PASO 2: Creando modelo neuronal\n");
    const model = new Sequential();
    // 2 -> 8 (ReLU) -> 4 (ReLU) -> 2
    model.addLinear(new Linear(2, 8));
    model.addRelu(new ReLU());
    model.addLinear(new Linear(8, 4));
    model.addRelu(new ReLU());
    model.addLinear(new Linear(4, 2));

    console.log("  Arquitectura: 2 -> 8 -> 4 -> 2");

    // Trainer setup
    const builder = new TrainerBuilder(model);
    builder.learningRate(0.05);
    const trainer = builder.build("adam", "mse");

    console.log("  Entrenando (50 epochs)...\n");
    const history = trainer.fit(dataset, null, 50, false);

    // Show logs
    console.log("  ┌────────┬─────────────┐");
    console.log("  │ Epoch  │    Loss     │");
    console.log("  ├────────┼─────────────┤");
    history.forEach((metrics, i) => {
        if (i % 10 === 0 || i === history.length - 1) {
            console.log(`  │  ${i.toString().padEnd(3)}   │   ${metrics.trainLoss.toFixed(6)}  │`);
        }
    });
    console.log("  └────────┴─────────────┘\n");

    // ========================================================================
    // PASO 3: Validación (Inference)
    // ========================================================================
    console.log("🔍 PASO 3: Validando predicciones\n");

    // Test samples
    const testSamples = [
        { val: [0.1, 0.1], label: "Class 0" },
        { val: [0.5, 0.2], label: "Class 0" },
        { val: [0.8, 0.5], label: "Class 1" },
        { val: [1.2, 0.8], label: "Class 1" },
    ];

    console.log("  ┌──────────────┬─────────────┬────────────┐");
    console.log("  │    Input     │  Esperado   │ Predicción │");
    console.log("  ├──────────────┼─────────────┼────────────┤");

    for (const sample of testSamples) {
        // Create 2D tensor for single sample: [1, 2]
        const inputArr = new Float32Array(sample.val);
        const input = new Tensor(new NumRsArray(inputArr, [1, 2]), [1, 2], false);

        // Forward
        const out = model.forward(input);
        const data = out.data.data; // Float32Array

        const predictedClass = data[0] > data[1] ? "Class 0" : "Class 1";

        console.log(`  │ [${sample.val}]  │ ${sample.label.padEnd(8)}    │ ${predictedClass.padEnd(10)} │`);
    }
    console.log("  └──────────────┴─────────────┴────────────┘\n");

    // ========================================================================
    // PASO 4: ONNX Export
    // ========================================================================
    console.log("💾 PASO 4: Exportando modelo a ONNX\n");
    try {
        const dummyF32 = new Float32Array([0.5, 0.5]);
        const dummyInput = new Tensor(new NumRsArray(dummyF32, [1, 2]), [1, 2], false);
        model.saveOnnx(dummyInput, "end_to_end_model.onnx");
        console.log("  ✓ Modelo exportado a end_to_end_model.onnx");

        // Cleanup
        const fs = require('fs');
        if (fs.existsSync("end_to_end_model.onnx")) {
            fs.unlinkSync("end_to_end_model.onnx");
            console.log("  ✓ Cleanup OK");
        }
    } catch (e) {
        console.error("  Error exportando ONNX:", e);
    }

    console.log("\n🎉 Demo Finalizado!");
}

main().catch(console.error);
