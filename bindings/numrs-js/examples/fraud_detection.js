const { Tensor, nn, optimizers: optim, NativeTensor, TrainerBuilder, Dataset } = require('../dist/native_loader');
const fs = require('fs');
const { Array: NumRsArray } = require('../dist/array');
// const data = require('../dist/data'); // Removing old data import

const { Linear, Sequential, ReLU, Sigmoid } = nn;

// Helper: Random Float in range
function randRange(min, max) {
    return Math.random() * (max - min) + min;
}

// 1. Data Generation
function getFraudData(n) {
    const xData = [];
    const yData = [];

    for (let i = 0; i < n; i++) {
        const amount = randRange(0.0, 1.0);
        const hour = randRange(0.0, 1.0);
        const risk = randRange(0.0, 1.0);

        // Simple fraud logic
        const isFraud = (risk > 0.8 && amount > 0.5) || (risk > 0.95);

        xData.push([amount, hour, risk]);
        yData.push([isFraud ? 1.0 : 0.0]);
    }
    return { x: xData, y: yData };
}

async function main() {
    console.log("🚀 Running Fraud Detection Example (MLP) with Trainer API...");

    // 2. Prepare Data
    console.log("📊 Generating Dataset...");
    const { x, y } = getFraudData(200);
    // Use Native Dataset (renamed to Dataset in exports)
    const dataset = new Dataset(x, y, 32);
    console.log(`   Created dataset with ${dataset.numBatches} batches.`);

    // 3. Define Model
    console.log("\n🧠 Defining Architecture...");
    const model = new Sequential();
    model.addLinear(new Linear(3, 16));
    model.addRelu(new ReLU());
    model.addLinear(new Linear(16, 8));
    model.addRelu(new ReLU());
    model.addLinear(new Linear(8, 1));
    model.addSigmoid(new Sigmoid());
    console.log("   Linear(3->16) -> ReLU -> Linear(16->8) -> ReLU -> Linear(8->1) -> Sigmoid");

    // 4. Training Setup via Trainer API
    const EPOCHS = 50;

    // Validation Data
    const valDataRaw = getFraudData(50);
    const valDataset = new Dataset(valDataRaw.x, valDataRaw.y, 32);

    console.log("\n🏋️  Training via Trainer...");
    const builder = new TrainerBuilder(model);
    builder.learningRate(0.01);

    // Using Adam and MSE
    const trainer = builder.build("adam", "mse");

    const history = trainer.fit(dataset, valDataset, EPOCHS, true);

    console.log("✓ Training Complete");

    // Log final metrics (napi converts to camelCase)
    const lastMetrics = history[history.length - 1];
    console.log(`   Final Train Loss: ${lastMetrics.trainLoss.toFixed(4)}`);
    if (lastMetrics.valLoss !== undefined) {
        console.log(`   Final Val Loss:   ${lastMetrics.valLoss.toFixed(4)}`);
    }

    // 6. Simulate Production Inference (Manual Forward Pass)
    console.log("\n🚀 Simulating Production Inference (Manual Forward)...");
    const cases = [
        { amt: 0.1, hr: 0.5, rsk: 0.1, label: "Legit" },  // Low risk
        { amt: 0.9, hr: 0.2, rsk: 0.96, label: "Fraud" }, // High risk > 0.95
        { amt: 0.6, hr: 0.8, rsk: 0.85, label: "Fraud" }, // Risk > 0.8 && Amt > 0.5
        { amt: 0.2, hr: 0.9, rsk: 0.85, label: "Legit" }, // High risk but Low Amt
    ];

    console.log("   ---------------------------------------------------------------");
    console.log("   | Amount | Hour   | Risk   | Prob     | Pred     | Match? |");
    console.log("   ---------------------------------------------------------------");

    for (const c of cases) {
        // NativeTensor expects Float32Array for data
        const inputDataCase = new Float32Array([c.amt, c.hr, c.rsk]);
        // NumRsArray wrapping Float32Array
        // Tensor wrapping NumRsArray
        const inputCase = new Tensor(new NumRsArray(inputDataCase, [1, 3]), [1, 3], false);

        const outCase = model.forward(inputCase);

        // outCase is Tensor. data getter returns NumRsArray. data getter returns Float32Array.
        // Or directly values() if bound? 
        // Let's assume .data.data access pattern from TS example.
        const probCase = outCase.data.data[0];
        const predStrCase = probCase > 0.5 ? "Fraud" : "Legit";
        const match = predStrCase === c.label ? "✅" : "❌";

        console.log(`   | ${c.amt.toFixed(2).padEnd(6)} | ${c.hr.toFixed(2).padEnd(6)} | ${c.rsk.toFixed(2).padEnd(6)} | ${probCase.toFixed(4).padEnd(8)} | ${predStrCase.padEnd(8)} |   ${match}    |`);
    }
    console.log("   ---------------------------------------------------------------");

    // 7. Export (Optional, mocking ONNX save via binding if available)
    if (model.saveOnnx) {
        console.log("\n💾 Exporting to ONNX...");
        try {
            const dummy_f32 = new Float32Array([0.0, 0.0, 0.0]);
            const dummy_input = new Tensor(new NumRsArray(dummy_f32, [1, 3]), [1, 3], false);
            model.saveOnnx(dummy_input, "fraud_model_trainer.onnx");
            console.log("   Model saved.");
            if (fs.existsSync("fraud_model_trainer.onnx")) {
                fs.unlinkSync("fraud_model_trainer.onnx");
                console.log("   Cleaned up.");
            }
        } catch (e) {
            console.log("   Export failed: " + e);
        }
    }
}

main().catch(console.error);
