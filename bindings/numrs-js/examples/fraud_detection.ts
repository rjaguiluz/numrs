import { Tensor, nn, NativeTensor, TrainerBuilder, Dataset } from '../ts/native_loader'; // Directly import from loader to ensure access or re-export
import * as fs from 'fs';
import { Array as NumRsArray } from '../ts/array';

// Helper: Random Float in range
function randRange(min: number, max: number): number {
    return Math.random() * (max - min) + min;
}

// 1. Data Generation
function getFraudData(n: number): { x: number[][], y: number[][] } {
    const xData: number[][] = [];
    const yData: number[][] = [];

    for (let i = 0; i < n; i++) {
        const amount = randRange(0.0, 1.0);
        const hour = randRange(0.0, 1.0);
        const risk = randRange(0.0, 1.0);

        const isFraud = (risk > 0.8 && amount > 0.5) || (risk > 0.95);

        xData.push([amount, hour, risk]);
        yData.push([isFraud ? 1.0 : 0.0]);
    }
    return { x: xData, y: yData };
}

async function main() {
    console.log(" Running Fraud Detection Example (MLP) with Trainer API...");

    // 2. Prepare Data
    console.log("📊 Generating Dataset...");
    const { x, y } = getFraudData(200);
    // Use Native Dataset
    const dataset = new Dataset(x, y, 32);
    console.log(`   Created dataset with ${dataset.numBatches} batches.`);

    // 3. Define Model
    console.log("\n🧠 Defining Architecture...");
    const model = new nn.Sequential();
    model.addLinear(new nn.Linear(3, 16));
    model.addRelu(new nn.ReLU());
    model.addLinear(new nn.Linear(16, 8));
    model.addRelu(new nn.ReLU());
    model.addLinear(new nn.Linear(8, 1));
    model.addSigmoid(new nn.Sigmoid());
    console.log("   Linear(3->16) -> ReLU -> Linear(16->8) -> ReLU -> Linear(8->1) -> Sigmoid");

    // 4. Training Setup using Trainer API
    const EPOCHS = 10;

    // Prepare validation data for the trainer (optional)
    const valDataRaw = getFraudData(50);
    const valDataset = new Dataset(valDataRaw.x, valDataRaw.y, 32);

    console.log("\n🏋️  Training via Trainer...");
    const builder = new TrainerBuilder(model);
    builder.learningRate(0.01);

    // Using Adam and MSE (compatible with current binary regression setup)
    const trainer = builder.build("adam", "mse");

    const history = trainer.fit(dataset, valDataset, EPOCHS, true);

    console.log("✓ Training Complete");

    // Log history
    const lastMetrics = history[history.length - 1];
    console.log(`   Final Train Loss: ${lastMetrics.trainLoss.toFixed(4)}`);
    if (lastMetrics.valLoss !== undefined) {
        console.log(`   Final Val Loss:   ${lastMetrics.valLoss.toFixed(4)}`);
    }

    // 6. Simulate Production Inference (Manual Forward Pass)
    console.log("\n🚀 Simulating Production Inference...");
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
        const inputDataCase = new Float32Array([c.amt, c.hr, c.rsk]);
        const inputCase = new Tensor(new NumRsArray(inputDataCase, [1, 3]), [1, 3], false);
        const outCase = model.forward(inputCase);
        const probCase = outCase.data.data[0];
        const predStrCase = probCase > 0.5 ? "Fraud" : "Legit";
        const match = predStrCase === c.label ? "✅" : "❌";

        console.log(`   | ${c.amt.toFixed(2).padEnd(6)} | ${c.hr.toFixed(2).padEnd(6)} | ${c.rsk.toFixed(2).padEnd(6)} | ${probCase.toFixed(4).padEnd(8)} | ${predStrCase.padEnd(8)} |   ${match}    |`);
    }
    console.log("   ---------------------------------------------------------------");

    // 7. Export to ONNX
    console.log("\n💾 Exporting to ONNX...");
    try {
        const dummy_input_array = x[0]; // Is [3]
        const dummy_f32 = new Float32Array(dummy_input_array);
        const dummy_input = new Tensor(new NumRsArray(dummy_f32, [1, 3]), [1, 3], false);

        model.saveOnnx(dummy_input, "fraud_model_trace.onnx");
        console.log("   Model saved to fraud_model_trace.onnx");
        fs.unlinkSync("fraud_model_trace.onnx");
        console.log("   Cleaned up fraud_model_trace.onnx");
    } catch (e) {
        console.error("   Failed to save ONNX model:", e);
    }
}

main().catch(console.error);
