const {
    Tensor,
    TrainerBuilder,
    Dataset,
    Sequential,
    Linear,
    ReLU,
    SGD
} = require('../dist/native_loader');

console.log("Verifying Trainer Bindings...");

// 1. Prepare Data
// XOR problem:
// [0,0] -> [0]
// [0,1] -> [1]
// [1,0] -> [1]
// [1,1] -> [0]
const inputs = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
];
const targets = [
    [0.0],
    [1.0],
    [1.0],
    [0.0]
];

console.log("Creating Dataset...");
const dataset = new Dataset(inputs, targets, 2);
console.log(`Dataset created. Num batches: ${dataset.numBatches}`);

// 2. Build Model
console.log("Building Sequential Model...");
const model = new Sequential();
// Hidden layer: 2 -> 4
const lin1 = new Linear(2, 4);
const relu = new ReLU();
// Output layer: 4 -> 1
const lin2 = new Linear(4, 1);

model.addLinear(lin1);
model.addRelu(relu);
model.addLinear(lin2);

// 3. Setup Trainer
console.log("Setting up Trainer...");
const builder = new TrainerBuilder(model);
builder.learningRate(0.1);

// Build with SGD and MSE
const trainer = builder.build("sgd", "mse");
console.log("Trainer built.");

// 4. Train
console.log("Starting training (5 epochs)...");
try {
    const history = trainer.fit(dataset, null, 5, true);
    console.log("Training complete!");
    console.log("History length:", history.length);
    console.log("Last Epoch Metrics:", history[history.length - 1]);
} catch (e) {
    console.error("Training failed:", e);
}

console.log("Verification finished.");
