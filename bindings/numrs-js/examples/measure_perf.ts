import { Tensor, nn, TrainerBuilder, Dataset } from '../ts/native_loader';
import { Array as NumRsArray } from '../ts/array';

const { Linear, Sequential, ReLU } = nn;

function bench(name: string, fn: () => void) {
    const start = performance.now();
    fn();
    const end = performance.now();
    console.log(`[${name}] Time: ${(end - start).toFixed(2)}ms`);
}

async function main() {
    console.log("🚀 Starting Performance Benchmark");

    const N = 10000;
    const D = 100;

    // 1. Array Creation
    console.log(`\n1. Benchmarking NumRsArray creation (${N}x${D} = ${N * D} elements)`);
    const rawData = new Float32Array(N * D);
    for (let i = 0; i < N * D; i++) rawData[i] = Math.random();

    bench("NumRsArray.new (Copy JS->Rust)", () => {
        new NumRsArray(rawData, [N, D]);
    });

    // 2. Dataset Creation
    console.log(`\n2. Benchmarking Dataset creation (Vec<Vec<f64>> -> Vec<Vec<f32>>)`);
    const jsInputs: number[][] = [];
    const jsTargets: number[][] = [];
    for (let i = 0; i < N; i++) {
        const row = [];
        for (let j = 0; j < D; j++) row.push(Math.random());
        jsInputs.push(row);
        jsTargets.push([Math.random()]);
    }

    let dataset: any;
    bench("Dataset.new (Conversion + Allocation)", () => {
        dataset = new Dataset(jsInputs, jsTargets, 32);
    });

    // 2.5 Dataset Creation (Optimized)
    console.log(`\n2.5 Benchmarking Dataset.fromRaw (Float32Array -> Array)`);
    const flatInputs = new Float32Array(N * D);
    const flatTargets = new Float32Array(N * 1);
    // Fill
    for (let i = 0; i < N * D; i++) flatInputs[i] = Math.random();
    for (let i = 0; i < N; i++) flatTargets[i] = Math.random();

    bench("Dataset.fromRaw (Memcpy)", () => {
        const ds = Dataset.fromRaw(flatInputs, [N, D], flatTargets, [N, 1], 32);
    });

    // 3. Training Loop
    console.log(`\n3. Benchmarking Training Loop (1 Epoch, batch_size=32)`);
    const model = new Sequential();
    model.addLinear(new Linear(D, 64));
    model.addRelu(new ReLU());
    model.addLinear(new Linear(64, 1));

    const builder = new TrainerBuilder(model);
    const trainer = builder.build("adam", "mse");

    bench("Trainer.fit (50 Epochs)", () => {
        trainer.fit(dataset, null, 50, false);
    });
}

main().catch(console.error);
