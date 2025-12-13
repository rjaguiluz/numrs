
console.log("Module script starting...");

import init, { NumRsArray, Tensor, Sequential, Linear, ReLU, Sigmoid, Trainer, OnnxModelWrapper } from '../../pkg/numrs_wasm.js?v=v3';

console.log("Imports successful.");

let wasmInstance = null;
let trainedModel = null;
let savedJson = null;

function log(msg) {
    const logEl = document.getElementById('log');
    if (logEl) {
        logEl.textContent += `[${new Date().toLocaleTimeString()}] ${msg}\n`;
        logEl.scrollTop = logEl.scrollHeight;
    }
    console.log(`[LOG] ${msg}`);
}

function setStatus(msg, type = 'ready') {
    const el = document.getElementById('status');
    if (el) {
        el.textContent = msg;
        el.className = `status-badge status-${type}`;
    }
}

async function initWasm() {
    try {
        log("Initializing WASM...");
        wasmInstance = await init();
        log("WASM Initialized successfully.");
        setStatus("Ready");
        const btnTrain = document.getElementById('btnTrain');
        const btnInit = document.getElementById('btnInit');
        if (btnTrain) btnTrain.disabled = false;
        if (btnInit) btnInit.disabled = true;
    } catch (e) {
        log(`Error initializing WASM: ${e}`);
        setStatus("Error", "busy");
    }
};

async function runTraining() {
    if (!wasmInstance) return;

    const epochs = parseInt(document.getElementById('epochs').value);
    const lr = parseFloat(document.getElementById('lr').value);
    const batchSize = parseInt(document.getElementById('batchSize').value);

    setStatus("Training...", "busy");
    document.getElementById('btnTrain').disabled = true;
    document.getElementById('btnExport').disabled = true;
    document.getElementById('btnLoad').disabled = true;

    // Use setTimeout to allow UI to render status change before heavy lifting
    setTimeout(async () => {
        try {
            log("Generating synthetic fraud data...");
            // 100 samples, 29 features
            const N = 100;
            const D = 29;
            const xData = new Float32Array(N * D);
            const yData = new Float32Array(N);

            for (let i = 0; i < N; i++) {
                // Simple logic: if feature 0 > 0.5, label = 1, else 0 (plus noise)
                let sum = 0;
                for (let j = 0; j < D; j++) {
                    const val = Math.random();
                    xData[i * D + j] = val;
                    sum += val;
                }
                // Arbitrary rule for 'fraud'
                yData[i] = sum > (D * 0.5) ? 1.0 : 0.0;
            }

            const xTrainArray = new NumRsArray(xData, new Uint32Array([N, D]));
            const yTrainArray = new NumRsArray(yData, new Uint32Array([N, 1]));

            const xTrain = new Tensor(xTrainArray, false);
            const yTrain = new Tensor(yTrainArray, false);

            log("Building Model...");
            const model = new Sequential();
            // Sequential in WASM uses specific add methods
            model.add_linear(new Linear(D, 64)); // Input -> Hidden 1
            model.add_relu(new ReLU());
            model.add_linear(new Linear(64, 32));      // Hidden 1 -> Hidden 2
            model.add_relu(new ReLU());
            model.add_linear(new Linear(32, 1));       // Hidden 2 -> Output
            model.add_sigmoid(new Sigmoid());

            log("Configuring Trainer (SGD)...");
            // Trainer(model, optimizer_name, loss_name, lr)
            const trainer = new Trainer(model, "adam", "cross_entropy", 0.01);

            const epochs = 5;
            log(`Starting training for ${epochs} epochs...`);

            // Allow UI to update
            await new Promise(r => setTimeout(r, 100));

            // fit(inputs, targets, epochs)
            const history = trainer.fit(xTrain, yTrain, epochs);

            // Check if history is array (success) or we caught error
            if (history && history.length > 0) {
                log(`Training complete. Final Loss: ${history[history.length - 1].toFixed(4)}`);
            } else {
                log("Training finished but no history returned?");
            }

            trainedModel = model;

            document.getElementById('btnTrain').disabled = false;
            document.getElementById('btnExport').disabled = false;
            setStatus("Trained", "ready");
        } catch (e) {
            log(`Training Error: ${e}`);
            console.error(e);
            setStatus("Error", "busy");
            document.getElementById('btnTrain').disabled = false;
        }
    }, 100);
};

function exportModel() {
    if (!trainedModel) return;

    try {
        log("Exporting model to ONNX (JSON)...");

        // Dummy input for trace
        const D = 29;
        const dummyData = new Float32Array(D).fill(0.5);
        const dummyArr = new NumRsArray(dummyData, new Uint32Array([1, D]));
        const dummyTensor = new Tensor(dummyArr, false);

        const jsonStr = OnnxModelWrapper.export_model_to_json(trainedModel, dummyTensor);
        log(`Export successful. JSON size: ${jsonStr.length} chars.`);
        // log(`Preview: ${jsonStr.substring(0, 100)}...`);

        savedJson = jsonStr;
        document.getElementById('btnLoad').disabled = false;
    } catch (e) {
        log(`Export Error: ${e}`);
    }
};

function loadAndInfer() {
    if (!savedJson) return;

    try {
        log("Loading model from JSON...");
        const wrapper = OnnxModelWrapper.load_from_json(savedJson);
        log("Model loaded successfully.");

        log("Running Inference...");
        const D = 29;
        // Create random input
        const inputData = new Float32Array(D);
        for (let i = 0; i < D; i++) inputData[i] = Math.random();

        const inputArr = new NumRsArray(inputData, new Uint32Array([1, D]));

        const inputNames = ["input_0"];
        const inputTensors = [inputArr]; // infer_simple takes [Array]

        const results = wrapper.infer_simple(inputNames, inputTensors);

        // results is a Map
        log("Inference Result:");
        results.forEach((val, key) => {
            log(`Output [${key}]: ${val.to_string()}`);
        });

    } catch (e) {
        log(`Inference Error: ${e}`);

        // Fallback tip if name logic failed
        if (e.toString().includes("not found")) {
            log("Tip: Check input names. Defaults are often 'input_0'.");
        }
    }
};

// Bind events after DOM content matched
document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM Loaded. Binding buttons.");
    const btnInit = document.getElementById('btnInit');
    if (btnInit) btnInit.addEventListener('click', initWasm);

    const btnTrain = document.getElementById('btnTrain');
    if (btnTrain) btnTrain.addEventListener('click', runTraining);

    const btnExport = document.getElementById('btnExport');
    if (btnExport) btnExport.addEventListener('click', exportModel);

    const btnLoad = document.getElementById('btnLoad');
    if (btnLoad) btnLoad.addEventListener('click', loadAndInfer);
});

