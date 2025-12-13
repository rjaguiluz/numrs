// Import NumRs WASM with cache busting
import init, {
    NumRsArray,
    Tensor,
    Sequential,
    Linear,
    ReLU,
    Sigmoid,
    Softmax,
    Conv1d,
    BatchNorm1d,
    Dropout,
    Flatten,
    Trainer,
    OnnxModelWrapper,
    init_webgpu,
    init_backend,
    backend_info,
    sum, max, min, argmax, mean, variance,
    dot, matmul, concat
} from '../../pkg/numrs_wasm.js?v=v13';

// Modificar URL para cache busting
const wasmUrl = `pkg/numrs_wasm_bg.wasm?v=v13`;

const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const runBtn = document.getElementById('run-btn');
const editor = document.getElementById('code-editor');
const consoleOutput = document.getElementById('console-output');

// Initial State
let isWasmReady = false;

// --- UTILS ---

/**
 * Creates a Tensor from a flat array and shape.
 * Usage: tensor([1, 2, 3, 4], [2, 2])
 * @param {number[]} data 
 * @param {number[]} shape 
 * @param {boolean} requiresGrad 
 * @returns {Tensor}
 */
function tensor(data, shape, requiresGrad = false) {
    if (!shape) {
        shape = [data.length];
    }
    const d = new Float32Array(data);
    const s = new Uint32Array(shape);
    const arr = new NumRsArray(d, s);
    return new Tensor(arr, requiresGrad);
}

/**
 * Pretty prints a Tensor or Array
 */
function print(obj, label = "") {
    if (label) log(label, 'info');

    if (obj instanceof Tensor || obj instanceof NumRsArray) {
        try {
            const shapeStr = Array.from(obj.shape).join('x');
            // For Tensor, .data returns a NumRsArray, so we need .data.data
            // For NumRsArray, .data returns the raw data directly
            let rawData = obj.data;
            if (rawData instanceof NumRsArray || (rawData && typeof rawData.data !== 'undefined')) {
                rawData = rawData.data;
            }
            const data = Array.from(rawData);
            log(`Tensor/Array [${shapeStr}]:`, 'result');

            // Basic formatting for 1D/2D
            if (obj.shape.length === 1) {
                log(`[${data.map(n => n.toFixed(4)).join(', ')}]`, 'result');
            } else if (obj.shape.length === 2) {
                const cols = obj.shape[1];
                let out = "[\n";
                for (let i = 0; i < obj.shape[0]; i++) {
                    const row = data.slice(i * cols, (i + 1) * cols);
                    out += "  [" + row.map(n => n.toFixed(4)).join(', ') + "]";
                    if (i < obj.shape[0] - 1) out += ",\n";
                }
                out += "\n]";
                log(out, 'result');
            } else {
                log(`[Flat Data]: ${data.map(n => n.toFixed(4)).join(', ')}`, 'result');
            }
        } catch (e) {
            log(`Error printing tensor: ${e}`, 'error');
        }
    } else {
        log(obj);
    }
}


// --- SNIPPETS ---

const snippets = {
    basic_tensor: `// Basic Tensor Creation
// Helper: tensor(data, shape, requiresGrad)

const t1 = tensor([1, 2, 3, 4], [2, 2]);
print(t1, "Tensor 1 (2x2):");

const t2 = tensor([10, 20, 30, 40], [2, 2]);
print(t2, "Tensor 2 (2x2):");

// Basic Math (Manual loop for now, or exposed ops)
// Note: Elementwise 'add', 'sub' etc are built-in for Tensor in Core
// but might need explicit export or method usage in WASM.
// For now, let's verify data access works:
console.log("Verified data access via print()!");
`,

    matmul: `// Matrix Multiplication
// A = Identity [2, 2]
const A = new NumRsArray(new Float32Array([1, 0, 0, 1]), new Uint32Array([2, 2]));

// B = [1, 2, 3, 4]
const B = new NumRsArray(new Float32Array([1, 2, 3, 4]), new Uint32Array([2, 2]));

console.log("Computing Matrix Multiplication...");
const C = matmul(A, B);

print(C, "Result (A @ B):");
`,

    reduction: `// Reductions
const t = tensor([1, 5, -2, 8, 3, 0], [2, 3]);
print(t, "Input Tensor:");

// Accessing inner array for ops
const arr = t.inner.borrow(); // Internal access if needed, or use helpers on wrappers if exposed

// Note: sum/max exposed as global functions acting on NumRsArray
// We extract the underlying array from Tensor wrapper for these ops 
// (Assuming helper extracts it, or we create NumRsArray directly)

const rawArray = new NumRsArray(new Float32Array([1, 5, -2, 8, 3, 0]), new Uint32Array([2, 3]));
print(rawArray, "Raw Array:");

const total = sum(rawArray);
print(total, "Sum (all):");

const maxVal = max(rawArray);
print(maxVal, "Max (all):");
`,

    linear_layer: `// Linear Layer Forward
const batch = 2;
const input_dim = 3;
const output = 2;

const x = tensor(
  [0.1, 0.2, 0.3,  0.4, 0.5, 0.6], 
  [2, 3]
);
print(x, "Input X:");

const layer = new Linear(input_dim, output);
console.log("Linear Layer initialized.");

const y = layer.forward(x);
print(y, "Output Y (Forward Pass):");
`,

    full_nn: `// Training Loop
// Data: XOR Problem
const xTrain = tensor([0,0, 0,1, 1,0, 1,1], [4, 2]);
const yTrain = tensor([0,   1,   1,   0],   [4, 1]);

print(xTrain, "Training Data X:");
print(yTrain, "Labels Y:");

// Model
const model = new Sequential();
model.add_linear(new Linear(2, 4));
model.add_relu(new ReLU());
model.add_linear(new Linear(4, 1));
model.add_sigmoid(new Sigmoid());

// Train
// Trainer(model, optimizer, loss, lr)
const trainer = new Trainer(model, "sgd", "mse", 0.5);

console.log("Training for 50 epochs...");
const losses = trainer.fit(xTrain, yTrain, 50);

console.log("Final Loss:", losses[losses.length-1].toFixed(5));

// Inference
const pred = model.forward(xTrain);
print(pred, "Predictions:");
`,

    fraud_detection: `// 🚨 Fraud Detection Example
// Features: [Amount, Hour, RiskScore]
// Label: 1.0 (Fraud) | 0.0 (Legit)

// --- Generate Synthetic Data ---
function generateFraudData(n) {
    const x = [];
    const y = [];
    
    for (let i = 0; i < n; i++) {
        const amount = Math.random();  // Normalized: $0-$10,000
        const hour = Math.random();    // Normalized: 00:00-23:59
        const risk = Math.random();    // External risk score
        
        // Business logic: Fraud if (risk > 0.8 && amount > 0.5) || risk > 0.95
        const isFraud = (risk > 0.8 && amount > 0.5) || (risk > 0.95);
        
        x.push(amount, hour, risk);
        y.push(isFraud ? 1.0 : 0.0);
    }
    return { x, y };
}

console.log("🚨 NumRs WASM: Fraud Detection Demo");
console.log("════════════════════════════════════\\n");

// Step 1: Generate Training Data
console.log("📊 Step 1: Generating synthetic fraud data...");
const trainData = generateFraudData(100);  // 100 samples for demo
const testData = generateFraudData(20);

const xTrain = tensor(trainData.x, [100, 3]);
const yTrain = tensor(trainData.y, [100, 1]);
const xTest = tensor(testData.x, [20, 3]);
const yTest = tensor(testData.y, [20, 1]);

console.log("   Train samples:", 100);
console.log("   Test samples:", 20);

// Step 2: Define MLP Model
console.log("\\n🧠 Step 2: Building MLP...");
console.log("   Architecture: Linear(3→16) → ReLU → Linear(16→8) → ReLU → Linear(8→1) → Sigmoid");

const model = new Sequential();
model.add_linear(new Linear(3, 16));
model.add_relu(new ReLU());
model.add_linear(new Linear(16, 8));
model.add_relu(new ReLU());
model.add_linear(new Linear(8, 1));
model.add_sigmoid(new Sigmoid());

// Step 3: Train
console.log("\\n🏋️ Step 3: Training with Adam optimizer...");
const trainer = new Trainer(model, "adam", "mse", 0.1);

const epochs = 50;
const losses = trainer.fit(xTrain, yTrain, epochs);

console.log("   Final Loss:", losses[losses.length-1].toFixed(5));

// Step 4: Test Predictions
console.log("\\n🔍 Step 4: Running predictions on test cases...");

const testCases = [
    { amount: 0.1, hour: 0.5, risk: 0.1, expected: "Legit" },   // Low risk
    { amount: 0.9, hour: 0.2, risk: 0.96, expected: "Fraud" },  // High risk > 0.95
    { amount: 0.6, hour: 0.8, risk: 0.85, expected: "Fraud" },  // High risk + high amount
    { amount: 0.2, hour: 0.9, risk: 0.85, expected: "Legit" },  // High risk but low amount
];

console.log("\\n   | Amount | Hour  | Risk  | Prob   | Pred   | Match |");
console.log("   |--------|-------|-------|--------|--------|-------|");

for (const tc of testCases) {
    const input = tensor([tc.amount, tc.hour, tc.risk], [1, 3]);
    const output = model.forward(input);
    
    // Get probability from output
    let rawData = output.data;
    if (rawData && rawData.data) rawData = rawData.data;
    const prob = Array.from(rawData)[0];
    
    const pred = prob > 0.5 ? "Fraud" : "Legit";
    const match = pred === tc.expected ? "✅" : "❌";
    
    console.log(\`   | \${tc.amount.toFixed(2)}   | \${tc.hour.toFixed(2)}  | \${tc.risk.toFixed(2)}  | \${prob.toFixed(4)} | \${pred.padEnd(6)} | \${match}    |\`);
}

console.log("\\n✅ Fraud Detection demo complete!");
`,

    time_series: `// 📈 Time Series Forecasting with 1D CNN
// Architecture matching numrs-core/examples/timeseries_cnn.rs
// Using async training for real-time progress updates

// --- Generate Synthetic Time Series (Noisy Sine Wave) ---
function generateTimeSeries(numSamples, seqLen, offset = 0) {
    const inputs = [];
    const targets = [];
    const totalPoints = numSamples + seqLen;
    const rawData = [];
    
    for (let k = 0; k < totalPoints; k++) {
        const t = offset + k;
        let val = Math.sin(t * 0.1) * 10.0;
        val += (((k * 12345 + 6789) % 100) / 20.0) - 2.5;
        rawData.push(val);
    }
    
    const center = 0.0, scale = 12.5;
    
    for (let i = 0; i < numSamples; i++) {
        const window = [];
        for (let j = 0; j < seqLen; j++) {
            window.push((rawData[i + j] - center) / scale);
        }
        inputs.push(...window);
        targets.push((rawData[i + seqLen] - center) / scale);
    }
    
    return { inputs, targets, scale, center };
}

console.log("📈 NumRs WASM: Time Series CNN Forecasting");
console.log("═══════════════════════════════════════════════════════════\\n");

// Hyperparameters (optimized for browser responsiveness)
const seqLen = 20;       // Sequence length (20 steps look-back)
const numTrain = 100;    // Training samples (reduced for speed)
const numTest = 10;      // Test samples  
const epochs = 20;       // Training epochs
const lr = 0.01;         // Learning rate

// Step 1: Generate Data
console.log("📊 Step 1: Generating synthetic time series (Noisy Sine Wave)");
const trainData = generateTimeSeries(numTrain, seqLen, 0);
const testData = generateTimeSeries(numTest, seqLen, 200);

const xTrain = tensor(trainData.inputs, [numTrain, seqLen]);
const yTrain = tensor(trainData.targets, [numTrain, 1]);

console.log("   Sequence length:", seqLen);
console.log("   Train samples:", numTrain);
console.log("   Test samples:", numTest);

// Step 2: Define CNN Model
console.log("\\n🧠 Step 2: Building CNN Architecture");
console.log("   Unsqueeze → Conv1d(1→8) → BN → ReLU → Conv1d(8→16) → BN → ReLU → Flatten → Linear");

const model = new Sequential();

// Reshape: [B, SeqLen] -> [B, 1, SeqLen]
model.add_unsqueeze(1);

// Block 1: Conv(1 → 8) | Kernel=3
model.add_conv1d(new Conv1d(1, 8, 3));       // [B, 8, 18]
model.add_batch_norm1d(new BatchNorm1d(8));
model.add_relu(new ReLU());

// Block 2: Conv(8 → 16) | Kernel=3  
model.add_conv1d(new Conv1d(8, 16, 3));      // [B, 16, 16]
model.add_batch_norm1d(new BatchNorm1d(16));
model.add_relu(new ReLU());

// Output: (20-2) -> 18 -> (18-2) -> 16
const outputLen = seqLen - 4;
const flatFeatures = 16 * outputLen; // 16 * 16 = 256

// Head
model.add_flatten(new Flatten(1, 2));        // [B, 256]
model.add_linear(new Linear(flatFeatures, 32));
model.add_relu(new ReLU());
model.add_linear(new Linear(32, 1));

// Step 3: Train with real-time progress
console.log("\\n🏋️ Step 3: Training with Adam optimizer...");
const trainer = new Trainer(model, "adam", "mse", lr);

console.log("   Starting", epochs, "epochs (real-time progress):\\n");

// Capture console.log before async (otherwise it escapes to real console)
const log = console.log.bind(console);

// Async training for responsive UI
async function trainAsync() {
    for (let epoch = 0; epoch < epochs; epoch++) {
        const loss = trainer.train_step(xTrain, yTrain);
        
        // Log every 5 epochs or first/last
        if (epoch === 0 || epoch === epochs - 1 || (epoch + 1) % 5 === 0) {
            log(\`   Epoch \${(epoch + 1).toString().padStart(2)}/\${epochs}: Loss = \${loss.toFixed(6)}\`);
        }
        
        // Yield to browser for UI update
        await new Promise(r => setTimeout(r, 0));
    }
    
    // Step 4: Test Predictions
    log("\\n🔍 Step 4: Verifying predictions (first 5 examples)");
    log("   (De-normalized values)\\n");
    log("   | Sample |  Pred  | Target |  Diff |");
    log("   |--------|--------|--------|-------|");
    
    for (let i = 0; i < 5; i++) {
        const startIdx = i * seqLen;
        const window = testData.inputs.slice(startIdx, startIdx + seqLen);
        const target = testData.targets[i];
        
        const input = tensor(window, [1, seqLen]);
        const output = model.forward(input);
        
        let rawOutput = output.data;
        if (rawOutput && rawOutput.data) rawOutput = rawOutput.data;
        const pred = Array.from(rawOutput)[0];
        
        const predReal = (pred * testData.scale).toFixed(2);
        const targetReal = (target * testData.scale).toFixed(2);
        const diff = Math.abs(pred * testData.scale - target * testData.scale).toFixed(2);
        
        log(\`   |   \${i+1}    | \${predReal.padStart(6)} | \${targetReal.padStart(6)} | \${diff.padStart(5)} |\`);
    }
    
    log("\\n✅ Time Series CNN demo complete!");
}

trainAsync();
`
};

// Console Logging Helper
function log(msg, type = 'info') {
    const el = document.createElement('div');
    el.className = `log-entry log-${type}`;

    // Timestamp
    const ts = document.createElement('span');
    ts.className = 'timestamp';
    ts.textContent = `[${new Date().toLocaleTimeString()}]`;
    el.appendChild(ts);

    // Content
    const content = document.createElement('span');
    if (typeof msg === 'string') {
        content.textContent = msg; // Text
    } else {
        try {
            if (typeof msg === 'object') {
                // Simple object dump
                content.textContent = JSON.stringify(msg, (k, v) => k.startsWith('__') ? undefined : v, 2);
                content.style.whiteSpace = "pre";
            } else {
                content.textContent = String(msg);
            }
        } catch (e) {
            content.textContent = String(msg);
        }
    }

    el.appendChild(content);
    consoleOutput.appendChild(el);
    consoleOutput.scrollTop = consoleOutput.scrollHeight;
}

// Redirect real console methods
const originalLog = console.log;
const originalError = console.error;
const originalWarn = console.warn;

function hijackConsole() {
    console.log = (...args) => {
        args.forEach(arg => log(arg, 'info'));
    };
    console.error = (...args) => {
        args.forEach(arg => log(arg, 'error'));
    };
    console.warn = (...args) => {
        args.forEach(arg => log(arg, 'warn'));
    };
}

function restoreConsole() {
    console.log = originalLog;
    console.error = originalError;
    console.warn = originalWarn;
}

// Initialization
async function initialize() {
    try {
        await init();

        // Expose Globals
        window.NumRsArray = NumRsArray;
        window.Tensor = Tensor;
        window.Sequential = Sequential;
        window.Linear = Linear;
        window.ReLU = ReLU;
        window.Sigmoid = Sigmoid;
        window.Softmax = Softmax;
        window.Conv1d = Conv1d;
        window.BatchNorm1d = BatchNorm1d;
        window.Dropout = Dropout;
        window.Flatten = Flatten;
        window.Trainer = Trainer;
        window.OnnxModelWrapper = OnnxModelWrapper;

        // Ops
        window.sum = sum;
        window.max = max;
        window.min = min;
        window.dot = dot;

        // Helpers
        window.tensor = tensor;
        window.print = print;

        // Try WebGPU init
        try {
            log(`Attempting to initialize WebGPU... (navigator.gpu: ${navigator.gpu ? 'AVAILABLE' : 'UNDEFINED'})`, "info");
            if (!navigator.gpu) {
                throw new Error("WebGPU is not supported in this browser.");
            }
            await init_webgpu();
            // init_backend(); // DISABLED: WebGPU backend in numrs uses blocking calls (mpsc::recv) which panic in WASM.
            // backend_info() will report CPU-SIMD, which is safe.
            log("WebGPU Init: Supported but disabled (Sync/Async mismatch). Using CPU-SIMD.", "warn");
            log("Core Architecture requires async refactor for WebGPU in WASM.", "info");
        } catch (gpuErr) {
            log(`WebGPU Init Failed (Using CPU): ${gpuErr}`, "warn");
        }

        // Check active backend
        const infoStr = backend_info();
        try {
            const info = JSON.parse(infoStr);
            log(`Active Backend: ${info.selected.toUpperCase()}`, "info");
            log(`Dispatch Table: Add=${info.add}, Matmul=${info.matmul}`, "info");
        } catch (e) {
            log(`Backend Info: ${infoStr}`, "info");
        }

        isWasmReady = true;
        statusText.textContent = "WASM Ready";
        statusDot.className = "status-dot ready";
        runBtn.disabled = false;

        log("WASM Framework Ready. Helpers: tensor(), print()", "success");

        // Load default snippet
        editor.value = snippets.basic_tensor;

    } catch (e) {
        statusText.textContent = "Init Failed";
        statusDot.className = "status-dot error";
        log(`Initialization Error: ${e}`, "error");
    }
}

// Execution
async function runCode() {
    if (!isWasmReady) return;

    const code = editor.value;
    if (!code.trim()) return;

    statusText.textContent = "Running...";
    statusDot.className = "status-dot busy";
    runBtn.disabled = true;

    hijackConsole();

    try {
        const wrappedCode = `return (async () => { 
            ${code} 
        })()`;

        const func = new Function(wrappedCode);
        await func();

        log("Execution Complete", "success");
    } catch (e) {
        log(e, "error");
    } finally {
        restoreConsole();
        statusText.textContent = "WASM Ready";
        statusDot.className = "status-dot ready";
        runBtn.disabled = false;
    }
}

// UI Bindings
runBtn.addEventListener('click', runCode);

editor.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        runCode();
    }
});

document.getElementById('clear-btn').addEventListener('click', () => {
    consoleOutput.innerHTML = '';
});

// Sidebar Examples
document.querySelectorAll('.example-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const key = btn.dataset.snippet;
        if (snippets[key]) {
            editor.value = snippets[key];
        }
    });
});

// Start
initialize();
