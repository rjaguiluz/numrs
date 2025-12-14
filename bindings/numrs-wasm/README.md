# @numrs/wasm

**NumRs WASM** is the WebAssembly binding for the [NumRs](https://github.com/rjaguiluz/numrs) numerical engine, bringing high-performance tensor operations and deep learning to the browser and Node.js.

## 🚀 Features

- **Zero FFI overhead**: Direct WASM calls.
- **Deep Learning**: Full support for Autograd, Neural Networks, and Optimizers.
- **Universal**: Works in Browser (ESM), Node.js, Deno, and Bundlers.
- **TypeScript**: Full type definitions included.

## 📦 Installation

```bash
npm install @numrs/wasm
```

## 🎯 Usage

### 🌐 Browser (No Bundler)

To use `@numrs/wasm` directly in the browser without a bundler (like Vite/Webpack), you must use an **Import Map** to point to the `pkg-web` (ES Modules) build.

Add this to your HTML `<head>`:

```html
<script type="importmap">
{
    "imports": {
        "@numrs/wasm": "./node_modules/@numrs/wasm/pkg-web/numrs_wasm.js"
    }
}
</script>

<script type="module">
    import init, { Tensor, nn } from '@numrs/wasm';

    async function run() {
        await init(); // Initialize the WASM module
        
        console.log("NumRs WASM loaded!");
        
        // Create a tensor
        let x = Tensor.randn([10, 5]);
        
        // Define a model
        let model = new nn.Sequential();
        model.add_linear(new nn.Linear(5, 10));
        model.add_relu(new nn.ReLU());
        model.add_linear(new nn.Linear(10, 2));

        // Forward pass
        let output = model.forward(x);
        console.log("Output shape:", output.shape());
    }

    run();
</script>
```

### 📦 Bundlers (Vite, Webpack) & Node.js

```javascript
import init, { Tensor, Sequential, Linear, ReLU, Trainer } from '@numrs/wasm';

// Initialize WASM
await init(); 

// 1. Define Model
const model = new Sequential();
model.add_linear(new Linear(10, 32));
model.add_relu(new ReLU());
model.add_linear(new Linear(32, 1));

// 2. Training Loop
const trainer = new Trainer(model, "adam", "mse", 0.01);
// Assuming xTrain and yTrain are Tensors
trainer.fit(xTrain, yTrain, 10);
```

### 🧠 Optimizers & Loss Functions

NumRs supports a wide range of optimizers and loss functions for training:

| Context        | Supported Values                                                                                                                         |
| :------------- | :--------------------------------------------------------------------------------------------------------------------------------------- |
| **Optimizers** | `"sgd"`, `"adam"`, `"adamw"`, `"nadam"`, `"radam"`, `"rmsprop"`, `"adagrad"`, `"adadelta"`, `"lamb"`, `"adabound"`, `"lbfgs"`, `"rprop"` |
| **Losses**     | `"mse"` (Regression), `"cross_entropy"` (Classification)                                                                                 |

### 🔍 Advanced Topics

#### Time Series (1D CNN)
Use `Conv1d` for sequence processing. input shape should be `[Batch, Channels, Length]`.
```javascript
model.add_conv1d(new nn.Conv1d(in_channels, out_channels, kernel_size));
model.add_relu(new nn.ReLU());
model.add_flatten(new nn.Flatten(1, -1));
model.add_linear(new nn.Linear(hidden_size, output_size));
```

#### ONNX Support
NumRs WASM can export models to JSON representation compatible with web inference.
- **Export**: `OnnxModelWrapper.export_model_to_json(...)`
- **Inference**: `OnnxModelWrapper.load_from_json(...)`

## 📚 Documentation

For full API documentation, please refer to the main [NumRs Repository](https://github.com/rjaguiluz/numrs) or the detailed docs in `DOCS.md`.

## License

AGPL-3.0-only
