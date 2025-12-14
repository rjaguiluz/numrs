# @numrs/wasm

**NumRs WASM** is the WebAssembly binding for the [NumRs](https://github.com/rjaguiluz/numrs) numerical engine, bringing high-performance tensor operations and neural networks to the browser and Node.js.

## 🚀 Features

- **Zero FFI overhead**: Direct WASM calls.
- **SIMD acceleration**: Uses WebAssembly SIMD when available.
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

If you are using a bundler or Node.js, standard imports work automatically:

```javascript
import init, { Tensor } from '@numrs/wasm';

// In Node.js/Bundlers, init() might not be needed depending on setup, 
// but for WASM it's good practice to ensure initialization.
await init(); 

const t = Tensor.ones([2, 2]);
```

## 📚 Documentation

For full API documentation, please refer to the main [NumRs Repository](https://github.com/rjaguiluz/numrs).

## License

AGPL-3.0-only
