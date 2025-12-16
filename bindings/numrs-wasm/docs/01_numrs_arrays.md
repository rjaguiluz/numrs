# 01. NumRs Arrays (WASM)

## Class: `NumRsArray`
WASM-side handle to data.

## Creation
```javascript
import { NumRsArray } from '@numrs/wasm';

const data = new Float32Array([1, 2, 3, 4]);
const shape = new Uint32Array([2, 2]); // MUST be Uint32Array
const arr = new NumRsArray(data, shape);
```

## Setup
Remember to wait for initialization!
```javascript
import init from '@numrs/wasm';
await init();
```
