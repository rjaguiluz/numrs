# 01. NumRs Arrays (JS)

## Class: `NumRsArray`
Zero-copy wrapper around logical `Float32Array`.

## Creation
```javascript
const { NumRsArray } = require('@numrs/node');

const data = new Float32Array([1, 2, 3, 4]);
const shape = [2, 2];
const arr = new NumRsArray(data, shape);
```

## Zero-Copy Note
`NumRsArray` shares memory with the Rust backend where possible to avoid copies.
