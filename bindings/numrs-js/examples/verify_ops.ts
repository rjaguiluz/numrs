import { Tensor } from '../ts/tensor';
import { Sequential } from '../ts/nn/sequential';
import { Linear } from '../ts/nn/linear';

console.log("Verifying new bindings...");

// 1. Array Factories
console.log("Testing zeros/ones...");
// Assuming NativeTensor exposes these static methods or we need to access via Tensor wrapper?
// The bindings exported them on NumRsArray factory. 
// We need to see if TS Tensor class exposes them. 
// If not, we might fail here unless we use native directly.
// But we want to test public API.
// Let's check if we can import native module directly for verification if wrapper lacks it.
const native = require('../index.node');

try {
    const z = native.zeros([2, 5]);
    console.log("zeros([2,5]) shape:", z.shape);
    if (z.shape[0] !== 2 || z.shape[1] !== 5) throw new Error("zeros shape mismatch");

    const o = native.ones([3]);
    console.log("ones([3]) shape:", o.shape);
} catch (e) {
    console.log("Array factories test failed (or not exposed in index.node?):", e);
}

// 2. Unary Ops
console.log("Testing unary ops (relu, exp, log)...");
const t1 = new Tensor([[-1.0, 2.0], [0.0, -5.0]], [2, 2]);
const t_relu = t1.relu();
console.log("ReLU output:", t_relu.data); // Should be [0, 2, 0, 0]

const t2 = new Tensor([1.0, 2.0], [2]);
const t_exp = t2.exp();
console.log("Exp output:", t_exp.data);

// 3. Stats
console.log("Testing stats (softmax, cross_entropy)...");
const logits = new Tensor([[1.0, 2.0, 3.0]], [1, 3]);
const sm = native.softmax(logits.native, 1);
console.log("Softmax output shape:", sm.shape);

// 4. Reduction
console.log("Testing reduction (argmax, sum, mean)...");
const t3 = new Tensor([1, 5, 2], [3]);
const sum_val = t3.sum();
console.log("Sum output:", sum_val.data); // Should be [8] around 8

const max_val = native.max(t3.native);
console.log("Max output (native):", max_val.data); // Should be [5]

// 5. Shape
console.log("Testing shape (reshape, flatten)...");
const t4 = new Tensor([1, 2, 3, 4], [2, 2]);
const t_flat = t4.flatten();
console.log("Flatten shape:", t_flat.shape); // [4]

const t_reshape = t4.reshape([4, 1]);
console.log("Reshape shape:", t_reshape.shape); // [4, 1]

// 6. Dot
console.log("Testing dot...");
const v1 = new Tensor([1, 2, 3], [3]);
const v2 = new Tensor([4, 5, 6], [3]);
const dot_res = native.dot(v1.native, v2.native);
console.log("Dot result:", dot_res.data); // 4+10+18 = 32

console.log("Verification finished.");
