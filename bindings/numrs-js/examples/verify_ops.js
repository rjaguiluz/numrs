const { Tensor, ops, NumRsArray } = require('../dist/native_loader');

console.log("Verifying new bindings (Fixed types)...");

// 1. Array Factories
console.log("Testing zeros/ones...");
try {
    const z = NumRsArray.zeros([2, 5]);
    console.log("zeros([2,5]) shape:", z.shape);

    const o = NumRsArray.ones([3]);
    console.log("ones([3]) shape:", o.shape);
} catch (e) {
    console.log("Array factories test failed:", e);
}

// 2. Unary Ops via Tensor Class
console.log("Testing unary ops (relu, exp, log) via Tensor methods...");
try {
    // Pass Float32Array to constructor
    const data = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    const tNative = new Tensor(data, [2, 2], false);

    const t_relu = tNative.relu();
    console.log("Native Tensor.relu() success. Result shape:", t_relu.shape);
} catch (e) {
    console.log("Native Tensor method test failed:", e);
}

// 3. Standalone Ops via `ops` object
console.log("Testing standalone ops...");
try {
    const data = new Float32Array([1.0, 2.0]);
    const t = new Tensor(data, [2], false);

    // softmax
    const sm = ops.softmax(t, 0);
    console.log("ops.softmax shape:", sm.shape);

    // max
    const max_val = ops.max(t);
    // Try to access data. NativeTensor exposes `data` getter returning NumRsArray?
    // And NumRsArray exposes `data` getter returning Float32Array?
    // Let's see.
    try {
        if (max_val.data && max_val.data.data) {
            console.log("ops.max result data:", max_val.data.data);
        } else if (max_val.data) {
            console.log("ops.max result data (NumRsArray?):", max_val.data);
        } else {
            console.log("ops.max result:", max_val);
        }
    } catch (err) {
        console.log("ops.max result print error:", err);
    }

    // dot
    const dot_res = ops.dot(t, t);
    // dot usually returns Tensor with 1 element?
    // Or scalar?
    // Core dot returns Array.
    // So result is Tensor wrapping Array.
    console.log("ops.dot result shape:", dot_res.shape);

} catch (e) {
    console.log("Standalone ops test failed:", e);
}

// 4. Shape
console.log("Testing shape...");
try {
    const data = new Float32Array([1, 2, 3, 4]);
    const t = new Tensor(data, [2, 2], false);

    const t_flat = ops.flatten(t);
    console.log("ops.flatten shape:", t_flat.shape);

    const t_resh = ops.reshape(t, [4, 1]);
    console.log("ops.reshape shape:", t_resh.shape);

} catch (e) {
    console.log("Shape test failed:", e);
}

console.log("Verification finished.");
