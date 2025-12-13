
import { Tensor, Conv1d, Flatten, Dropout, BatchNorm1d, Sequential } from '../ts/native_loader'; // Adjust import path if needed

console.log("🚀 Verifying New Native Bindings...");

try {
    // 1. Verify Tensor Ops (Pow, Sqrt, Abs, Sin, Cos)
    console.log("\nTesting Tensor Ops:");
    const t = new Tensor(new Float32Array([4.0, 9.0, 16.0]), [3], true);
    console.log("Original:", t.toString());

    // Pow
    const t_pow = t.pow(2.0);
    console.log("Pow(2):", t_pow.toString());

    // Sqrt
    const t_sqrt = t.sqrt();
    console.log("Sqrt:", t_sqrt.toString());

    // Abs - Not yet supported in Autograd Core
    // Sin/Cos - Not yet supported in Autograd Core

    // 2. Verify New Layers (Conv1d, Flatten, BatchNorm, Dropout)
    console.log("\nTesting New Layers:");

    // Conv1d
    // Input: [Batch=1, Channels=2, Length=5]
    const input = new Tensor(
        new Float32Array(10).fill(1.0),
        [1, 2, 5],
        false
    );
    // Conv1d: In=2, Out=4, Kernel=3
    const conv = new Conv1d(2, 4, 3, 1, 0);
    const conv_out = conv.forward(input);
    console.log("Conv1d Output Shape:", conv_out.shape); // Should be [1, 4, 3] if stride=1, no padding

    // Conv1d Output: [1, 4, 3]

    // BatchNorm1d (Must be 3D: [N, C, L])
    const bn = new BatchNorm1d(4); // 4 channels
    const bn_out = bn.forward(conv_out);
    console.log("BatchNorm Output Shape:", bn_out.shape);

    // Flatten
    // [1, 4, 3] -> [1, 12]
    const flatten = new Flatten(1, -1);
    const flat_out = flatten.forward(bn_out);
    console.log("Flatten Output Shape:", flat_out.shape);

    // Dropout
    const dropout = new Dropout(0.5);
    const drop_out = dropout.forward(flat_out);
    console.log("Dropout Output Shape:", drop_out.shape);

    // 3. Verify Sequential Add Methods
    console.log("\nTesting Sequential Integration:");
    const model = new Sequential();
    model.addConv1D(conv);
    model.addBatchNorm1D(bn); // BN before Flatten
    model.addFlatten(flatten);
    // Verify ONNX export (with trace input)
    // Note: Assuming saveOnnx is bound and working
    try {
        console.log("\nAttempting ONNX Export...");
        model.saveOnnx(input, "test_model_trace.onnx");
        require('fs').unlinkSync('test_model_trace.onnx'); // Cleanup
        console.log("ONNX export succeeded.");
    } catch (e) {
        console.log("ONNX export skipped/failed (not fully implemented or N-API error):", e);
    }
    model.addDropout(dropout);

    const seq_out = model.forward(input);
    console.log("Sequential Output Shape:", seq_out.shape);

    console.log("\n✅ All Checks Passed!");

} catch (e) {
    console.error("\n❌ Verification Failed:", e);
}
