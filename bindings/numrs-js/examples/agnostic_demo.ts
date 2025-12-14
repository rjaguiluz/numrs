import { Array as NumRsArray, Tensor, ops } from '../ts';
const { add, mul, relu } = ops;

async function run() {
    console.log("🔥 Agnostic Ops Demo\n");

    // 1. Array Ops (No Autograd)
    console.log("1. Array Operations:");
    const a1 = new NumRsArray(new Float32Array([1, 2, 3]), [3]);
    const a2 = new NumRsArray(new Float32Array([10, 20, 30]), [3]);

    // Explicitly using standalone 'add'
    const a3 = add(a1, a2);
    console.log("   add(Arr, Arr) =", a3.toString());

    // Unary
    const a4 = relu(new NumRsArray(new Float32Array([-1, 0, 1]), [3]));
    console.log("   relu(Arr) =", a4.toString());

    // 2. Tensor Ops (Autograd)
    console.log("\n2. Tensor Operations:");
    const t1 = new Tensor(a1, undefined, true);
    const t2 = new Tensor(a2, undefined, true);

    const t3 = add(t1, t2); // Should return Tensor
    console.log("   add(Tensor, Tensor) =", t3.data.toString());

    const t4 = mul(t3, t1);
    console.log("   mul(Tensor, Tensor) =", t4.data.toString());

    console.log("\n   Backward pass check:");
    t4.backward();
    console.log("   t1.grad =", t1.grad?.data.toString());

    // 3. Mixed Ops (Should Fail currently)
    console.log("\n3. Mixed Operations (Expect Error):");
    try {
        add(t1, a1);
    } catch (e: any) {
        console.log("   Caught expected error:", e.message);
    }
}

run().catch(console.error);
