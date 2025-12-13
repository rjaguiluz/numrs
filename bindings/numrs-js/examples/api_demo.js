const numrs = require('../index.js');

async function run() {
    console.log(`NumRs Node.js Version: ${numrs.version()}`);

    // 1. Create NumRsArray
    console.log("\n--- NumRsArray ---");
    const data = new Float32Array([1.0, 2.0, 3.0]);
    const arr = new numrs.NumRsArray(data, [3]);
    console.log(`Array created: ${arr.toString()}`);

    const arr2 = new numrs.NumRsArray(new Float32Array([10.0, 20.0, 30.0]), [3]);
    const res = arr.add(arr2);
    console.log(`Array Add: ${res.toString()}`);

    // 2. Create Tensor
    console.log("\n--- Tensor & Autograd ---");
    const t1 = new numrs.Tensor(arr, [3], true); // requires_grad=true
    console.log(`Tensor t1: ${t1.toString()}`);

    const t2 = new numrs.Tensor(arr2, [3], false);
    console.log(`Tensor t2: ${t2.toString()}`);

    // 3. Ops
    // z = t1 * t2 + t1
    const t3 = t1.mul(t2);
    const z = t3.add(t1);
    console.log(`z = t1 * t2 + t1: ${z.toString()}`);

    // 4. Backward
    z.backward();
    console.log("Backward pass executed.");

    const grad = t1.grad;
    if (grad) {
        console.log(`t1.grad: ${grad.toString()}`);
        // Expected: t2 + 1 = [10, 20, 30] + 1 = [11, 21, 31]
    } else {
        console.log("t1.grad is missing!");
    }
}

run().catch(console.error);
