import { Tensor } from '../autograd';
import { Array } from '../array';

export interface Batch {
    inputs: Tensor;
    targets: Tensor;
}

export class Dataset {
    inputs: any[]; // Assuming raw JS arrays for now as per previous logic
    targets: any[];
    batchSize: number;

    constructor(inputs: any[], targets: any[], batchSize: number) {
        this.inputs = inputs;
        this.targets = targets;
        this.batchSize = batchSize;
    }

    *batches(): Generator<Batch> {
        const n = this.inputs.length;
        for (let i = 0; i < n; i += this.batchSize) {
            const end = Math.min(i + this.batchSize, n);
            const batchX = this.inputs.slice(i, end);
            const batchY = this.targets.slice(i, end);

            // Flatten logic depends on input shape. 
            // Assuming 2D arrays (list of lists)
            const dimX = batchX[0].length;
            const dimY = batchY[0].length;

            const flattenX = batchX.flat();
            const flattenY = batchY.flat();

            const bx = new Float32Array(flattenX);
            const by = new Float32Array(flattenY);

            const tx = new Tensor(new Array(bx, [batchX.length, dimX]), [batchX.length, dimX], false);
            const ty = new Tensor(new Array(by, [batchY.length, dimY]), [batchY.length, dimY], false);

            yield { inputs: tx, targets: ty };
        }
    }
}
