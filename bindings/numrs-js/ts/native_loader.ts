import path from 'path';
import fs from 'fs';

// --- Interfaces for Native Bindings ---

export interface NativeNumRsArray {
    readonly data: Float32Array;
    shape(): number[];
}

export interface NativeTensor {
    backward(): void;
    // Methods
    add(other: NativeTensor): NativeTensor;
    sub(other: NativeTensor): NativeTensor;
    mul(other: NativeTensor): NativeTensor;
    div(other: NativeTensor): NativeTensor;
    matmul(other: NativeTensor): NativeTensor;
    pow(exponent: number): NativeTensor;
    neg(): NativeTensor;
    abs(): NativeTensor;
    sqrt(): NativeTensor;
    sin(): NativeTensor;
    cos(): NativeTensor;
    exp(): NativeTensor;
    log(): NativeTensor;
    relu(): NativeTensor;
    sigmoid(): NativeTensor;
    mean(axis?: number): NativeTensor;
    sum(axis?: number): NativeTensor;
    mseLoss(target: NativeTensor): NativeTensor;
    flatten(startDim?: number, endDim?: number): NativeTensor;
    reshape(shape: number[]): NativeTensor;
    transpose(dim0?: number, dim1?: number): NativeTensor;

    // In-place
    assign(other: NativeTensor): void;
    zeroGrad(): void;
    backward(): void;
    detach(): NativeTensor;
    toString(): string;

    // Properties
    readonly data: any; // NumRsArray
    readonly grad: NativeTensor | undefined;
    readonly requiresGrad: boolean;
    readonly shape: number[];
}

export interface NativeOptimizer {
    step(): void;
    zeroGrad(): void;
}

export interface NativeModule {
    forward(input: NativeTensor): NativeTensor;
    parameters(): NativeTensor[];
}

export interface NativeSequential {
    addLinear(layer: any): void;
    addRelu(layer: any): void;
    addSigmoid(layer: any): void;
    addConv1D(layer: any): void; // Runtime name
    addFlatten(layer: any): void;
    addDropout(layer: any): void;
    addBatchNorm1D(layer: any): void; // Runtime name
    addUnsqueeze(dim: number): void;
    forward(input: NativeTensor): NativeTensor;
    parameters(): NativeTensor[];
    saveOnnx(input: NativeTensor, path: string): void; // Trace-based export
}

export interface NativeLinear {
    new(inFeatures: number, outFeatures: number): NativeLinear;
    forward(input: NativeTensor): NativeTensor;
    parameters(): NativeTensor[];
}

export interface NativeConv1d {
    new(inChannels: number, outChannels: number, kernelSize: number, stride?: number, padding?: number): NativeConv1d;
    forward(input: NativeTensor): NativeTensor;
}

export interface NativeFlatten {
    new(startDim: number, endDim: number): NativeFlatten;
    forward(input: NativeTensor): NativeTensor;
}

export interface NativeDropout {
    new(p: number): NativeDropout;
    forward(input: NativeTensor): NativeTensor;
}

export interface NativeBatchNorm1d {
    new(numFeatures: number): NativeBatchNorm1d;
    forward(input: NativeTensor): NativeTensor;
    train(): void;
    eval(): void;
}

// --- Specific Optimizer Constructors ---
export interface OptimizerConstructor {
    new(params: NativeTensor[], ...args: number[]): NativeOptimizer;
}

// --- Training API ---

export interface NativeMetrics {
    trainLoss: number;
    trainAcc?: number;
    valLoss?: number;
    valAcc?: number;
}

export interface NativeDataset {
    // Inputs/Targets: Array of Arrays of number
    new(inputs: number[][], targets: number[][], batchSize: number): NativeDataset;
    readonly numBatches: number;
}

export interface NativeTrainer {
    fit(trainDataset: NativeDataset, valDataset: NativeDataset | undefined | null, epochs: number, verbose?: boolean): NativeMetrics[];
}

export interface NativeTrainerBuilder {
    new(model: NativeSequential): NativeTrainerBuilder;
    learningRate(lr: number): NativeTrainerBuilder;
    build(optimizer: string, loss: string): NativeTrainer;
}

export interface NativeExports {
    // Array
    NumRsArray: any;
    Tensor: {
        new(array: any, shape?: number[], requires_grad?: boolean): NativeTensor;
        fromArray(data: number[], shape?: number[], requires_grad?: boolean): NativeTensor;
    };

    // NN
    Linear: NativeLinear;
    ReLu: any;     // Runtime name is ReLu
    Sigmoid: any;
    Conv1D: NativeConv1d; // Runtime name is Conv1D
    Flatten: NativeFlatten;
    Dropout: NativeDropout;
    BatchNorm1D: NativeBatchNorm1d; // Runtime name is BatchNorm1D
    Sequential: {
        new(): NativeSequential;
    };

    // Optimizers
    Optimizer: any;
    Sgd: any;      // Runtime name
    Adam: any;
    RmSprop: any;  // Runtime name
    Adagrad: any;
    AdamW: any;
    Lamb: any;     // Runtime name
    Lbfgs: any;    // Runtime name

    // Training
    TrainerBuilder: {
        new(model: any): NativeTrainerBuilder; // any -> NativeSequential
    };
    Dataset: {
        new(inputs: number[][], targets: number[][], batchSize: number): NativeDataset;
        fromRaw(inputs: Float32Array, inputShape: number[], targets: Float32Array, targetShape: number[], batchSize: number): NativeDataset;
    };

    // Ops - Expanded
    add: (a: any, b: any) => any;
    sub: (a: any, b: any) => any;
    mul: (a: any, b: any) => any;
    div: (a: any, b: any) => any;
    matmul: (a: any, b: any) => any;
    dot: (a: any, b: any) => any; // New

    // Elementwise
    relu: (a: any) => any;
    sigmoid: (a: any) => any;
    exp: (a: any) => any;
    log: (a: any) => any;
    sqrt: (a: any) => any;
    abs: (a: any) => any;
    cos: (a: any) => any;
    sin: (a: any) => any;
    tan: (a: any) => any;
    neg: (a: any) => any;
    acos: (a: any) => any;
    asin: (a: any) => any;
    atan: (a: any) => any;
    ceil: (a: any) => any;
    floor: (a: any) => any;
    round: (a: any) => any;
    square: (a: any) => any;
    tanh: (a: any) => any;

    // Reduction
    sum: (a: any) => any;
    mean: (a: any) => any;
    max: (a: any, axis?: number) => any;
    min: (a: any, axis?: number) => any;
    argmax: (a: any, axis?: number) => any;
    variance: (a: any, axis?: number) => any;

    // Stats
    softmax: (a: any, dim?: number) => any;
    crossEntropy: (input: any, target: any) => any;
    norm: (a: any) => any;

    // Shape
    reshape: (a: any, shape: number[]) => any;
    flatten: (a: any, start?: number, end?: number) => any;
    transpose: (a: any, d0?: number, d1?: number) => any;
    broadcastTo: (a: any, shape: number[]) => any;
    concat: (arrays: any[], axis: number) => any;

    // Misc
    version: () => string;
}

// --- Load Native Binary ---
const binaryName = 'numrs_node.node';
const rootPath = path.join(__dirname, '..', binaryName);
const releasePath = path.join(__dirname, '../target/release', binaryName);
const debugPath = path.join(__dirname, '../target/debug', binaryName);

let nativeBinding: NativeExports;

if (fs.existsSync(rootPath)) {
    nativeBinding = require(rootPath);
} else if (fs.existsSync(releasePath)) {
    nativeBinding = require(releasePath);
} else if (fs.existsSync(debugPath)) {
    nativeBinding = require(debugPath);
} else {
    throw new Error(`Could not find native binary at ${rootPath}, ${releasePath} or ${debugPath}`);
}

// --- Grouped Exports ---

export const {
    NumRsArray,
    Tensor,
    version,
    // Extract raw optimizers
    Sgd,
    Adam,
    AdamW,
    RmSprop,
    Adagrad,
    Lamb,
    Lbfgs,
    // Extract items with casing mismatches
    Conv1D: Conv1d,      // Map Conv1D (interface) -> Conv1d (variable)
    BatchNorm1D: BatchNorm1d, // Map BatchNorm1D -> BatchNorm1d
    ReLu: ReLU           // Map ReLu -> ReLU
} = nativeBinding;

export const nn = {
    Linear: nativeBinding.Linear,
    Sequential: nativeBinding.Sequential,
    ReLU: ReLU,
    Sigmoid: nativeBinding.Sigmoid,
    Conv1d: Conv1d,
    Flatten: nativeBinding.Flatten,
    Dropout: nativeBinding.Dropout,
    BatchNorm1d: BatchNorm1d
};

export const optimizers = {
    SGD: Sgd,
    Adam: Adam,
    AdamW: AdamW,
    RMSprop: RmSprop,
    Adagrad: Adagrad,
    LAMB: Lamb,
    LBFGS: Lbfgs
};

// Re-export constants for easy access
export const Linear = nativeBinding.Linear;
export const Sequential = nativeBinding.Sequential;
export const Flatten = nativeBinding.Flatten;
export const Dropout = nativeBinding.Dropout;
export const Sigmoid = nativeBinding.Sigmoid;

export const ops = {
    add: nativeBinding.add,
    sub: nativeBinding.sub,
    mul: nativeBinding.mul,
    div: nativeBinding.div,
    matmul: nativeBinding.matmul,
    dot: nativeBinding.dot,
    // Unary
    relu: nativeBinding.relu,
    sigmoid: nativeBinding.sigmoid,
    exp: nativeBinding.exp,
    log: nativeBinding.log,
    sqrt: nativeBinding.sqrt,
    abs: nativeBinding.abs,
    cos: nativeBinding.cos,
    sin: nativeBinding.sin,
    tan: nativeBinding.tan,
    neg: nativeBinding.neg,
    acos: nativeBinding.acos,
    asin: nativeBinding.asin,
    atan: nativeBinding.atan,
    ceil: nativeBinding.ceil,
    floor: nativeBinding.floor,
    round: nativeBinding.round,
    square: nativeBinding.square,
    tanh: nativeBinding.tanh,
    // Reduction
    sum: nativeBinding.sum,
    mean: nativeBinding.mean,
    max: nativeBinding.max,
    min: nativeBinding.min,
    argmax: nativeBinding.argmax,
    variance: nativeBinding.variance,
    // Stats
    softmax: nativeBinding.softmax,
    crossEntropy: nativeBinding.crossEntropy,
    norm: nativeBinding.norm,
    // Shape
    reshape: nativeBinding.reshape,
    flatten: nativeBinding.flatten,
    transpose: nativeBinding.transpose,
    broadcastTo: nativeBinding.broadcastTo,
    concat: nativeBinding.concat,
};

// Re-export individually for legacy compatibility (and ts/ops/index.ts support)
export const add = nativeBinding.add;
export const sub = nativeBinding.sub;
export const mul = nativeBinding.mul;
export const div = nativeBinding.div;
export const matmul = nativeBinding.matmul;
export const dot = nativeBinding.dot;

export const relu = nativeBinding.relu;
export const sigmoid = nativeBinding.sigmoid;
export const exp = nativeBinding.exp;
export const log = nativeBinding.log;
export const sqrt = nativeBinding.sqrt;
export const abs = nativeBinding.abs;
export const cos = nativeBinding.cos;
export const sin = nativeBinding.sin;
export const tan = nativeBinding.tan;
export const neg = nativeBinding.neg;
export const acos = nativeBinding.acos;
export const asin = nativeBinding.asin;
export const atan = nativeBinding.atan;
export const ceil = nativeBinding.ceil;
export const floor = nativeBinding.floor;
export const round = nativeBinding.round;
export const square = nativeBinding.square;
export const tanh = nativeBinding.tanh;

export const sum = nativeBinding.sum;
export const mean = nativeBinding.mean;
export const max = nativeBinding.max;
export const min = nativeBinding.min;
export const argmax = nativeBinding.argmax;
export const variance = nativeBinding.variance;

export const softmax = nativeBinding.softmax;
export const crossEntropy = nativeBinding.crossEntropy;
export const norm = nativeBinding.norm;

export const reshape = nativeBinding.reshape;
export const flatten = nativeBinding.flatten;
export const transpose = nativeBinding.transpose;
export const broadcastTo = nativeBinding.broadcastTo;
export const concat = nativeBinding.concat;

// Conv1d, BatchNorm1d, ReLU are already exported via destructuring above
// But wait, are they?
// Step 4803 replacent used `Conv1D: Conv1d` inside `export const { ... }`.
// So `Conv1d` IS exported.
// So I should remove the redundant exports here.

// Re-export optimizers (legacy name access if needed, favoring grouped)
// Re-export optimizers (legacy name access if needed, favoring grouped)
export const SGD = Sgd;

// Export Trainer classes
export const TrainerBuilder = nativeBinding.TrainerBuilder;
export const Dataset = nativeBinding.Dataset;
