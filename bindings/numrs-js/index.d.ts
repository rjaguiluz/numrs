/**
 * NumRs JavaScript/TypeScript Bindings
 * Zero-copy Float32Array API for maximum performance
 */

export interface ArrayOptions {
  shape?: number[];
}

export interface BackendInfo {
  selected: string;
  add: string;
  mul: string;
  sum: string;
  matmul: string;
}

// Binary Operations (elementwise)
export function add(a: Float32Array, b: Float32Array, options?: ArrayOptions): Float32Array;
export function sub(a: Float32Array, b: Float32Array, options?: ArrayOptions): Float32Array;
export function mul(a: Float32Array, b: Float32Array, options?: ArrayOptions): Float32Array;
export function div(a: Float32Array, b: Float32Array, options?: ArrayOptions): Float32Array;
export function pow(a: Float32Array, b: Float32Array, options?: ArrayOptions): Float32Array;

// Unary Operations
export function neg(data: Float32Array, options?: ArrayOptions): Float32Array;
export function abs(data: Float32Array, options?: ArrayOptions): Float32Array;
export function sin(data: Float32Array, options?: ArrayOptions): Float32Array;
export function cos(data: Float32Array, options?: ArrayOptions): Float32Array;
export function tan(data: Float32Array, options?: ArrayOptions): Float32Array;
export function exp(data: Float32Array, options?: ArrayOptions): Float32Array;
export function log(data: Float32Array, options?: ArrayOptions): Float32Array;
export function sqrt(data: Float32Array, options?: ArrayOptions): Float32Array;
export function sigmoid(data: Float32Array, options?: ArrayOptions): Float32Array;
export function tanh(data: Float32Array, options?: ArrayOptions): Float32Array;
export function tanhOp(data: Float32Array, options?: ArrayOptions): Float32Array;  // Alias
export function relu(data: Float32Array, options?: ArrayOptions): Float32Array;

// Reduction Operations
export function sum(data: Float32Array, options?: ArrayOptions): Float32Array;
export function mean(data: Float32Array, options?: ArrayOptions): Float32Array;
export function max(data: Float32Array, options?: ArrayOptions): Float32Array;
export function min(data: Float32Array, options?: ArrayOptions): Float32Array;
export function variance(data: Float32Array, options?: ArrayOptions): Float32Array;

// Linear Algebra Operations
export function matmul(a: Float32Array, b: Float32Array, options?: ArrayOptions): Float32Array;
export function dot(a: Float32Array, b: Float32Array, options?: ArrayOptions): Float32Array;

// Shape Operations
export function transpose(data: Float32Array, options?: ArrayOptions): Float32Array;
export function reshape(data: Float32Array, newShape: number[], options?: ArrayOptions): Float32Array;

// Backend Information
export function startupLog(): boolean;
export function backendInfo(): BackendInfo;
