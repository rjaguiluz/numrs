import { Tensor as NativeTensor } from '../native_loader';

// Export as value (constructor)
export const Tensor = NativeTensor;
// Export as type (instance)
export type Tensor = any;

// Re-export submodules
export * as nn from './nn';
export * as optim from './optim';
