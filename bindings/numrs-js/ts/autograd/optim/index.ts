const { optimizers } = require('../../native_loader');
import { Tensor } from '../index';

export class SGD {
    inner: any;
    constructor(params: Tensor[], lr: number, momentum: number = 0.0, weight_decay: number = 0.0) {
        this.inner = new optimizers.SGD(params, lr, momentum, weight_decay);
    }
    step(): void { this.inner.step(); }
    zeroGrad(): void { this.inner.zeroGrad(); }
}

export class Adam {
    inner: any;
    constructor(
        params: Tensor[],
        lr: number = 0.001,
        beta1: number = 0.9,
        beta2: number = 0.999,
        eps: number = 1e-8,
        weight_decay: number = 0.0
    ) {
        this.inner = new optimizers.Adam(params, lr, beta1, beta2, eps, weight_decay);
    }
    step(): void { this.inner.step(); }
    zeroGrad(): void { this.inner.zeroGrad(); }
}

export class AdamW {
    inner: any;
    constructor(
        params: Tensor[],
        lr: number = 0.001,
        beta1: number = 0.9,
        beta2: number = 0.999,
        eps: number = 1e-8,
        weight_decay: number = 0.01
    ) {
        this.inner = new optimizers.AdamW(params, lr, beta1, beta2, eps, weight_decay);
    }
    step(): void { this.inner.step(); }
    zeroGrad(): void { this.inner.zeroGrad(); }
}

export class RMSprop {
    inner: any;
    constructor(
        params: Tensor[],
        lr: number = 0.01,
        alpha: number = 0.99,
        eps: number = 1e-8,
        weight_decay: number = 0.0,
        momentum: number = 0.0
    ) {
        this.inner = new optimizers.RMSprop(params, lr, alpha, eps, weight_decay, momentum);
    }
    step(): void { this.inner.step(); }
    zeroGrad(): void { this.inner.zeroGrad(); }
}

export class AdaGrad {
    inner: any;
    constructor(params: Tensor[], lr: number = 0.01, eps: number = 1e-10, weight_decay: number = 0.0) {
        this.inner = new optimizers.AdaGrad(params, lr, eps, weight_decay);
    }
    step(): void { this.inner.step(); }
    zeroGrad(): void { this.inner.zeroGrad(); }
}

export class NAdam {
    inner: any;
    constructor(
        params: Tensor[],
        lr: number = 0.002,
        beta1: number = 0.9,
        beta2: number = 0.999,
        eps: number = 1e-8,
        weight_decay: number = 0.0
    ) {
        this.inner = new optimizers.NAdam(params, lr, beta1, beta2, eps, weight_decay);
    }
    step(): void { this.inner.step(); }
    zeroGrad(): void { this.inner.zeroGrad(); }
}

export class RAdam {
    inner: any;
    constructor(
        params: Tensor[],
        lr: number = 0.001,
        beta1: number = 0.9,
        beta2: number = 0.999,
        eps: number = 1e-8,
        weight_decay: number = 0.0
    ) {
        this.inner = new optimizers.RAdam(params, lr, beta1, beta2, eps, weight_decay);
    }
    step(): void { this.inner.step(); }
    zeroGrad(): void { this.inner.zeroGrad(); }
}

export class AdaDelta {
    inner: any;
    constructor(params: Tensor[], rho: number = 0.9, eps: number = 1e-6, weight_decay: number = 0.0) {
        this.inner = new optimizers.AdaDelta(params, rho, eps, weight_decay);
    }
    step(): void { this.inner.step(); }
    zeroGrad(): void { this.inner.zeroGrad(); }
}

export class LAMB {
    inner: any;
    constructor(
        params: Tensor[],
        lr: number = 0.001,
        beta1: number = 0.9,
        beta2: number = 0.999,
        eps: number = 1e-6,
        weight_decay: number = 0.01
    ) {
        this.inner = new optimizers.LAMB(params, lr, beta1, beta2, eps, weight_decay);
    }
    step(): void { this.inner.step(); }
    zeroGrad(): void { this.inner.zeroGrad(); }
}

export class AdaBound {
    inner: any;
    constructor(
        params: Tensor[],
        lr: number = 0.001,
        final_lr: number = 0.1,
        beta1: number = 0.9,
        beta2: number = 0.999,
        eps: number = 1e-8,
        weight_decay: number = 0.0,
        gamma: number = 1e-3
    ) {
        this.inner = new optimizers.AdaBound(params, lr, final_lr, beta1, beta2, eps, weight_decay, gamma);
    }
    step(): void { this.inner.step(); }
    zeroGrad(): void { this.inner.zeroGrad(); }
}

export class LBFGS {
    inner: any;
    constructor(params: Tensor[], lr: number = 1.0, history_size: number = 100, max_iter: number = 20) {
        this.inner = new optimizers.LBFGS(params, lr, history_size, max_iter);
    }
    step(): void { this.inner.step(); }
    zeroGrad(): void { this.inner.zeroGrad(); }
}

export class Rprop {
    inner: any;
    constructor(
        params: Tensor[],
        lr_init: number = 0.01,
        eta_plus: number = 1.2,
        eta_minus: number = 0.5,
        lr_min: number = 1e-6,
        lr_max: number = 50.0
    ) {
        this.inner = new optimizers.Rprop(params, lr_init, eta_plus, eta_minus, lr_min, lr_max);
    }
    step(): void { this.inner.step(); }
    zeroGrad(): void { this.inner.zeroGrad(); }
}
