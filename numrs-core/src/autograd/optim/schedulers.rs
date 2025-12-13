//! Learning rate schedulers



/// Trait para learning rate schedulers
pub trait Scheduler {
    /// Actualiza el learning rate basado en el número de epoch
    fn step(&mut self, optimizer: &mut dyn super::base::Optimizer);
    
    /// Obtiene el learning rate actual
    fn get_lr(&self) -> f32;
}

/// Step LR Scheduler - reduce learning rate cada N epochs
/// 
/// # Ejemplo
/// ```ignore
/// let scheduler = StepLR::new(10, 0.1); // Reduce 10x cada 10 epochs
/// ```
pub struct StepLR {
    step_size: usize,
    gamma: f32,
    current_epoch: usize,
    initial_lr: f32,
}

impl StepLR {
    /// Crea un nuevo StepLR scheduler
    /// 
    /// # Argumentos
    /// - `step_size`: Número de epochs entre reducciones de learning rate
    /// - `gamma`: Factor de multiplicación (típico: 0.1)
    pub fn new(step_size: usize, gamma: f32) -> Self {
        Self {
            step_size,
            gamma,
            current_epoch: 0,
            initial_lr: 0.0, // Se inicializa en primer step
        }
    }
}

impl Scheduler for StepLR {
    fn step(&mut self, optimizer: &mut dyn super::base::Optimizer) {
        if self.current_epoch == 0 {
            self.initial_lr = optimizer.learning_rate();
        }
        
        self.current_epoch += 1;
        
        if self.current_epoch % self.step_size == 0 {
            let new_lr = optimizer.learning_rate() * self.gamma;
            optimizer.set_learning_rate(new_lr);
        }
    }
    
    fn get_lr(&self) -> f32 {
        self.initial_lr * self.gamma.powf((self.current_epoch / self.step_size) as f32)
    }
}

/// Exponential LR Scheduler - reduce learning rate exponencialmente
/// 
/// # Ejemplo
/// ```ignore
/// let scheduler = ExponentialLR::new(0.95); // Reduce 5% cada epoch
/// ```
pub struct ExponentialLR {
    gamma: f32,
    current_epoch: usize,
    initial_lr: f32,
}

impl ExponentialLR {
    /// Crea un nuevo ExponentialLR scheduler
    /// 
    /// # Argumentos
    /// - `gamma`: Factor de decay exponencial (típico: 0.95)
    pub fn new(gamma: f32) -> Self {
        Self {
            gamma,
            current_epoch: 0,
            initial_lr: 0.0,
        }
    }
}

impl Scheduler for ExponentialLR {
    fn step(&mut self, optimizer: &mut dyn super::base::Optimizer) {
        if self.current_epoch == 0 {
            self.initial_lr = optimizer.learning_rate();
        }
        
        self.current_epoch += 1;
        let new_lr = optimizer.learning_rate() * self.gamma;
        optimizer.set_learning_rate(new_lr);
    }
    
    fn get_lr(&self) -> f32 {
        self.initial_lr * self.gamma.powf(self.current_epoch as f32)
    }
}

/// Cosine Annealing LR Scheduler - learning rate sigue cosine curve
/// 
/// Reduce el learning rate siguiendo una curva coseno desde el valor inicial
/// hasta el mínimo en T_max epochs.
/// 
/// # Ejemplo
/// ```ignore
/// let scheduler = CosineAnnealingLR::new(100, 1e-6); // 100 epochs, min_lr=1e-6
/// ```
pub struct CosineAnnealingLR {
    t_max: usize,
    eta_min: f32,
    current_epoch: usize,
    initial_lr: f32,
}

impl CosineAnnealingLR {
    /// Crea un nuevo CosineAnnealingLR scheduler
    /// 
    /// # Argumentos
    /// - `t_max`: Número total de epochs para el schedule
    /// - `eta_min`: Learning rate mínimo (típico: 0.0 o 1e-6)
    pub fn new(t_max: usize, eta_min: f32) -> Self {
        Self {
            t_max,
            eta_min,
            current_epoch: 0,
            initial_lr: 0.0,
        }
    }
}

impl Scheduler for CosineAnnealingLR {
    fn step(&mut self, optimizer: &mut dyn super::base::Optimizer) {
        if self.current_epoch == 0 {
            self.initial_lr = optimizer.learning_rate();
        }
        
        self.current_epoch += 1;
        let new_lr = self.get_lr();
        optimizer.set_learning_rate(new_lr);
    }
    
    fn get_lr(&self) -> f32 {
        use std::f32::consts::PI;
        
        let t = self.current_epoch as f32;
        let t_max = self.t_max as f32;
        
        self.eta_min + (self.initial_lr - self.eta_min) * 
            (1.0 + (PI * t / t_max).cos()) / 2.0
    }
}

/// Reduce LR on Plateau - reduce cuando métrica deja de mejorar
/// 
/// Monitorea una métrica (e.g., loss) y reduce el learning rate cuando
/// la métrica no mejora por `patience` epochs.
/// 
/// # Ejemplo
/// ```ignore
/// let mut scheduler = ReduceLROnPlateau::new(0.1, 10, 1e-4);
/// 
/// for epoch in 0..100 {
///     let loss = train_epoch();
///     scheduler.step_with_metric(loss, &mut optimizer);
/// }
/// ```
pub struct ReduceLROnPlateau {
    factor: f32,
    patience: usize,
    min_lr: f32,
    
    best_metric: f32,
    epochs_since_improvement: usize,
    initial_lr: f32,
}

impl ReduceLROnPlateau {
    /// Crea un nuevo ReduceLROnPlateau scheduler
    /// 
    /// # Argumentos
    /// - `factor`: Factor de reducción cuando no hay mejora (típico: 0.1)
    /// - `patience`: Número de epochs sin mejora antes de reducir (típico: 10)
    /// - `min_lr`: Learning rate mínimo (típico: 1e-6)
    pub fn new(factor: f32, patience: usize, min_lr: f32) -> Self {
        Self {
            factor,
            patience,
            min_lr,
            best_metric: f32::INFINITY,
            epochs_since_improvement: 0,
            initial_lr: 0.0,
        }
    }
    
    /// Step con métrica (e.g., validation loss)
    /// 
    /// # Argumentos
    /// - `metric`: Valor de la métrica a monitorear (menor es mejor)
    /// - `optimizer`: Optimizer cuyo learning rate será actualizado
    pub fn step_with_metric(&mut self, metric: f32, optimizer: &mut dyn super::base::Optimizer) {
        if self.initial_lr == 0.0 {
            self.initial_lr = optimizer.learning_rate();
        }
        
        if metric < self.best_metric {
            self.best_metric = metric;
            self.epochs_since_improvement = 0;
        } else {
            self.epochs_since_improvement += 1;
            
            if self.epochs_since_improvement >= self.patience {
                let current_lr = optimizer.learning_rate();
                let new_lr = (current_lr * self.factor).max(self.min_lr);
                
                if new_lr < current_lr {
                    optimizer.set_learning_rate(new_lr);
                    self.epochs_since_improvement = 0;
                }
            }
        }
    }
}

impl Scheduler for ReduceLROnPlateau {
    fn step(&mut self, _optimizer: &mut dyn super::base::Optimizer) {
        // ReduceLROnPlateau requiere una métrica, usar step_with_metric() en su lugar
        panic!("Use step_with_metric() for ReduceLROnPlateau scheduler");
    }
    
    fn get_lr(&self) -> f32 {
        self.initial_lr
    }
}

/// Linear warmup scheduler - aumenta learning rate linealmente
/// 
/// Útil para estabilizar el training al inicio. Después del warmup,
/// mantiene el learning rate constante.
/// 
/// # Ejemplo
/// ```ignore
/// let scheduler = LinearWarmup::new(1000, 0.0); // Warmup por 1000 steps desde 0
/// ```
pub struct LinearWarmup {
    warmup_steps: usize,
    start_lr: f32,
    current_step: usize,
    target_lr: f32,
}

impl LinearWarmup {
    /// Crea un nuevo LinearWarmup scheduler
    /// 
    /// # Argumentos
    /// - `warmup_steps`: Número de steps para warmup
    /// - `start_lr`: Learning rate inicial (típico: 0.0)
    pub fn new(warmup_steps: usize, start_lr: f32) -> Self {
        Self {
            warmup_steps,
            start_lr,
            current_step: 0,
            target_lr: 0.0,
        }
    }
}

impl Scheduler for LinearWarmup {
    fn step(&mut self, optimizer: &mut dyn super::base::Optimizer) {
        if self.current_step == 0 {
            self.target_lr = optimizer.learning_rate();
            optimizer.set_learning_rate(self.start_lr);
        }
        
        self.current_step += 1;
        
        if self.current_step <= self.warmup_steps {
            let lr = self.start_lr + 
                (self.target_lr - self.start_lr) * (self.current_step as f32 / self.warmup_steps as f32);
            optimizer.set_learning_rate(lr);
        }
    }
    
    fn get_lr(&self) -> f32 {
        if self.current_step >= self.warmup_steps {
            self.target_lr
        } else {
            self.start_lr + 
                (self.target_lr - self.start_lr) * (self.current_step as f32 / self.warmup_steps as f32)
        }
    }
}
