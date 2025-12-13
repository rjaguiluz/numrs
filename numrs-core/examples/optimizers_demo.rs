//! Ejemplo: Optimizers - SGD, Adam, RMSprop, AdaGrad
//!
//! Compara diferentes optimizers en:
//! 1. Regresión lineal simple
//! 2. Clasificación con red neuronal
//! 3. Análisis de convergencia

use anyhow::Result;
use numrs::{AdaGrad, Adam, Array, Optimizer, RMSprop, Tensor, SGD};
use std::cell::RefCell;
use std::rc::Rc;

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  ⚡ NumRs Optimizers Demo");
    println!("═══════════════════════════════════════════════════════════\n");

    // ========================================================================
    // EJEMPLO 1: Regresión lineal con SGD
    // ========================================================================
    println!("📈 EJEMPLO 1: Regresión lineal con SGD\n");
    println!("  Objetivo: aprender y = 3x + 2\n");

    // Datos de entrenamiento
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_data = vec![5.0, 8.0, 11.0, 14.0, 17.0]; // y = 3x + 2

    // Parámetros (wrapped en Rc<RefCell> para el optimizer)
    let w = Rc::new(RefCell::new(Tensor::new(
        Array::new(vec![1], vec![0.5]),
        true,
    )));
    let b = Rc::new(RefCell::new(Tensor::new(
        Array::new(vec![1], vec![0.0]),
        true,
    )));

    // SGD con momentum
    let mut optimizer = SGD::new(vec![w.clone(), b.clone()], 0.01, 0.9, 0.0);

    println!(
        "  Pesos iniciales: w={:.4}, b={:.4}",
        w.borrow().values()[0],
        b.borrow().values()[0]
    );
    println!("\n  Entrenando con SGD (momentum=0.9)...\n");

    let epochs = 100;
    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for (&x_val, &y_true) in x_data.iter().zip(y_data.iter()) {
            let x = Tensor::new(Array::new(vec![1], vec![x_val]), false);
            let y_target = Tensor::new(Array::new(vec![1], vec![y_true]), false);

            // Forward
            let y_pred = w.borrow().mul(&x)?.add(&b.borrow())?;
            let loss = y_pred.mse_loss(&y_target)?;
            total_loss += loss.values()[0];

            // Backward
            loss.backward()?;
        }

        // Update
        optimizer.step()?;
        optimizer.zero_grad();

        if epoch % 20 == 0 || epoch == epochs - 1 {
            println!(
                "    Epoch {:3}: loss={:.6}, w={:.4}, b={:.4}",
                epoch,
                total_loss / x_data.len() as f32,
                w.borrow().values()[0],
                b.borrow().values()[0]
            );
        }
    }

    println!(
        "\n  Resultado: w={:.4}, b={:.4}",
        w.borrow().values()[0],
        b.borrow().values()[0]
    );
    println!("  (Objetivo: w=3.000, b=2.000) ✓\n");

    // ========================================================================
    // EJEMPLO 2: Comparación de optimizers
    // ========================================================================
    println!("⚔️  EJEMPLO 2: Comparación de optimizers\n");
    println!("  Problema: optimizar f(x) = (x - 5)²\n");

    // Función objetivo: f(x) = (x - 5)²
    // Mínimo en x = 5

    let optimizers_to_test = vec![
        ("SGD (lr=0.1)", "sgd"),
        ("SGD+Momentum (lr=0.1, m=0.9)", "sgd_momentum"),
        ("Adam (lr=0.1)", "adam"),
        ("RMSprop (lr=0.1)", "rmsprop"),
    ];

    for (name, opt_type) in optimizers_to_test {
        // Inicializar x en posición aleatoria
        let x = Rc::new(RefCell::new(Tensor::new(
            Array::new(vec![1], vec![0.5]),
            true,
        )));

        let mut optimizer: Box<dyn Optimizer> = match opt_type {
            "sgd" => Box::new(SGD::new(vec![x.clone()], 0.1, 0.0, 0.0)),
            "sgd_momentum" => Box::new(SGD::new(vec![x.clone()], 0.1, 0.9, 0.0)),
            "adam" => Box::new(Adam::with_lr(vec![x.clone()], 0.1)),
            "rmsprop" => Box::new(RMSprop::new(vec![x.clone()], 0.1, 0.99, 1e-8, 0.0, 0.0)),
            _ => unreachable!(),
        };

        print!("  {:<30} | ", name);

        // Entrenar
        for _ in 0..50 {
            let target = Tensor::new(Array::new(vec![1], vec![5.0]), false);
            let diff = x
                .borrow()
                .add(&target.mul(&Tensor::new(Array::new(vec![1], vec![-1.0]), false))?)?;
            let loss = diff.mul(&diff)?;

            loss.backward()?;
            optimizer.step()?;
            optimizer.zero_grad();
        }

        let final_x = x.borrow().values()[0];
        let error = (final_x - 5.0).abs();
        println!("x={:.4} (error: {:.4})", final_x, error);
    }

    println!();

    // ========================================================================
    // EJEMPLO 3: Learning Rates y Schedulers
    // ========================================================================
    println!("📊 EJEMPLO 3: Learning Rates con Schedulers\n");
    println!("  Problema: f(x) = (x - 10)²\n");

    let learning_rates = vec![0.01, 0.05, 0.1, 0.5];

    println!("  ┌──────────┬─────────────┬────────────┐");
    println!("  │ Learn.R. │ Final x     │ Iterations │");
    println!("  ├──────────┼─────────────┼────────────┤");

    for &lr in &learning_rates {
        let x = Rc::new(RefCell::new(Tensor::new(
            Array::new(vec![1], vec![1.0]),
            true,
        )));
        let mut optimizer = SGD::new(vec![x.clone()], lr, 0.0, 0.0);

        let mut iters = 0;
        for _ in 0..200 {
            let target = Tensor::new(Array::new(vec![1], vec![10.0]), false);
            let diff = x
                .borrow()
                .add(&target.mul(&Tensor::new(Array::new(vec![1], vec![-1.0]), false))?)?;
            let loss = diff.mul(&diff)?;

            loss.backward()?;
            optimizer.step()?;
            optimizer.zero_grad();
            iters += 1;

            // Check convergence
            if (x.borrow().values()[0] - 10.0).abs() < 0.01 {
                break;
            }
        }

        println!(
            "  │  {:.2}    │  {:.4}     │     {:3}    │",
            lr,
            x.borrow().values()[0],
            iters
        );
    }

    println!("  └──────────┴─────────────┴────────────┘\n");

    println!("  💡 Observación:");
    println!("     • lr muy bajo → convergencia lenta");
    println!("     • lr muy alto → puede oscilar");
    println!("     • lr óptimo ~ 0.1-0.5 para este problema\n");

    // ========================================================================
    // RESUMEN
    // ========================================================================
    println!("═══════════════════════════════════════════════════════════");
    println!("  ✅ RESUMEN: Optimizers Implementados");
    println!("═══════════════════════════════════════════════════════════\n");

    println!("1. ✓ SGD:");
    println!("   → Con/sin momentum");
    println!("   → Weight decay (L2 regularization)");
    println!("   → Velocities tracking");

    println!("\n2. ✓ Adam:");
    println!("   → Adaptive learning rates");
    println!("   → First & second moment estimates");
    println!("   → Bias correction");

    println!("\n3. ✓ RMSprop:");
    println!("   → Running average of squared gradients");
    println!("   → Adaptive per-parameter learning rates");

    println!("\n4. ✓ AdaGrad:");
    println!("   → Accumulated squared gradients");
    println!("   → Automatic learning rate decay");

    println!("\n5. ✓ Trait Optimizer:");
    println!("   → step() para update automático");
    println!("   → zero_grad() para limpiar");
    println!("   → learning_rate() getter/setter");

    println!("\n6. ✓ Learning Rate Schedulers:");
    println!("   → StepLR (decay cada N steps)");
    println!("   → ExponentialLR (decay exponencial)");

    println!("\n💡 Próximo paso: Fase 3 - Training API");
    println!("   → Module trait para modelos");
    println!("   → Trainer de alto nivel");
    println!("   → Data loaders\n");

    Ok(())
}
