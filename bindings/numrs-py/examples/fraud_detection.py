import numrs
import numrs.ops.reduction as F

def main():
    print("NumRs Python Bindings Verification")
    
    # 1. Data (XOR-like problem logic but larger)
    inputs = [
        [0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]
    ]
    targets = [
        [0.0], [1.0], [1.0], [0.0]
    ]
    
    dataset = numrs.Dataset(inputs, targets, batch_size=2)
    print("Dataset created.")


    # 2. Model
    model = numrs.Sequential()
    model.add(numrs.Linear(3, 16))
    model.add(numrs.ReLU())
    model.add(numrs.Linear(16, 8))
    model.add(numrs.ReLU())
    model.add(numrs.Linear(8, 1))
    model.add(numrs.Sigmoid())
    print("Model constructed.")

    # 3. Trainer
    # Verify new dynamic builder: "adamw"
    print("Building trainer with AdamW/MSE...")
    try:
        trainer = numrs.Trainer(model, optimizer="adam", loss="mse", lr=0.01)
    except ValueError as e:
        print(f"Error building trainer: {e}")
        return
        
    print("Trainer constructed.")
    
    # 4. Train
    print("Starting training...")
    trainer.fit(dataset, epochs=10) # C-side prints loss
    print("Training finished.")

if __name__ == "__main__":
    main()
