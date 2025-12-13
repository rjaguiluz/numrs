#!/usr/bin/env python3
"""
Benchmark: PyTorch vs NumRs - House Price Prediction
Comparación justa del mismo modelo y dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np

# Same dataset as NumRs example
train_data = np.array([
    [50.0, 1.0, 5.0, 2.0],
    [75.0, 2.0, 10.0, 3.0],
    [100.0, 3.0, 5.0, 1.5],
    [120.0, 3.0, 15.0, 5.0],
    [60.0, 2.0, 8.0, 4.0],
    [90.0, 2.0, 3.0, 2.5],
    [150.0, 4.0, 10.0, 3.0],
    [45.0, 1.0, 20.0, 6.0],
    [85.0, 3.0, 12.0, 4.5],
    [110.0, 3.0, 7.0, 2.0],
    [65.0, 2.0, 15.0, 5.5],
    [95.0, 2.0, 5.0, 1.0],
    [55.0, 1.0, 3.0, 3.0],
    [130.0, 4.0, 8.0, 2.5],
    [70.0, 2.0, 18.0, 7.0],
], dtype=np.float32)

train_targets = np.array([
    [150.0], [200.0], [350.0], [250.0], [180.0],
    [280.0], [400.0], [120.0], [220.0], [330.0],
    [160.0], [320.0], [170.0], [380.0], [140.0],
], dtype=np.float32)

# Same architecture: 4 -> 16 -> 8 -> 1
class HousePriceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(8, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

def train_pytorch():
    print("=" * 60)
    print("  PyTorch: House Price Prediction")
    print("=" * 60)
    print()
    
    # Setup
    model = HousePriceModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Convert to tensors
    X = torch.from_numpy(train_data)
    y = torch.from_numpy(train_targets)
    
    print(f"Dataset: {len(train_data)} examples")
    print(f"Architecture: 4 -> 16 -> 8 -> 1")
    print(f"Optimizer: Adam (lr=0.001)")
    print(f"Epochs: 100")
    print()
    
    # Training
    start_time = time.time()
    
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/100: loss={loss.item():.4f}")
    
    training_time = time.time() - start_time
    
    print()
    print(f"✓ Training completed!")
    print(f"✓ Final loss: {loss.item():.6f}")
    print(f"✓ Training time: {training_time:.2f}s")
    print()
    
    # Inference
    model.eval()
    with torch.no_grad():
        test_cases = [
            ([80.0, 2.0, 7.0, 3.0], "Medium house, close"),
            ([140.0, 4.0, 5.0, 1.5], "Large house, new, central"),
            ([50.0, 1.0, 15.0, 6.0], "Small house, old, far"),
        ]
        
        print("Predictions:")
        for features, desc in test_cases:
            x = torch.tensor([features], dtype=torch.float32)
            pred = model(x).item()
            print(f"  {desc}: ${pred:.0f}k USD")
    
    print()
    
    # Export to ONNX
    start_export = time.time()
    dummy_input = torch.randn(1, 4)
    torch.onnx.export(
        model,
        dummy_input,
        "house_price_pytorch.onnx",
        input_names=['input'],
        output_names=['output'],
        opset_version=18
    )
    export_time = time.time() - start_export
    
    print(f"✓ ONNX export time: {export_time:.3f}s")
    print()
    
    return {
        'training_time': training_time,
        'export_time': export_time,
        'final_loss': loss.item(),
        'total_time': training_time + export_time
    }

def train_tensorflow():
    print("=" * 60)
    print("  TensorFlow/Keras: House Price Prediction")
    print("=" * 60)
    print()
    
    import tensorflow as tf
    
    # Same architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    
    print(f"Dataset: {len(train_data)} examples")
    print(f"Architecture: 4 -> 16 -> 8 -> 1")
    print(f"Optimizer: Adam (lr=0.001)")
    print(f"Epochs: 100")
    print()
    
    # Training
    start_time = time.time()
    
    history = model.fit(
        train_data, 
        train_targets,
        epochs=100,
        verbose=0,
        batch_size=len(train_data)  # Full batch like NumRs
    )
    
    training_time = time.time() - start_time
    final_loss = history.history['loss'][-1]
    
    print(f"✓ Training completed!")
    print(f"✓ Final loss: {final_loss:.6f}")
    print(f"✓ Training time: {training_time:.2f}s")
    print()
    
    # Inference
    test_cases = np.array([
        [80.0, 2.0, 7.0, 3.0],
        [140.0, 4.0, 5.0, 1.5],
        [50.0, 1.0, 15.0, 6.0],
    ], dtype=np.float32)
    
    predictions = model.predict(test_cases, verbose=0)
    
    print("Predictions:")
    descriptions = [
        "Medium house, close",
        "Large house, new, central",
        "Small house, old, far"
    ]
    for pred, desc in zip(predictions, descriptions):
        print(f"  {desc}: ${pred[0]:.0f}k USD")
    
    print()
    
    return {
        'training_time': training_time,
        'final_loss': final_loss,
        'total_time': training_time
    }

if __name__ == "__main__":
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  Benchmark: PyTorch vs TensorFlow vs NumRs              ║")
    print("║  Model: House Price Prediction (4->16->8->1)            ║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    # PyTorch
    pytorch_results = train_pytorch()
    
    print()
    
    # TensorFlow
    try:
        tf_results = train_tensorflow()
    except ImportError:
        print("TensorFlow not installed, skipping...")
        tf_results = None
    
    print()
    print("=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    print()
    print(f"PyTorch:")
    print(f"  Training:  {pytorch_results['training_time']:.2f}s")
    print(f"  Export:    {pytorch_results['export_time']:.3f}s")
    print(f"  Total:     {pytorch_results['total_time']:.2f}s")
    print(f"  Final Loss: {pytorch_results['final_loss']:.2f}")
    print()
    
    if tf_results:
        print(f"TensorFlow:")
        print(f"  Training:  {tf_results['training_time']:.2f}s")
        print(f"  Total:     {tf_results['total_time']:.2f}s")
        print(f"  Final Loss: {tf_results['final_loss']:.2f}")
        print()
    
    print(f"NumRs (measured):")
    print(f"  Total:     ~7.90s (includes compilation overhead)")
    print(f"  Final Loss: 65217.90")
    print()
    print("Note: NumRs includes first-time WebGPU initialization")
    print("      and adaptive lookup table setup overhead.")
    print()
