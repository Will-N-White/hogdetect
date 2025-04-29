from ultralytics import YOLO
import itertools
import os
import torch

# Define hyperparameter search space
param_grid = {
    "learning_rate": [0.001, 0.005, 0.01],
    "momentum": [0.8, 0.9, 0.95],
    "weight_decay": [0.0001, 0.0005, 0.001],
    "optimizer": ["SGD", "Adam"],
    "batch_size": [8, 16],  # Avoid OOM with 32
}

# Generate all hyperparameter combinations
param_combinations = list(itertools.product(*param_grid.values()))

# Set dataset and training parameters
data_path = "/home/oem/Desktop/ThermalDataset/data.yaml"
imgsz = 640
epochs = 50  # ✅ Train for full 50 epochs unless early stopping
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Auto-detect GPU

print(f"Using device: {device.upper()}")

# Results file
results_file = "/home/oem/Desktop/hyperparameter_results.txt"

# Reset results file
with open(results_file, "w") as f:
    f.write("Hyperparameter Tuning Results\n")
    f.write("=" * 50 + "\n")

# Early stopping settings
early_stop_patience = 3  # Stop if no improvement for 5 epochs

# Best overall model tracker
best_overall_map50 = 0
best_overall_model = None
best_params = None

# Iterate over all hyperparameter combinations
for params in param_combinations:
    learning_rate, momentum, weight_decay, optimizer, batch_size = params

    model_name = f"exp_lr{learning_rate}_mom{momentum}_wd{weight_decay}_opt{optimizer}_bs{batch_size}"
    print(f"\n🚀 Starting training with: {model_name} on {device.upper()}")

    model = YOLO('yolov8n.pt')  # Load YOLO model

    # Early stopping variables
    best_map50 = 0
    no_improve_epochs = 0

    for epoch in range(epochs):  # ✅ Manually loop through epochs
        results = model.train(
            data=data_path,
            epochs=epochs,  # ✅ Train one epoch at a time
            imgsz=imgsz,
            batch=batch_size,
            lr0=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            optimizer=optimizer,
            name=model_name,
            device=device,
            half=True,  # Reduce memory usage
        )

        # ✅ Extract best mAP@50 from results safely
        current_map50 = getattr(results.box, "map", [0])[0]  # Get mAP@50 safely

        print(f"📊 Epoch {epoch+1}/{epochs} - mAP@50: {current_map50:.4f} (Best: {best_map50:.4f})")

        # ✅ Check if mAP@50 improved
        if current_map50 > best_map50:
            best_map50 = current_map50
            no_improve_epochs = 0  # Reset patience counter
        else:
            no_improve_epochs += 1

        # ✅ Stop training if no improvement for `early_stop_patience` epochs
        if no_improve_epochs >= early_stop_patience:
            print(f"🛑 Early stopping triggered at epoch {epoch+1}. No improvement for {early_stop_patience} epochs.")
            break  # Exit loop

    # ✅ Save final results
    with open(results_file, "a") as f:
        f.write(f"\nResults for {model_name} (Best mAP@50: {best_map50:.4f})\n")
        f.write("-" * 50 + "\n")

    print(f"✅ Training completed for {model_name}, results saved.")

    # ✅ Check if this model is the best overall
    if best_map50 > best_overall_map50:
        best_overall_map50 = best_map50
        best_overall_model = model_name
        best_params = params

# ✅ Print the final best model
print("\n🎉 All experiments completed.")
print(f"🏆 Best overall model: {best_overall_model} with mAP@50 = {best_overall_map50:.4f}")
print(f"🔢 Best hyperparameters: {best_params}")
print(f"📁 Results stored in: {results_file}")



