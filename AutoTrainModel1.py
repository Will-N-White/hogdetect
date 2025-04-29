from ultralytics import YOLO
import itertools
import os

# Define adjustable hyperparameters and their values for testing
param_grid = {
    "learning_rate": [0.001, 0.005, 0.01],  # Test different learning rates
    "momentum": [0.8, 0.9, 0.95],          # Test different momentum values
    "weight_decay": [0.0001, 0.0005, 0.001], # Test different weight decay
    "optimizer": ["SGD", "Adam"],           # Test different optimizers
    "batch_size": [8, 16, 32],              # Test different batch sizes
}

# Generate all possible combinations of parameters
param_combinations = list(itertools.product(*param_grid.values()))

# Define static parameters
data_path = "/home/oem/Desktop/ThermalDataset/data.yaml"
imgsz = 640
epochs = 50  # Keep epochs moderate to prevent excessive training time
device = 'cuda'  # Force GPU usage

    # Save results to a single file
    with open(results_file, "a") as f:
        f.write(f"\nResults for {model_name}:\n")
        f.write(str(results) + "\n")
        f.write("-" * 50 + "\n")

    print(f"Training completed for {model_name}, results saved.")

print("\nAll experiments completed. Results stored in:")
print(results_file)
