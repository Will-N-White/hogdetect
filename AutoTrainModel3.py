from ultralytics import YOLO
import itertools
import os
import torch

# Define your hyperparameter search space
param_grid = {
    "learning_rate": [0.001, 0.005, 0.01],
    "momentum": [0.8, 0.9, 0.95],         # Example: two momentum values
    "optimizer": ["SGD", "Adam"],
    "batch_size": [8, 16],
}

# Generate all hyperparameter combinations
param_combinations = list(itertools.product(*param_grid.values()))

# Dataset and training settings
data_path = "/home/oem/Desktop/ThermalDataset/data.yaml"
imgsz = 640
epochs = 50  # Each experiment runs for 50 epochs
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device.upper()}")
print(f"Total experiments: {len(param_combinations)}\n")

# Loop over each hyperparameter combination
for params in param_combinations:
    learning_rate, momentum, optimizer, batch_size = params
    # Create a unique model name for this combination
    model_name = f"exp_lr{learning_rate}_mom{momentum}_opt{optimizer}_bs{batch_size}"
    print(f"\n===== Starting training for {model_name} on {device.upper()} =====")
    
    # Load the YOLO model (this will download or load the pretrained weights)
    model = YOLO('yolov8n.pt')
    
    # Start training for 50 epochs; results are saved automatically into a folder with the model_name
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        lr0=learning_rate,
        momentum=momentum,
        optimizer=optimizer,
        name=model_name,
        device=device,
        half=True,         # Use AMP to reduce memory usage
        exist_ok=True      # Reuse directory if it exists
    )
    
    # Optionally, export the best model. Remove unsupported arguments before exporting.
    try:
        if hasattr(model, 'args'):
            model.args.pop('half', None)
        model.export(format="torchscript", dynamic=False)
    except Exception as e:
        print(f"Export failed for {model_name}: {e}")
    
    print(f"===== Completed training for {model_name}.")
    print(f"Results (including logs, weights, and exported model) are saved in: runs/detect/{model_name}\n")

print("🎉 All experiments completed successfully!")


