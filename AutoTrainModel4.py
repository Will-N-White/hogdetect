from ultralytics import YOLO
import itertools
import os
import torch
import time

# Updated hyperparameter search space
param_grid = {
    "learning_rate": [0.001, 0.002, 0.003],
    "momentum": [0.7, 0.8, 0.9],
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
for i, params in enumerate(param_combinations, start=1):
    learning_rate, momentum, optimizer, batch_size = params
    # Create a unique model name for this combination
    model_name = f"exp_lr{learning_rate}_mom{momentum}_opt{optimizer}_bs{batch_size}"
    print(f"\n===== Starting experiment {i}/{len(param_combinations)}: {model_name} on {device.upper()} =====")
    
    try:
        # Load the YOLO model (this will load pretrained weights)
        model = YOLO('yolov8n.pt')
        
        # Train the model for 50 epochs; results are saved in a folder with the model_name
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
        
        # Optionally, export the best model. Remove unsupported 'half' argument before exporting.
        try:
            if hasattr(model, 'args'):
                model.args.pop('half', None)
            model.export(format="torchscript", dynamic=False)
        except Exception as e:
            print(f"Export failed for {model_name}: {e}")
        
        print(f"===== Completed training for {model_name}. Results are in: runs/detect/{model_name} =====")
    
    except Exception as e:
        print(f"Error encountered during training for {model_name}: {e}")
    
    # Explicitly free GPU memory after each experiment
    del model
    torch.cuda.empty_cache()
    # Optional short delay to ensure cleanup completes before starting the next experiment
    time.sleep(10)

print("🎉 All experiments completed successfully!")



