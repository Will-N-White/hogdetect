


import os
import itertools
import torch
import yaml
from ultralytics import YOLO

base_dataset_path = "/home/oem/Desktop/FinalDataset_Augmented"

# expanded hyperparameter sweep
learning_rates = [
    0.0005, 0.001, 0.002, 0.003,
    0.005, 0.007, 0.010, 0.015,
    0.020
]
momentums = [
    0.60, 0.70, 0.75, 0.80,
    0.85, 0.90, 0.95, 0.99
]

epochs     = 30
imgsz      = 640
batch_size = 16
device     = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print(f"\nUsing device: {device.upper()}\n")

    # discover all dataset subfolders
    datasets = [
        d for d in os.listdir(base_dataset_path)
        if os.path.isdir(os.path.join(base_dataset_path, d))
    ]
    if not datasets:
        print("❌ No subfolders found under", base_dataset_path)
        return

    print("Found datasets:", datasets, "\n")

    for ds in datasets:
        train_images = os.path.join(base_dataset_path, ds, "train", "images")
        val_images   = os.path.join(base_dataset_path, ds,   "val", "images")

        # skip if missing or empty
        if not os.path.isdir(train_images) or not os.listdir(train_images):
            print(f"⚠️  Skipping {ds}: no images in {train_images}")
            continue
        if not os.path.isdir(val_images) or not os.listdir(val_images):
            print(f"⚠️  Skipping {ds}: no images in {val_images}")
            continue

        # build data dict on the fly
        data_dict = {
            "train": train_images,
            "val":   val_images,
            "nc":    2,
            "names": ["pig", "not_pig"]
        }

        # write a temporary YAML file
        yaml_path = f"{ds}_data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data_dict, f)

        print(f"=== Training on dataset: {ds} ===")

        # grid search
        for lr, mom in itertools.product(learning_rates, momentums):
            run_name = f"{ds}_lr{lr}_mom{mom}"
            print(f"\n– Experiment: {run_name}")

            model = YOLO("yolov8n.pt")
            try:
                model.train(
                    data=yaml_path,        # path to the YAML we just wrote
                    epochs=epochs,
                    imgsz=imgsz,
                    batch=batch_size,
                    lr0=lr,
                    momentum=mom,
                    optimizer="SGD",
                    name=run_name,
                    exist_ok=True,
                    device=device,
                    half=True             # mixed precision
                )
            except Exception as e:
                print(f"⚠️  Error on {run_name}: {e}")
            finally:
                # cleanup to free GPU memory
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # remove the temporary YAML
        os.remove(yaml_path)

    print("\n🎉 All runs completed!")

if __name__ == "__main__":
    main()
