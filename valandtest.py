import os
import shutil
import random

# Define dataset paths
dataset_path = "/home/oem/Desktop/ThermalDataset"
train_images_path = os.path.join(dataset_path, "train", "images")
train_labels_path = os.path.join(dataset_path, "train", "labels")
val_images_path = os.path.join(dataset_path, "val", "images")
val_labels_path = os.path.join(dataset_path, "val", "labels")
test_images_path = os.path.join(dataset_path, "test", "images")
test_labels_path = os.path.join(dataset_path, "test", "labels")

# Ensure required directories exist
if not os.path.exists(train_images_path) or not os.path.exists(train_labels_path):
    raise FileNotFoundError(f"Training images or labels directory not found.")

os.makedirs(val_images_path, exist_ok=True)
os.makedirs(val_labels_path, exist_ok=True)
os.makedirs(test_images_path, exist_ok=True)
os.makedirs(test_labels_path, exist_ok=True)

# List all images in the training images folder
image_files = [f for f in os.listdir(train_images_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

if not image_files:
    raise ValueError("No image files found in the training directory.")

# Shuffle images for random selection
random.shuffle(image_files)

# Determine split counts
num_images = len(image_files)
num_val = int(num_images * 0.2)  # 20% for validation
num_test = int(num_images * 0.1)  # 10% for testing

# Function to move images and corresponding labels
def move_files(file_list, src_img_path, src_lbl_path, dest_img_path, dest_lbl_path):
    for file in file_list:
        base_name, ext = os.path.splitext(file)
        label_file = base_name + ".txt"
        
        img_src = os.path.join(src_img_path, file)
        img_dest = os.path.join(dest_img_path, file)
        lbl_src = os.path.join(src_lbl_path, label_file)
        lbl_dest = os.path.join(dest_lbl_path, label_file)
        
        shutil.move(img_src, img_dest)
        if os.path.exists(lbl_src):
            shutil.move(lbl_src, lbl_dest)

# Move validation and test images
move_files(image_files[:num_val], train_images_path, train_labels_path, val_images_path, val_labels_path)
move_files(image_files[num_val:num_val+num_test], train_images_path, train_labels_path, test_images_path, test_labels_path)

print(f"Moved {num_val} images to validation and {num_test} images to test folder.")
