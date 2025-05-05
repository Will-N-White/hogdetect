import os
import shutil

# Define dataset paths
dataset_1 = "/home/oem/Desktop/ThermalDataset_1"
dataset_2 = "/home/oem/Desktop/ThermalDataset_2"
merged_dataset = "/home/oem/Desktop/ThermalDataset"

# Define subdirectories for images and labels
subdirs = ["images", "labels"]

# Ensure the merged dataset structure exists
for subdir in subdirs:
    os.makedirs(os.path.join(merged_dataset, subdir), exist_ok=True)

# Function to copy files from source to destination
def copy_files(src_folder, dest_folder):
    if os.path.exists(src_folder):
        for file in os.listdir(src_folder):
            src_file = os.path.join(src_folder, file)
            dest_file = os.path.join(dest_folder, file)

            # Copy only if it's a file (not a folder)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dest_file)  # Preserve metadata

# Copy images and labels from both datasets
for dataset in [dataset_1, dataset_2]:
    for subdir in subdirs:
        copy_files(os.path.join(dataset, subdir), os.path.join(merged_dataset, subdir))

print("✅ Merging complete! ThermalDataset now contains all images and labels.")
