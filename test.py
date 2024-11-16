import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import random

# Load the YOLOv8 model (replace with the path to your .pt file)
model_path = 'C:\\MLfiles\\runs\\detect\\hog_detection_model_pc3\\weights\\best.pt'
model = YOLO(model_path)

# Path to the dataset (update with your actual path)
data_path = r'C:\MLfiles\Artemis\Farm-Harmful-Animals-Dataset-2\test\images'

# List all image files in the directory
image_files = os.listdir(data_path)

# Select 20 random images
random_images = random.sample(image_files, 20)

# Loop through each selected image
for img_file in random_images:
    img_path = os.path.join(data_path, img_file)
    img = cv2.imread(img_path)

    # Run the YOLO model on the image
    results = model(img)

    # Draw bounding boxes on the image
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            confidence = box.conf[0]  # Confidence score
            class_id = int(box.cls[0])  # Class ID

            if class_id == 0:  # Assuming 'pig' is class 0; update if different
                # Draw rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # Add label with confidence
                label = f'Pig: {confidence:.2f}'
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Convert the image from BGR (OpenCV default) to RGB for displaying with matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image with bounding boxes
    plt.figure()
    plt.imshow(img_rgb)
    plt.title(img_file)
    plt.axis('off')
    plt.show()
    plt.close()  # Close the figure after displaying to free up memory

print("Detection displayed for 20 random images.")

