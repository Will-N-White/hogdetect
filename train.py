from ultralytics import YOLO

# Load the model
model = YOLO('yolov8m.pt')  # Loads the medium YOLOv8 model, balancing performance and accuracy.

# Train the model with detailed explanations for each hyperparameter
model.train(
    data='C:\MLfiles\Artemis\data.yaml',  # Path to the data.yaml file specifying training and validation images.
    epochs=2,                    # Total number of training epochs (passes through the dataset).
    imgsz=640,                    # Image size to which all images will be resized (640x640).
    batch=16,                     # Number of images per training batch. Smaller batch sizes can reduce memory usage.
    lr0=0.001,                    # Initial learning rate, controlling the step size at the beginning of training.
    lrf=0.01,                     # Final learning rate at the end of training. Model gradually reduces to this rate.
    momentum=0.937,               # Momentum helps the model to stabilize updates and improve convergence.
    weight_decay=0.0005,          # Regularization to reduce overfitting by penalizing larger weights.
    patience=5,                   # Stops training if no improvement in validation loss for 'patience' number of epochs.
    
    # Data Augmentation
    augment=True,                 # Enables data augmentation to improve model generalization on unseen data.
    hsv_h=0.015,                  # Randomly changes hue by this fraction to add color variability.
    hsv_s=0.7,                    # Randomly changes saturation by this fraction for more color diversity.
    hsv_v=0.4,                    # Randomly changes brightness by this fraction to account for lighting changes.
    flipud=0.0,                   # Probability of flipping images vertically (up-down) to diversify data.
    fliplr=0.5,                   # Probability of flipping images horizontally (left-right).
    mosaic=1.0,                   # Mosaic augmentation probability, combining four images for varied backgrounds.
    mixup=0.2,                    # Mixup augmentation probability, blending two images to reduce noise sensitivity.
    
    # Validation and Output
    split='val',                  # Specifies using the validation split in data.yaml for validation during training.
    verbose=True                  # Prints detailed output for each training epoch to track progress.
)
