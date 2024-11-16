import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model_path = 'C:\\MLfiles\\runs\\detect\\hog_detection_model_pc3\\weights\\best.pt'  # Update this to the actual path of your YOLOv8 model
model = YOLO(model_path)

# Define thresholds


# Open a connection to the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera; change if using another camera

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Process each frame from the camera
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Run the YOLOv8 model on the frame
    results = model.predict(source=frame, save=False, show=False, stream=True)  # Stream mode for real-time use

    # Draw bounding boxes and labels on the frame
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers
            confidence = box.conf[0]
            label = model.names[int(box.cls[0])]

            # Apply different thresholds based on the label
            if label == "pig" and confidence >= pig_threshold:
                # Draw rectangle and label for pig
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            elif label == "not_pig" and confidence >= not_pig_threshold:
                # Draw rectangle and label for not_pig with raised threshold
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame with detections
    cv2.imshow('YOLOv8 Real-Time Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
