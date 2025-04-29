import cv2
import numpy as np
import torch
import pygame
import time
from ultralytics import YOLO

# Load YOLOv8 models
thermal_model_path = "/home/artemis/Artemis/ThermalDataset/runs/detect/exp_lr0.001_mom0.8_optSGD_bs16/weights/best.pt"
normal_model_path = "/home/artemis/Artemis/HogDataset/runs/detect/exp_lr0.001_mom0.8_optAdam_bs16/weights/best.pt"
thermal_model = YOLO(thermal_model_path)
normal_model = YOLO(normal_model_path)

# Camera configuration
usb_camera_index = 2  # ONN face camera
thermal_camera_index = 0  # FLIR Lepton camera at video1

# Open cameras with specific settings
usb_cam = cv2.VideoCapture(usb_camera_index)
usb_cam.set(cv2.CAP_PROP_FPS, 30)  # Set to 30 FPS

# Use UYVY format that matches ffplay output
thermal_cam = cv2.VideoCapture(thermal_camera_index, cv2.CAP_V4L2)
thermal_cam.set(cv2.CAP_PROP_FPS, 9)  # Set to 9 FPS
thermal_cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer size
thermal_cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('U', 'Y', 'V', 'Y'))  # Set to UYVY format
thermal_cam.set(cv2.CAP_PROP_CONVERT_RGB, 1)  # Convert to RGB

# Verify camera access
if not usb_cam.isOpened():
    print(f"Error: Could not open USB camera at index {usb_camera_index}")
    exit()
if not thermal_cam.isOpened():
    print(f"Error: Could not open thermal camera at index {thermal_camera_index}")
    exit()

# Initialize Pygame for sound
pygame.mixer.init()
sound_path = "/home/artemis/Artemis/WagnerTheRideOfTheValkyrieswww.keepvid.com.mp3"
pygame.mixer.music.load(sound_path)

# Video recording variables
video_writer = None
is_recording = False
recording_start_time = 0

def start_recording(frame_width, frame_height):
    """Initialize video writer with current timestamp as filename"""
    global video_writer, is_recording, recording_start_time
    timestamp = int(time.time())
    filename = f"recording_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4 format
    video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (frame_width, frame_height))
    is_recording = True
    recording_start_time = time.time()
    print(f"Started recording to {filename}")
    return filename

def stop_recording():
    """Release the video writer and stop recording"""
    global video_writer, is_recording
    if video_writer is not None:
        video_writer.release()
        duration = time.time() - recording_start_time
        print(f"Recording stopped. Duration: {duration:.2f} seconds")
    video_writer = None
    is_recording = False

def run_detection(model, frame):
    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy() if len(results[0].boxes) > 0 else np.array([])
    return detections

def process_thermal_frame(frame):
    """Process thermal data from FLIR Lepton for display"""
    if frame is None or frame.size == 0:
        print("Empty thermal frame received")
        return np.zeros((120, 160, 3), dtype=np.uint8)
    
    try:
        # Extract Y channel for thermal data (UYVY format has Y as intensity)
        if len(frame.shape) == 3:
            # If frame is in color format after conversion
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Normalize for better visualization
        min_val, max_val, _, _ = cv2.minMaxLoc(gray)
        if max_val > min_val:
            normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        else:
            normalized = gray
            
        normalized = normalized.astype(np.uint8)
        
        # Apply colormap for thermal visualization
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
        
        return colored
    except Exception as e:
        print(f"Error processing thermal frame: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros((120, 160, 3), dtype=np.uint8)

# Warm up the cameras
print("Warming up cameras...")
for _ in range(5):
    usb_cam.read()
    thermal_cam.read()
    time.sleep(0.1)

# Frame synchronization variables
last_thermal_frame = None
thermal_frame_timestamp = 0
thermal_frame_interval = 1.0 / 9.0  # ~111ms between frames at 9 FPS

# Detection rate limiting variables
last_detection_time = 0
detection_interval = 1.0 / 10.0  # 10 FPS for object detection
actual_fps = 0
fps_counter = 0
fps_timer = time.time()

# Keyboard debug flag
debug_keys = True

while True:
    loop_start_time = time.time()
    
    # Always read from USB camera (30 FPS)
    ret_usb, usb_frame = usb_cam.read()
    
    # Read from thermal camera only when needed (9 FPS)
    current_time = time.time()
    if current_time - thermal_frame_timestamp >= thermal_frame_interval:
        # Flush the buffer
        for _ in range(2):
            thermal_cam.grab()
        
        # Read the thermal frame
        ret_thermal, thermal_frame = thermal_cam.read()
        
        if ret_thermal and thermal_frame is not None and thermal_frame.size > 0:
            # Process new thermal frame
            thermal_processed = process_thermal_frame(thermal_frame)
            last_thermal_frame = thermal_processed
            thermal_frame_timestamp = current_time
    
    if not ret_usb:
        print("Error: Could not read from USB camera")
        break
    
    # Use the last valid thermal frame or create a placeholder
    if last_thermal_frame is None:
        thermal_display = np.zeros((usb_frame.shape[0], usb_frame.shape[1], 3), dtype=np.uint8)
        thermal_display[:, :] = (0, 0, 128)  # Dark blue placeholder
    else:
        thermal_display = cv2.resize(last_thermal_frame, (usb_frame.shape[1], usb_frame.shape[0]))
    
    # Run object detection at limited frame rate to reduce CPU load
    usb_detections = []
    thermal_detections = []
    if current_time - last_detection_time >= detection_interval:
        usb_detections = run_detection(normal_model, usb_frame)
        thermal_detections = run_detection(thermal_model, thermal_display)
        last_detection_time = current_time
    
    # Extract detection data
    usb_pig_conf = [det[4] for det in usb_detections if len(det) > 5 and det[5] == 0]
    thermal_pig_conf = [det[4] for det in thermal_detections if len(det) > 5 and det[5] == 0]
    
    usb_detects_pig = any(conf >= 0.8 for conf in usb_pig_conf)
    thermal_detects_pig = any(conf >= 0.8 for conf in thermal_pig_conf)
    usb_detects_not_pig = any(len(det) > 5 and det[5] == 1 for det in usb_detections)
    thermal_detects_not_pig = any(len(det) > 5 and det[5] == 1 for det in thermal_detections)
    
    # Draw detections on USB frame
    if len(usb_detections) > 0:
        for det in usb_detections:
            if len(det) > 5:
                x1, y1, x2, y2, conf, cls = det
                label = "Pig" if cls == 0 else "Not_Pig"
                color = (0, 255, 0) if cls == 0 else (0, 0, 255)
                cv2.rectangle(usb_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(usb_frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw detections on thermal frame
    if len(thermal_detections) > 0:
        for det in thermal_detections:
            if len(det) > 5:
                x1, y1, x2, y2, conf, cls = det
                label = "Pig" if cls == 0 else "Not_Pig"
                color = (0, 255, 0) if cls == 0 else (0, 0, 255)
                cv2.rectangle(thermal_display, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(thermal_display, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Calculate actual FPS
    fps_counter += 1
    if current_time - fps_timer >= 1.0:  # Update FPS every second
        actual_fps = fps_counter
        fps_counter = 0
        fps_timer = current_time
    
    # Add frame labels and FPS indicators
    cv2.putText(usb_frame, "Normal Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(thermal_display, "Thermal Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(usb_frame, f"FPS: {actual_fps} (Display) / 10 (Detection)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Stack images
    stacked_frames = np.hstack((usb_frame, thermal_display))
    
    # Add detection status
    if usb_detects_pig and thermal_detects_pig and not usb_detects_not_pig and not thermal_detects_not_pig:
        status = "DETECTION: PIG CONFIRMED (Both Cameras)"
        color = (0, 255, 0)
    else:
        status = "DETECTION: Waiting for confirmation..."
        color = (0, 0, 255)
    
    cv2.putText(stacked_frames, status, (10, stacked_frames.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Add recording indicator if recording
    if is_recording:
        # Calculate recording time
        rec_time = time.time() - recording_start_time
        rec_text = f"REC {int(rec_time // 60):02d}:{int(rec_time % 60):02d}"
        
        # Add red circle and recording time
        cv2.circle(stacked_frames, (stacked_frames.shape[1] - 30, 30), 10, (0, 0, 255), -1)
        cv2.putText(stacked_frames, rec_text, (stacked_frames.shape[1] - 120, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Write the frame to video
        if video_writer is not None:
            video_writer.write(stacked_frames)
    
    # Add keyboard controls info including recording controls
    controls_text = "Spacebar: Play sound | r: Record | t: Stop recording | q: Quit | s: Save | d: Debug"
    cv2.putText(stacked_frames, controls_text, (10, stacked_frames.shape[0] - 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display the result
    cv2.imshow("USB + FLIR YOLO Detection", stacked_frames)
    
    # Handle key presses with improved key detection
    key = cv2.waitKey(30) & 0xFF  # Increased wait time from 1ms to 30ms
    
    # Debug key detection
    if debug_keys and key != 255:
        print(f"Key pressed: {key}, char: {chr(key) if key < 128 else 'N/A'}")
    
    if key == ord(' '):
        print("Space bar pressed - playing sound if detection confirmed")
        if usb_detects_pig and thermal_detects_pig and not usb_detects_not_pig and not thermal_detects_not_pig:
            pygame.mixer.music.play()
            print("Playing sound: Pig confirmed")
        else:
            print("Sound not allowed: Detection requirements not met")
    elif key == ord('q'):
        print("Quit key pressed - exiting")
        if is_recording:
            stop_recording()
        break
    elif key == ord('s'):
        # Save current frames
        timestamp = int(time.time())
        cv2.imwrite(f"usb_{timestamp}.jpg", usb_frame)
        cv2.imwrite(f"thermal_{timestamp}.jpg", thermal_display)
        print(f"Saved frames with timestamp {timestamp}")
    elif key == ord('r'):
        # Toggle recording
        if not is_recording:
            # Get dimensions of the stacked frames for recording
            height, width = stacked_frames.shape[:2]
            start_recording(width, height)
        else:
            print("Already recording. Press 't' to stop.")
    elif key == ord('t'):
        # Stop recording
        if is_recording:
            stop_recording()
        else:
            print("Not currently recording. Press 'r' to start recording.")
    elif key == ord('d'):
        # Print detailed debug information
        print("\nDEBUG INFO:")
        print(f"Thermal camera index: {thermal_camera_index}")
        print(f"Frame dimensions: {thermal_cam.get(cv2.CAP_PROP_FRAME_WIDTH)}x{thermal_cam.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        print(f"FPS: {thermal_cam.get(cv2.CAP_PROP_FPS)}")
        print(f"Format: {thermal_cam.get(cv2.CAP_PROP_FORMAT)}")
        print(f"Mode: {thermal_cam.get(cv2.CAP_PROP_MODE)}")
        print(f"FOURCC: {int(thermal_cam.get(cv2.CAP_PROP_FOURCC))}")
        print(f"Actual display FPS: {actual_fps}")
        print(f"Detection FPS: {1.0/detection_interval}")
        print(f"Recording: {'Yes' if is_recording else 'No'}")
        
        # Try to get a frame and print its properties
        ret, frame = thermal_cam.read()
        if ret:
            print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
            print(f"Min/Max values: {frame.min()}/{frame.max()}")
        else:
            print("Failed to grab frame for debug")

# Clean up resources
print("Cleaning up resources...")
if is_recording:
    stop_recording()
usb_cam.release()
thermal_cam.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
print("Program terminated successfully")
