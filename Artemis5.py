import cv2
import numpy as np
import pygame
import time

# Camera configuration
usb_camera_index = 0  # ONN face camera
thermal_camera_index = 2  # FLIR Lepton camera at video1

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
try:
    pygame.mixer.music.load(sound_path)
    sound_available = True
except:
    print(f"Warning: Could not load sound file at {sound_path}")
    sound_available = False

# Video recording variables
video_writer = None
is_recording = False
recording_start_time = 0
recording_filename = ""

def start_recording(frame_width, frame_height):
    """Initialize video writer with current timestamp as filename"""
    global video_writer, is_recording, recording_start_time, recording_filename
    timestamp = int(time.time())
    recording_filename = f"recording_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4 format
    video_writer = cv2.VideoWriter(recording_filename, fourcc, 20.0, (frame_width, frame_height))
    is_recording = True
    recording_start_time = time.time()
    print(f"Started recording to {recording_filename}")
    return recording_filename

def stop_recording():
    """Release the video writer and stop recording"""
    global video_writer, is_recording, recording_filename
    if video_writer is not None:
        video_writer.release()
        duration = time.time() - recording_start_time
        print(f"Recording stopped. Duration: {duration:.2f} seconds")
        print(f"Saved to {recording_filename}")
    video_writer = None
    is_recording = False

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

# FPS tracking variables
actual_fps = 0
fps_counter = 0
fps_timer = time.time()

# Keyboard debug flag
debug_keys = True

print("Camera recording system initialized")
print("Controls:")
print("  Spacebar: Play sound")
print("  r: Start recording")
print("  t: Stop recording")
print("  s: Save current frame")
print("  d: Display debug info")
print("  q: Quit")

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

    # Calculate actual FPS
    fps_counter += 1
    if current_time - fps_timer >= 1.0:  # Update FPS every second
        actual_fps = fps_counter
        fps_counter = 0
        fps_timer = current_time

    # Add frame labels and FPS indicators
    cv2.putText(usb_frame, "Normal Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(thermal_display, "Thermal Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(usb_frame, f"FPS: {actual_fps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Stack images
    stacked_frames = np.hstack((usb_frame, thermal_display))

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

    # Add keyboard controls info
    controls_text = "Spacebar: Play sound | r: Record | t: Stop recording | s: Save frame | q: Quit | d: Debug"
    cv2.putText(stacked_frames, controls_text, (10, stacked_frames.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the result
    cv2.imshow("Dual Camera Recording System", stacked_frames)

    # Handle key presses
    key = cv2.waitKey(30) & 0xFF  # 30ms wait time for better key response

    # Debug key detection if enabled
    if debug_keys and key != 255:
        print(f"Key pressed: {key}, char: {chr(key) if key < 128 else 'N/A'}")

    if key == ord(' '):
        if sound_available:
            print("Space bar pressed - playing sound")
            pygame.mixer.music.play()
        else:
            print("Sound file not available")
    elif key == ord('q'):
        print("Quit key pressed - exiting")
        if is_recording:
            stop_recording()
        break
    elif key == ord('s'):
        # Save current frames
        timestamp = int(time.time())
        usb_filename = f"usb_{timestamp}.jpg"
        thermal_filename = f"thermal_{timestamp}.jpg"
        cv2.imwrite(usb_filename, usb_frame)
        cv2.imwrite(thermal_filename, thermal_display)
        print(f"Saved frames: {usb_filename} and {thermal_filename}")
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
        print(f"FPS setting: {thermal_cam.get(cv2.CAP_PROP_FPS)}")
        print(f"Format: {thermal_cam.get(cv2.CAP_PROP_FORMAT)}")
        print(f"FOURCC: {int(thermal_cam.get(cv2.CAP_PROP_FOURCC))}")
        print(f"Actual display FPS: {actual_fps}")
        print(f"Recording: {'Yes' if is_recording else 'No'}")
        if is_recording:
            print(f"Recording file: {recording_filename}")
            print(f"Recording time: {time.time() - recording_start_time:.2f} seconds")

        # Try to get a frame and print its properties
        ret, frame = thermal_cam.read()
        if ret:
            print(f"Thermal frame shape: {frame.shape}, dtype: {frame.dtype}")
            print(f"Min/Max values: {frame.min()}/{frame.max()}")
        else:
            print("Failed to grab thermal frame for debug")

# Clean up resources
print("Cleaning up resources...")
if is_recording:
    stop_recording()
usb_cam.release()
thermal_cam.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
print("Program terminated successfully")
