import sys
import os
import time
import cv2
from concurrent.futures import ThreadPoolExecutor

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from raybanmods_db.hand_gesture.hand_gesture_detection import detect_pointing_gesture, release_resources
from raybanmods_db.hand_gesture.hand_canvas_detection import detect_canvas_or_book
from raybanmods_db.facial_recognition.face_detection import detect_faces
from capture_live_feed import capture_screen, display_frame

# Define the region of the screen you want to capture
screen_region = {'top': 100, 'left': 100, 'width': 800, 'height': 600}

# Create a thread pool to limit the number of threads
executor = ThreadPoolExecutor(max_workers=4)

def downsample_frame(frame, scale=0.5):
    """Downsample the frame to reduce processing load."""
    height, width = frame.shape[:2]
    return cv2.resize(frame, (int(width * scale), int(height * scale)))

def main():
    try:
        # Start screen capture
        for frame in capture_screen(screen_region):
            # Downsample the frame to reduce CPU/GPU load
            frame = downsample_frame(frame, scale=0.5)

            # Submit pointing gesture detection to the thread pool
            future_gesture = executor.submit(detect_pointing_gesture, frame)
            pointing_detected, frame = future_gesture.result()

            if pointing_detected:
                print("Pointing gesture detected")

                # Submit canvas or book detection to the thread pool
                future_canvas = executor.submit(detect_canvas_or_book, frame)
                canvas_response = future_canvas.result()
                print(canvas_response["status"])

            # Detect faces in the frame
            frame, face_detected = detect_faces(frame)
            if face_detected:
                print("Face detected!")

            # Display the frame
            if not display_frame(frame):
                break

            # Control frame rate to reduce CPU usage
            time.sleep(0.05)

    finally:
        # Release resources
        release_resources()
        cv2.destroyAllWindows()
        executor.shutdown()

if __name__ == "__main__":
    main()

