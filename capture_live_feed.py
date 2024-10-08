import cv2
import numpy as np
import mss

def capture_screen(screen_region):
    with mss.mss() as sct:
        while True:
            # Capture the screen
            screenshot = sct.grab(screen_region)

            # Convert the captured screenshot to a numpy array (RGB)
            frame = np.array(screenshot)

            # Convert the frame to BGR (for OpenCV)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            yield frame  # Yield the frame to the caller

def display_frame(frame):
    # Display the frame
    cv2.imshow('Live Feed Capture', frame)

    # Press 'q' to exit the capture
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    return True

