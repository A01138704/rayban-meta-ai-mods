import mss
import numpy as np
import cv2

def capture_screen(region):
    """
    Capture a region of the screen using mss.
    Args:
        region (dict): The region of the screen to capture (top, left, width, height).
    Yields:
        frame (ndarray): The captured frame.
    """
    with mss.mss() as sct:
        while True:
            # Grab the screen
            screenshot = sct.grab(region)
            # Convert screenshot to a numpy array
            frame = np.array(screenshot)
            # Convert from BGRA to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            yield frame

def display_frame(frame):
    """
    Display the frame and check if the user wants to quit.
    Args:
        frame (ndarray): The frame to display.
    Returns:
        bool: False if the user pressed 'q' to quit, otherwise True.
    """
    cv2.imshow('Live Feed', frame)
    # Check if 'q' is pressed to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    return True

