import sys
import os
import cv2

# Adjust the path to find capture_live_feed.py in the root of rayban-meta-ai-mods
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from capture_live_feed import capture_screen, display_frame  # Import from root directory
from pushup_pose_detection import detect_pose_and_count_pushups  # Import from the same folder

# Define the region of the screen you want to capture
screen_region = {'top': 100, 'left': 100, 'width': 800, 'height': 600}

def main():
    # Capture frames from the live feed (screen)
    for frame in capture_screen(screen_region):
        # Process the frame for pose detection and push-up counting
        frame, pushup_count = detect_pose_and_count_pushups(frame)

        # Display the processed frame with push-up count
        if not display_frame(frame):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

