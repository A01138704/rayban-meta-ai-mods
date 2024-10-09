import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

pushup_counter = 0  # Initialize the push-up counter
pushup_position = None  # Track whether the body is currently "up" or "down"

def calculate_distance(point1, point2):
    """
    Calculate the Euclidean distance between two 2D points.
    Args:
        point1 (tuple): (x1, y1) coordinates of the first point
        point2 (tuple): (x2, y2) coordinates of the second point
    Returns:
        float: Euclidean distance between the two points
    """
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

def detect_pose_and_count_pushups(frame):
    """
    Detect body landmarks and count push-ups based on the relative movement of the body.
    Args:
        frame (ndarray): The input frame from the camera or live feed.
    Returns:
        frame (ndarray): The frame with pose landmarks and distance displayed.
        int: The current count of push-ups.
    """
    global pushup_counter, pushup_position  # Use global variables to track push-ups
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get landmarks for the nose (as a proxy for push-up depth)
        landmarks = result.pose_landmarks.landmark
        
        # Get the nose's vertical position
        nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y

        # Simple logic: detect when the user goes down and comes back up
        if pushup_position == "up" and nose_y > 0.6:  # The body is going down (arbitrary threshold)
            pushup_position = "down"
        elif pushup_position == "down" and nose_y < 0.4:  # The body is coming back up
            pushup_position = "up"
            pushup_counter += 1  # Count a completed push-up

        # Display push-up count on the frame
        cv2.putText(frame, f'Push-ups: {pushup_counter}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame, pushup_counter

