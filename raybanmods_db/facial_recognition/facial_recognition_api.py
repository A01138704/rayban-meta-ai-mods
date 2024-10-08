from .face_detection import detect_faces  # Ensure the face detection logic is imported correctly

def facial_recognition_api(frame):
    """
    Handle the facial recognition logic and provide a response.
    Args:
        frame: The frame captured from live feed.
    Returns:
        dict: Status indicating whether a face was detected.
    """
    frame, face_detected = detect_faces(frame)
    if face_detected:
        return {"status": "Face detected"}
    else:
        return {"status": "No face detected"}

