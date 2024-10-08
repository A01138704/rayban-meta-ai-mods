import cv2

# Load the pre-trained face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame):
    """
    Detect faces in the provided frame using Haar Cascade.
    Args:
        frame (ndarray): The frame captured from the screen.
    Returns:
        frame (ndarray): The frame with rectangles drawn around detected faces.
        bool: Whether any face was detected.
    """
    # Convert the frame to grayscale (required for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Return the frame with faces highlighted and a boolean indicating if faces were detected
    return frame, len(faces) > 0

