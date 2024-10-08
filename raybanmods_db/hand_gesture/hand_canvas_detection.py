import cv2

def detect_canvas_or_book(frame):
    """
    Detect a canvas or book (rectangular objects) in the frame using contour detection.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Approximate the contour to a polygon
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

        # If the polygon has 4 points, it's likely a rectangular object (canvas, screen, book)
        if len(approx) == 4:
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
            return {"status": "Canvas or book detected", "contour": approx}
    
    return {"status": "No canvas or book detected"}

