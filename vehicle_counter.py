import cv2
import numpy as np
from time import sleep

# Minimum width and height of a valid object
MIN_WIDTH = 80
MIN_HEIGHT = 80

# Allowed error for considering an object crossing the line
OFFSET = 6

# Y-coordinate of the counting line
LINE_POSITION = 550

# Delay between frames (controls frame rate)
FRAME_RATE = 60

# Lists to store detected objects and count vehicles
detected_objects = []
vehicle_count = 0

def get_center(x, y, w, h):
    """Calculates the center coordinates of a bounding box."""
    cx = x + w // 2
    cy = y + h // 2
    return cx, cy

cap = cv2.VideoCapture('video.mp4')
background_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret, frame = cap.read()
    if not ret:  # Handle end of video
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 5)
    foreground_mask = background_subtractor.apply(blur)
    dilated_mask = cv2.dilate(foreground_mask, np.ones((5, 5)))

    # Morphological closing to fill holes in objects
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the counting line
    cv2.line(frame, (25, LINE_POSITION), (1200, LINE_POSITION), (255, 127, 0), 3) 

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Filter out small objects
        if w >= MIN_WIDTH and h >= MIN_HEIGHT:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
            center = get_center(x, y, w, h)
            detected_objects.append(center)
            cv2.circle(frame, center, 4, (0, 0, 255), -1)  # Draw object center

            for center_x, center_y in detected_objects:
                if (LINE_POSITION - OFFSET) < center_y < (LINE_POSITION + OFFSET):
                    vehicle_count += 1
                    cv2.line(frame, (25, LINE_POSITION), (1200, LINE_POSITION), (0, 127, 255), 3)  # Change line color
                    detected_objects.remove((center_x, center_y))
                    print("Car detected:", vehicle_count)

    # Cleanup old detections
    detected_objects = [(cx, cy) for cx, cy in detected_objects if cy < (LINE_POSITION + OFFSET)]

    # Display vehicle count on the frame
    cv2.putText(frame, "VEHICLE COUNT: " + str(vehicle_count), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    cv2.imshow("Video Original", frame)
    cv2.imshow("Detectar", closed_mask)

    if cv2.waitKey(1) == 27:  # Exit on 'ESC'
        break

cv2.destroyAllWindows()
cap.release()
print(vehicle_count)