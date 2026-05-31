import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
from src.detector import PersonDetector

detector = PersonDetector()

# Use 0 for webcam, or "video.mp4" for a file
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit()

print("Camera opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    boxes = detector.detect(frame)
    
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imshow("Test - Person Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()