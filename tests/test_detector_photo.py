import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
from src.detector import PersonDetector

# Load your trained YOLO model
detector = PersonDetector("models/yolo_detector.pt")

# Path to your classroom photo (change this to your actual image path)
# Option 1: Photo in project root folder
# IMAGE_PATH = "test_photo.jpg"

# Option 2: Full path to photo anywhere on your computer
IMAGE_PATH = r"C:\Users\Unrea\OneDrive\Desktop\project\img124.jpg"

# Load image
img = cv2.imread(IMAGE_PATH)

if img is None:
    print(f"ERROR: Could not load image from {IMAGE_PATH}")
    print("Check the path and make sure the file exists.")
    exit()

print(f"Image loaded: {img.shape[1]}x{img.shape[0]}")

# Detect people
boxes = detector.detect(img)
print(f"Found {len(boxes)} people")

# Draw green boxes
for box in boxes:
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Optional: add confidence label
    label = "person"
    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Save result
output_path = "output_detection.jpg"
cv2.imwrite(output_path, img)
print(f"Saved result to: {output_path}")

# Show result
cv2.imshow("YOLO Detection Result", img)
print("Press any key to close window...")
cv2.waitKey(0)
cv2.destroyAllWindows()
