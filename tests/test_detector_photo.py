import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
from src.full_pipeline import SmartPipeline

# Use the new SmartPipeline (YOLOv8 + ByteTrack + MobileNetV2)
pipeline = SmartPipeline(
    classifier_weights="models/behavior_classifier.pth",
    tracker_model="yolov8n.pt",
    conf_threshold=0.3
)

# Path to your classroom photo
IMAGE_PATH = r"C:\Users\Unrea\OneDrive\Desktop\project\people-sitting-with-mobile-phones-in-classroom-2J3FGMT.jpg"


# Load image
img = cv2.imread(IMAGE_PATH)

if img is None:
    print(f"ERROR: Could not load image from {IMAGE_PATH}")
    print("Check the path and make sure the file exists.")
    exit()

print(f"Image loaded: {img.shape[1]}x{img.shape[0]}")

# Detect behaviors
annotated_frame, stats, detections = pipeline.process_frame(img)

print(f"\nTotal detections: {stats['total']}")
for behavior in ['sitting', 'reading', 'hand', 'phone', 'sleeping']:
    count = stats.get(behavior, 0)
    if count > 0:
        print(f"  {behavior.capitalize()}: {count}")

# Save result
output_path = "output_detection.jpg"
cv2.imwrite(output_path, annotated_frame)
print(f"\nSaved result to: {output_path}")

# Show result
cv2.imshow("Behavior Detection Result", annotated_frame)
print("Press any key to close window...")
cv2.waitKey(0)
cv2.destroyAllWindows()
