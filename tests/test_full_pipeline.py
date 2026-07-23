"""Test full two-stage pipeline: Person Detection + Behavior Classification.

Stage 1: YOLO (COCO pretrained) detects all persons
Stage 2: MobileNetV2 classifies each person's behavior
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
from src.detector import PersonDetector
from src.classifier import BehaviorClassifier

# ── Load models ──────────────────────────────────────────────────
detector = PersonDetector("yolov8m.pt")
classifier = BehaviorClassifier(weights_path="models/behavior_classifier.pth")

print(f"Detector: COCO YOLOv8m")
print(f"Classifier: MobileNetV2 ({classifier.class_names})")
print()

# ── Load test image ──────────────────────────────────────────────
IMAGE_PATH = r"C:\Users\Unrea\Downloads\img29.jpg"
img = cv2.imread(IMAGE_PATH)

if img is None:
    print(f"ERROR: Could not load image from {IMAGE_PATH}")
    sys.exit(1)

print(f"Image: {img.shape[1]}x{img.shape[0]}")

# ── Stage 1: Detect persons ─────────────────────────────────────
boxes, confs = detector.detect(img, conf_threshold=0.3, use_sliced=True)
print(f"Detected: {len(boxes)} person(s)\n")

# ── Stage 2: Classify each person's behavior ────────────────────
COLORS = {
    "Sitting": (0, 255, 0),        # Green
    "Sleeping": (0, 0, 255),       # Red
    "Mobile": (0, 165, 255),       # Orange
    "Reading": (255, 200, 0),      # Cyan
    "HandRaising": (255, 0, 255),  # Magenta
}

stats = {}
for i, (box, det_conf) in enumerate(zip(boxes, confs)):
    x1, y1, x2, y2 = box

    # Crop the person
    crop = img[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        continue

    # Classify behavior
    behavior, cls_conf, all_probs = classifier.predict(crop, return_all_probs=True)
    stats[behavior] = stats.get(behavior, 0) + 1

    # Draw
    color = COLORS.get(behavior, (255, 255, 255))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    label = f"{behavior} ({cls_conf:.0%})"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, label, (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    prob_str = ", ".join([f"{k}:{v:.0%}" for k, v in all_probs.items()])
    print(f"  Person {i+1}: {behavior} ({cls_conf:.0%})  |  det={det_conf:.0%}  |  Probs: [{prob_str}]")

# ── Summary ──────────────────────────────────────────────────────
print(f"\n--- Summary ---")
for behavior, count in sorted(stats.items()):
    print(f"  {behavior}: {count}")
print(f"  Total: {sum(stats.values())}")

# ── Save & show ──────────────────────────────────────────────────
output_path = "output_full_pipeline.jpg"
cv2.imwrite(output_path, img)
print(f"\nSaved to: {output_path}")

cv2.imshow("Full Pipeline Test", img)
print("Press any key to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()
