"""Test person detector using COCO-pretrained YOLOv8m.
No custom training needed — COCO already has 'person' as class 0,
trained on millions of diverse images. Works on ANY image.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from ultralytics import YOLO

# Use COCO pretrained model — it already detects persons perfectly
# No custom training needed, generalizes to ALL images
model = YOLO("yolov8m.pt")

# Path to test image — change this to test different images
IMAGE_PATH = r"C:\Users\Unrea\OneDrive\Desktop\project\img124.jpg"

img = cv2.imread(IMAGE_PATH)

if img is None:
    print(f"ERROR: Could not load image from {IMAGE_PATH}")
    sys.exit(1)

h, w = img.shape[:2]
print(f"Image loaded: {w}x{h}")
print(f"Model: COCO-pretrained YOLOv8m\n")


def nms_merge(boxes, confs, iou_thresh=0.4):
    """Merge overlapping boxes from tiled + full inference."""
    if len(boxes) == 0:
        return [], []

    boxes_arr = np.array(boxes, dtype=np.float32)
    confs_arr = np.array(confs, dtype=np.float32)

    x1 = boxes_arr[:, 0]
    y1 = boxes_arr[:, 1]
    x2 = boxes_arr[:, 2]
    y2 = boxes_arr[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = confs_arr.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w_i = np.maximum(0, xx2 - xx1)
        h_i = np.maximum(0, yy2 - yy1)
        intersection = w_i * h_i

        iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-6)
        min_areas = np.minimum(areas[i], areas[order[1:]])
        overlap_ratio = intersection / (min_areas + 1e-6)

        mask = (iou <= iou_thresh) & (overlap_ratio <= 0.6)
        order = order[1:][mask]

    return [boxes[i] for i in keep], [confs[i] for i in keep]


def detect_persons(model, frame, conf=0.3, tile_size=640, overlap=0.25):
    """Robust person detection with sliced inference.
    Uses classes=[0] to only detect persons from the 80 COCO classes.
    """
    h, w = frame.shape[:2]
    stride = int(tile_size * (1 - overlap))

    all_boxes = []
    all_confs = []

    # 1) Full-image inference — catches large/medium persons
    results = model(frame, verbose=False, conf=conf, iou=0.3, imgsz=640, classes=[0])
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            b = box.xyxy[0].cpu().numpy().astype(int).tolist()
            all_boxes.append(b)
            all_confs.append(float(box.conf[0]))

    # 2) Tiled inference — catches small persons at the back
    for y_start in range(0, h, stride):
        for x_start in range(0, w, stride):
            x_end = min(x_start + tile_size, w)
            y_end = min(y_start + tile_size, h)

            if (x_end - x_start) < tile_size * 0.5 or (y_end - y_start) < tile_size * 0.5:
                continue

            tile = frame[y_start:y_end, x_start:x_end]
            results = model(tile, verbose=False, conf=conf, iou=0.3, imgsz=640, classes=[0])

            if results[0].boxes is None or len(results[0].boxes) == 0:
                continue

            for box in results[0].boxes:
                b = box.xyxy[0].cpu().numpy().astype(int)
                global_box = [
                    b[0] + x_start, b[1] + y_start,
                    b[2] + x_start, b[3] + y_start
                ]
                all_boxes.append(global_box)
                all_confs.append(float(box.conf[0]))

    # 3) Merge duplicates
    all_boxes, all_confs = nms_merge(all_boxes, all_confs, iou_thresh=0.4)

    return all_boxes, all_confs


# Run detection
boxes, confs = detect_persons(model, img, conf=0.3)

print(f"✅ Detected {len(boxes)} person(s)\n")

for i, (box, conf) in enumerate(zip(boxes, confs)):
    x1, y1, x2, y2 = box

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    label = f"person {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
    cv2.putText(img, label, (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    print(f"  Person {i+1}: ({x1},{y1})-({x2},{y2})  conf={conf:.1%}")

# Save result
output_path = "output_person_test.jpg"
cv2.imwrite(output_path, img)
print(f"\nSaved result to: {output_path}")

cv2.imshow("Person Detection Test", img)
print("Press any key to close window...")
cv2.waitKey(0)
cv2.destroyAllWindows()
