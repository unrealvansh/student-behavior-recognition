"""Auto-crop full-frame images to person crops using YOLO.

Runs YOLO on each image in a folder, detects persons, and saves
each person crop as a separate image. Replaces the original images.
"""

import os
import sys
import cv2
from ultralytics import YOLO

def auto_crop_persons(folder, output_folder=None, model_path="yolov8m.pt"):
    """Detect persons in images and save crops.
    
    Args:
        folder: Path to folder containing full-frame images.
        output_folder: Where to save crops. If None, replaces originals.
        model_path: YOLO model to use for detection.
    """
    if output_folder is None:
        output_folder = folder + "_cropped"
    
    os.makedirs(output_folder, exist_ok=True)
    
    model = YOLO(model_path)
    
    files = [f for f in os.listdir(folder) 
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Processing {len(files)} images from: {folder}")
    print(f"Saving crops to: {output_folder}\n")
    
    total_crops = 0
    
    for fname in files:
        img_path = os.path.join(folder, fname)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"  SKIP (unreadable): {fname}")
            continue
        
        h, w = img.shape[:2]
        
        # Only process large images (likely full-frame)
        # Small images are probably already cropped
        if max(h, w) < 300:
            # Already a crop — copy as-is
            base = os.path.splitext(fname)[0]
            out_path = os.path.join(output_folder, f"{base}.jpg")
            cv2.imwrite(out_path, img)
            total_crops += 1
            continue
        
        # Run YOLO detection
        results = model(img, classes=[0], verbose=False, conf=0.3)
        
        if results[0].boxes is None or len(results[0].boxes) == 0:
            print(f"  NO PERSON: {fname}")
            continue
        
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        
        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            # Add small padding (5%)
            pad_x = int((x2 - x1) * 0.05)
            pad_y = int((y2 - y1) * 0.05)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            
            crop = img[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
            
            base = os.path.splitext(fname)[0]
            suffix = f"_p{j}" if len(boxes) > 1 else ""
            out_path = os.path.join(output_folder, f"{base}{suffix}.jpg")
            cv2.imwrite(out_path, crop)
            total_crops += 1
    
    print(f"\nDone! Saved {total_crops} person crops to: {output_folder}")
    return total_crops


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "data/behavior/mobile"
    auto_crop_persons(folder)
