from ultralytics import YOLO
import numpy as np

class PersonDetector:
    def __init__(self, model_path="yolov8m.pt"):
        # COCO pretrained — generalizes to all images, person = class 0
        self.model = YOLO(model_path)
    
    def _suppress_duplicates(self, boxes, overlap_thresh=0.5):
        """Remove duplicate/nested boxes that YOLO NMS misses.
        If a smaller box overlaps significantly with a larger box, drop it."""
        if len(boxes) <= 1:
            return boxes
        
        areas = [(b[2]-b[0]) * (b[3]-b[1]) for b in boxes]
        # Sort by area descending (keep larger boxes)
        order = np.argsort(areas)[::-1]
        boxes = [boxes[i] for i in order]
        areas = [areas[i] for i in order]
        
        keep = []
        suppressed = set()
        
        for i in range(len(boxes)):
            if i in suppressed:
                continue
            keep.append(boxes[i])
            
            for j in range(i + 1, len(boxes)):
                if j in suppressed:
                    continue
                # Calculate overlap between box i (larger) and box j (smaller)
                x1 = max(boxes[i][0], boxes[j][0])
                y1 = max(boxes[i][1], boxes[j][1])
                x2 = min(boxes[i][2], boxes[j][2])
                y2 = min(boxes[i][3], boxes[j][3])
                
                if x1 < x2 and y1 < y2:
                    intersection = (x2 - x1) * (y2 - y1)
                    # If intersection covers >50% of the smaller box, it's a duplicate
                    if intersection / areas[j] > overlap_thresh:
                        suppressed.add(j)
        
        return keep
    
    def detect(self, frame, conf_threshold=0.25, iou_threshold=0.3, min_area=8000):
        results = self.model(frame, classes=[0], verbose=False, conf=conf_threshold, iou=iou_threshold)
        
        if results[0].boxes is None or len(results[0].boxes) == 0:
            return []
        
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        filtered = []
        
        for box in boxes:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            if area > min_area:
                filtered.append(box)
        
        # Remove duplicate/nested boxes
        filtered = self._suppress_duplicates(filtered)
        
        return filtered