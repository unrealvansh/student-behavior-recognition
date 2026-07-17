import cv2
import numpy as np
from ultralytics import YOLO

class BehaviorRecognitionPipeline:
    def __init__(self):
        self.model = YOLO("C:/Users/Unrea/OneDrive/Desktop/project/runs/detect/models/yolo_behavior_v2-4/weights/best.pt")
        
        # Distinct colors for each behavior (BGR format)
        self.colors = {
            'hand-raising': (0, 165, 255),          # Orange
            'reading': (0, 255, 0),                 # Green
            'writing': (255, 200, 0),               # Cyan-blue
            'sing-phone': (0, 0, 255),              # Red
            'bowing-the-head': (255, 0, 255),       # Magenta
            'leaning-over-the-table': (0, 255, 255) # Yellow
        }
    
    def _nms(self, boxes, classes, confs, iou_thresh=0.4):
        """Non-maximum suppression to remove overlapping detections."""
        if len(boxes) == 0:
            return [], [], []
        
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
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h
            
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            # Also suppress if smaller box is mostly inside larger box
            min_areas = np.minimum(areas[i], areas[order[1:]])
            overlap_ratio = intersection / (min_areas + 1e-6)
            
            mask = (iou <= iou_thresh) & (overlap_ratio <= 0.6)
            order = order[1:][mask]
        
        return (
            [boxes[i] for i in keep],
            [classes[i] for i in keep],
            [confs[i] for i in keep]
        )
    
    def _sliced_inference(self, frame, conf=0.15, slice_size=640, overlap=0.25):
        """Run detection on overlapping tiles of the image for small object detection."""
        h, w = frame.shape[:2]
        stride = int(slice_size * (1 - overlap))
        
        all_boxes = []
        all_classes = []
        all_confs = []
        
        # Also run on the full image for large objects
        results = self.model(frame, verbose=False, conf=conf, imgsz=640)
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            cls = results[0].boxes.cls.cpu().numpy().astype(int)
            cfs = results[0].boxes.conf.cpu().numpy()
            for b, c, f in zip(boxes, cls, cfs):
                all_boxes.append(b.tolist())
                all_classes.append(int(c))
                all_confs.append(float(f))
        
        # Run on tiles
        for y_start in range(0, h, stride):
            for x_start in range(0, w, stride):
                x_end = min(x_start + slice_size, w)
                y_end = min(y_start + slice_size, h)
                
                # Skip tiny edge slices
                if (x_end - x_start) < slice_size * 0.5 or (y_end - y_start) < slice_size * 0.5:
                    continue
                
                tile = frame[y_start:y_end, x_start:x_end]
                
                results = self.model(tile, verbose=False, conf=conf, imgsz=640)
                
                if results[0].boxes is None or len(results[0].boxes) == 0:
                    continue
                
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                cls = results[0].boxes.cls.cpu().numpy().astype(int)
                cfs = results[0].boxes.conf.cpu().numpy()
                
                for b, c, f in zip(boxes, cls, cfs):
                    # Map tile coordinates back to full image
                    global_box = [
                        b[0] + x_start,
                        b[1] + y_start,
                        b[2] + x_start,
                        b[3] + y_start
                    ]
                    all_boxes.append(global_box)
                    all_classes.append(int(c))
                    all_confs.append(float(f))
        
        # Merge all detections with NMS
        all_boxes, all_classes, all_confs = self._nms(all_boxes, all_classes, all_confs)
        
        return all_boxes, all_classes, all_confs
    
    def process(self, frame):
        names = self.model.names
        
        stats = {
            'total': 0,
            'hand-raising': 0, 'reading': 0, 'writing': 0,
            'sing-phone': 0, 'bowing-the-head': 0, 'leaning-over-the-table': 0,
            'students': []
        }
        
        # Use sliced inference to detect small students
        boxes, classes, confs = self._sliced_inference(frame, conf=0.15)
        
        for box, cls, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = box
            behavior = names[cls]
            color = self.colors.get(behavior, (255, 255, 255))
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background
            label = f"{behavior} ({conf:.0%})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            stats['students'].append({
                'id': len(stats['students']) + 1,
                'behavior': behavior,
                'confidence': float(conf)
            })
            stats[behavior] = stats.get(behavior, 0) + 1
            stats['total'] += 1
        
        return frame, stats