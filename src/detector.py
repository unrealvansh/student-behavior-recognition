from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path="models/yolo_detector.pt"):
        self.model = YOLO(model_path)
    
    def detect(self, frame, conf_threshold=0.5, min_area=8000):
        results = self.model(frame, classes=[0], verbose=False, conf=conf_threshold)
        
        if results[0].boxes is None or len(results[0].boxes) == 0:
            return []
        
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        filtered = []
        
        for box in boxes:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            if area > min_area:
                filtered.append(box)
        
        return filtered