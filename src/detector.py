from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
    
    def detect(self, frame):
        results = self.model(frame, classes=[0], verbose=False)
        return results[0].boxes.xyxy.cpu().numpy().astype(int)