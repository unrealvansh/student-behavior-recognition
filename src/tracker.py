from ultralytics import YOLO

class StudentTracker:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
    
    def track(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            verbose=False
        )
        
        if results[0].boxes.id is None:
            return [], [], []
        
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()
        
        return boxes, ids, confs