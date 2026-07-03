from src.detector import PersonDetector
from src.tracker import StudentTracker
from src.classifier import BehaviorClassifier
import cv2
class BehaviorRecognitionPipeline:
    def __init__(self):
        self.detector = PersonDetector()
        self.tracker = StudentTracker()
        self.classifier = BehaviorClassifier()
        
        self.colors = {
            "Sitting": (128, 128, 128),
            "Sleeping": (0, 0, 255),
            "Mobile": (0, 255, 255),
            "Reading": (0, 255, 0),
            "HandRaising": (255, 0, 0),
            "Unknown": (255, 255, 255)
        }
    
    def process(self, frame):
        boxes, ids, confs = self.tracker.track(frame)
        
        stats = {
            "total": len(ids),
            "Sitting": 0, "Sleeping": 0, "Mobile": 0,
            "Reading": 0, "HandRaising": 0,
            "students": []
        }
        
        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            crop = frame[y1:y2, x1:x2]
            behavior, confidence = self.classifier.predict(crop)
            
            color = self.colors.get(behavior, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"#{track_id} {behavior} ({confidence:.0%})"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            stats["students"].append({
                "id": int(track_id),
                "behavior": behavior,
                "confidence": float(confidence)
            })
            stats[behavior] = stats.get(behavior, 0) + 1
        
        return frame, stats