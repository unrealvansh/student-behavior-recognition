"""Student Tracker using YOLO + ByteTrack.

Provides persistent tracking IDs across video frames so each student
keeps the same ID throughout the session. Uses COCO pretrained model
for robust person detection.
"""

from ultralytics import YOLO


class StudentTracker:
    def __init__(self, model_path="yolov8m.pt"):
        """Initialize with COCO pretrained model (person = class 0)."""
        self.model = YOLO(model_path)

    def track(self, frame, conf_threshold=0.3):
        """Track persons across frames with persistent IDs.

        Args:
            frame: BGR image (numpy array).
            conf_threshold: Minimum detection confidence.

        Returns:
            boxes: numpy array of [x1, y1, x2, y2] bounding boxes.
            ids: numpy array of persistent tracking IDs.
            confs: numpy array of confidence scores.
        """
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            verbose=False,
            conf=conf_threshold,
        )

        if results[0].boxes is None or results[0].boxes.id is None:
            return [], [], []

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()

        return boxes, ids, confs