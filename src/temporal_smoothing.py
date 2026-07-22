"""Temporal Smoothing for Video Detection.

Reduces flickering/jittering in video by:
1. Smoothing bounding box positions across frames (EMA)
2. Requiring detections to persist for N frames before showing
3. Keeping detections alive for N frames after they disappear

Completely independent of the classifier — works purely on
detection boxes and optional labels.
"""

import numpy as np
from collections import defaultdict


class TemporalSmoother:
    """Smooth detections across video frames to reduce flicker."""

    def __init__(self, smoothing_alpha=0.6, min_hits=3, max_age=5,
                 iou_match_thresh=0.3):
        """
        Args:
            smoothing_alpha: EMA weight for box position (higher = more responsive,
                             lower = smoother). Range [0, 1].
            min_hits: Number of consecutive frames a detection must appear
                      before it's shown (prevents flash detections).
            max_age: Number of frames a detection survives after disappearing
                     (prevents flicker when detection drops for 1-2 frames).
            iou_match_thresh: Minimum IoU to match a new detection to an
                              existing tracked box.
        """
        self.alpha = smoothing_alpha
        self.min_hits = min_hits
        self.max_age = max_age
        self.iou_match_thresh = iou_match_thresh

        # Tracked objects: id -> {box, hits, age, label, conf}
        self.tracks = {}
        self._next_id = 0

    def update(self, boxes, confs=None, labels=None):
        """Update tracks with new detections and return smoothed results.

        Args:
            boxes: List of [x1, y1, x2, y2] bounding boxes.
            confs: Optional list of confidence scores.
            labels: Optional list of label strings (for classifier output).

        Returns:
            smoothed_boxes: List of [x1, y1, x2, y2] smoothed boxes.
            smoothed_confs: List of confidence scores.
            smoothed_labels: List of labels (or None if no labels provided).
            track_ids: List of consistent track IDs for each detection.
        """
        if confs is None:
            confs = [1.0] * len(boxes)
        if labels is None:
            labels = [None] * len(boxes)

        # Match new detections to existing tracks
        matched, unmatched_dets, unmatched_tracks = self._match(boxes)

        # Update matched tracks
        for det_idx, track_id in matched:
            track = self.tracks[track_id]
            new_box = np.array(boxes[det_idx], dtype=np.float32)

            # Exponential moving average for smooth box positions
            track["box"] = (self.alpha * new_box +
                            (1 - self.alpha) * track["box"])
            track["hits"] += 1
            track["age"] = 0
            track["conf"] = confs[det_idx]
            if labels[det_idx] is not None:
                track["label"] = labels[det_idx]

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self.tracks[self._next_id] = {
                "box": np.array(boxes[det_idx], dtype=np.float32),
                "hits": 1,
                "age": 0,
                "conf": confs[det_idx],
                "label": labels[det_idx],
            }
            self._next_id += 1

        # Age unmatched tracks
        for track_id in unmatched_tracks:
            self.tracks[track_id]["age"] += 1

        # Remove dead tracks
        dead = [tid for tid, t in self.tracks.items()
                if t["age"] > self.max_age]
        for tid in dead:
            del self.tracks[tid]

        # Return only confirmed tracks (enough consecutive hits)
        out_boxes = []
        out_confs = []
        out_labels = []
        out_ids = []

        for tid, track in self.tracks.items():
            if track["hits"] >= self.min_hits:
                out_boxes.append(track["box"].astype(int).tolist())
                out_confs.append(track["conf"])
                out_labels.append(track["label"])
                out_ids.append(tid)

        return out_boxes, out_confs, out_labels, out_ids

    def reset(self):
        """Clear all tracks (e.g., when switching video sources)."""
        self.tracks.clear()
        self._next_id = 0

    def _match(self, boxes):
        """Match new detections to existing tracks using IoU.

        Returns:
            matched: List of (det_idx, track_id) pairs.
            unmatched_dets: List of detection indices with no match.
            unmatched_tracks: List of track IDs with no match.
        """
        if not self.tracks or not boxes:
            return [], list(range(len(boxes))), list(self.tracks.keys())

        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid]["box"] for tid in track_ids]

        # Build IoU matrix
        iou_matrix = np.zeros((len(boxes), len(track_boxes)))
        for d, det_box in enumerate(boxes):
            for t, trk_box in enumerate(track_boxes):
                iou_matrix[d, t] = self._iou(det_box, trk_box)

        # Greedy matching (highest IoU first)
        matched = []
        used_dets = set()
        used_tracks = set()

        while True:
            if iou_matrix.size == 0:
                break
            max_iou = iou_matrix.max()
            if max_iou < self.iou_match_thresh:
                break

            d, t = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            matched.append((d, track_ids[t]))
            used_dets.add(d)
            used_tracks.add(t)

            # Zero out matched row and column
            iou_matrix[d, :] = 0
            iou_matrix[:, t] = 0

        unmatched_dets = [d for d in range(len(boxes)) if d not in used_dets]
        unmatched_tracks = [track_ids[t] for t in range(len(track_ids))
                           if t not in used_tracks]

        return matched, unmatched_dets, unmatched_tracks

    @staticmethod
    def _iou(box_a, box_b):
        """Calculate IoU between two boxes [x1, y1, x2, y2]."""
        box_a = np.array(box_a, dtype=np.float32)
        box_b = np.array(box_b, dtype=np.float32)

        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        if x1 >= x2 or y1 >= y2:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        return intersection / (area_a + area_b - intersection + 1e-6)
