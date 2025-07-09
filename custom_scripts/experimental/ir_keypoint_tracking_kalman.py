import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

TOTAL_KEYPOINTS = 16
MAX_DISAPPEARANCE = 50
EXCLUDE_TOPMOST = 0
ASSIGN_DIST_THRESH = 100.0  # pixels

# ─────────────────────────────────────
# Kalman Filter for 2D keypoint tracking
# ─────────────────────────────────────
class KF2D:
    def __init__(self, x, y):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                              [0, 1, 0, 1],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
        self.kf.statePost = self.kf.statePre.copy()

    def predict(self):
        p = self.kf.predict()
        return (float(p[0]), float(p[1]))

    def correct(self, x, y):
        meas = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(meas)

# ─────────────────────────────────────
# Keypoint Tracker Class
# ─────────────────────────────────────
class KeypointTracker:
    def __init__(self):
        self.filters = [None] * TOTAL_KEYPOINTS
        self.positions = [None] * TOTAL_KEYPOINTS
        self.absent = [0] * TOTAL_KEYPOINTS
        self.excluded = []

    def initialize(self, detections):
        sorted_by_y = sorted(detections, key=lambda p: p[1])
        self.excluded = sorted_by_y[:EXCLUDE_TOPMOST]
        valid = [p for p in detections if p not in self.excluded]
        for i, p in enumerate(valid[:TOTAL_KEYPOINTS]):
            self.filters[i] = KF2D(*p)
            self.positions[i] = p
            self.absent[i] = 0

    def predict_all(self):
        return [
            self.filters[i].predict() if self.filters[i] and self.positions[i] else None
            for i in range(TOTAL_KEYPOINTS)
        ]

    def update(self, detections, frame_num):
        if frame_num == 0:
            self.initialize(detections)
            return

        filtered_detections = [
            p for p in detections
            if all(np.linalg.norm(np.array(p) - np.array(e)) > 10 for e in self.excluded)
        ]

        preds = self.predict_all()
        active_ids = [i for i, p in enumerate(self.positions) if p is not None]
        active_preds = [preds[i] for i in active_ids]

        matched_ids = {}
        assigned = set()

        if active_preds and filtered_detections:
            cost = np.zeros((len(active_preds), len(filtered_detections)), np.float32)
            for r, pr in enumerate(active_preds):
                for c, dt in enumerate(filtered_detections):
                    cost[r, c] = np.linalg.norm(np.array(pr) - np.array(dt))
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] < ASSIGN_DIST_THRESH:
                    tid = active_ids[r]
                    matched_ids[tid] = filtered_detections[c]
                    assigned.add(c)

        for tid, pt in matched_ids.items():
            x, y = pt
            self.filters[tid].correct(x, y)
            self.positions[tid] = pt
            self.absent[tid] = 0

        for tid in active_ids:
            if tid not in matched_ids:
                self.absent[tid] += 1
                if self.absent[tid] >= MAX_DISAPPEARANCE:
                    self.positions[tid] = None
                    self.filters[tid] = None

        unmatched = [filtered_detections[i] for i in range(len(filtered_detections)) if i not in assigned]
        for pt in unmatched:
            for i in range(TOTAL_KEYPOINTS):
                if self.positions[i] is None:
                    self.filters[i] = KF2D(*pt)
                    self.positions[i] = pt
                    self.absent[i] = 0
                    break

    def get_positions(self):
        return [
            self.positions[i] if self.positions[i] else (
                self.filters[i].predict() if self.filters[i] else None
            )
            for i in range(TOTAL_KEYPOINTS)
        ]
