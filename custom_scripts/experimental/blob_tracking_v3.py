import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────
TOTAL_KEYPOINTS   = 18
MAX_DISAPPEARANCE = 50
EXCLUDE_TOPMOST   = 2
ASSIGN_DIST_THRESH = 100.0  # max pixels for assignment gating

# ────────────────────────────────────────────────────────────────────────────
# Blob Detector (your original settings)
# ────────────────────────────────────────────────────────────────────────────
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = True
params.blobColor = 255
params.minThreshold = 180
params.maxThreshold = 255
params.filterByArea = True
params.minArea = 1
params.filterByCircularity = True
params.minCircularity = 0.01
params.filterByInertia = True
params.minInertiaRatio = 0.01
params.filterByConvexity = True
params.minConvexity = 0.1
detector = cv2.SimpleBlobDetector_create(params)

# ────────────────────────────────────────────────────────────────────────────
# Kalman Filter wrapper for 2D position+velocity
# ────────────────────────────────────────────────────────────────────────────
class KF2D:
    def __init__(self, x, y):
        # state = [x, y, vx, vy], meas = [x, y]
        self.kf = cv2.KalmanFilter(4,2)
        self.kf.measurementMatrix = np.array([[1,0,0,0],
                                              [0,1,0,0]], np.float32)
        self.kf.transitionMatrix  = np.array([[1,0,1,0],
                                              [0,1,0,1],
                                              [0,0,1,0],
                                              [0,0,0,1]], np.float32)
        self.kf.processNoiseCov   = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        # init both pre & post
        self.kf.statePre  = np.array([[x],[y],[0],[0]], np.float32)
        self.kf.statePost = self.kf.statePre.copy()
    def predict(self):
        p = self.kf.predict()
        return (float(p[0]), float(p[1]))
    def correct(self, x, y):
        meas = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(meas)

# ────────────────────────────────────────────────────────────────────────────
# Tracker: holds 18 Kalman filters, positions, disappearance counters, exclusion
# ────────────────────────────────────────────────────────────────────────────
class KeypointTracker:
    def __init__(self):
        self.filters = [None]*TOTAL_KEYPOINTS
        self.positions = [None]*TOTAL_KEYPOINTS
        self.absent     = [0]*TOTAL_KEYPOINTS
        self.excluded   = []

    def initialize(self, detections):
        # first frame: exclude top‐2 by Y, then assign first 18
        sorted_by_y = sorted(detections, key=lambda p: p[1])
        self.excluded = sorted_by_y[:EXCLUDE_TOPMOST]
        valid = [p for p in detections if p not in self.excluded]
        for i, p in enumerate(valid[:TOTAL_KEYPOINTS]):
            self.filters[i]   = KF2D(*p)
            self.positions[i] = p
            self.absent[i]    = 0

    def predict_all(self):
        preds = []
        for i in range(TOTAL_KEYPOINTS):
            if self.filters[i] is not None and self.positions[i] is not None:
                preds.append(self.filters[i].predict())
            else:
                preds.append(None)
        return preds

    def update(self, detections, frame_num):
        # first frame
        if frame_num==0:
            self.initialize(detections)
            return

        # filter out excluded
        dets = [p for p in detections
                if all(np.linalg.norm(np.array(p)-np.array(e))>10
                       for e in self.excluded)]

        preds = self.predict_all()
        active_ids  = [i for i,p in enumerate(self.positions) if p is not None]
        active_preds= [preds[i] for i in active_ids]

        matched_ids = {}
        assigned_det = set()

        # Hungarian assignment on active → current
        if active_preds and dets:
            cost = np.zeros((len(active_preds), len(dets)), np.float32)
            for r, pr in enumerate(active_preds):
                for c, dt in enumerate(dets):
                    cost[r,c] = np.linalg.norm(np.array(pr)-np.array(dt))
            rows, cols = linear_sum_assignment(cost)
            for r,c in zip(rows,cols):
                if cost[r,c] < ASSIGN_DIST_THRESH:
                    tid = active_ids[r]
                    matched_ids[tid] = dets[c]
                    assigned_det.add(c)

        # correct filters for matched
        for tid, pt in matched_ids.items():
            x,y = pt
            self.filters[tid].correct(x,y)
            self.positions[tid] = pt
            self.absent[tid] = 0

        # mark unmatched active as absent
        for tid in active_ids:
            if tid not in matched_ids:
                self.absent[tid] += 1
                if self.absent[tid]>=MAX_DISAPPEARANCE:
                    self.positions[tid] = None
                    self.filters[tid]   = None

        # any unassigned detections become new tracks if slots free
        unmatched = [dets[i] for i in range(len(dets)) if i not in assigned_det]
        for pt in unmatched:
            for i in range(TOTAL_KEYPOINTS):
                if self.positions[i] is None:
                    self.filters[i]   = KF2D(*pt)
                    self.positions[i] = pt
                    self.absent[i]    = 0
                    break

    def get_positions(self):
        # if a track is missing, we fallback to its prediction for drawing
        preds = self.predict_all()
        out = []
        for i in range(TOTAL_KEYPOINTS):
            if self.positions[i] is not None:
                out.append(self.positions[i])
            else:
                out.append(preds[i])
        return out

# ────────────────────────────────────────────────────────────────────────────
# Main loop: capture, detect, track, draw, save
# ────────────────────────────────────────────────────────────────────────────
tracker = KeypointTracker()
cap = cv2.VideoCapture(r'C:\Users\K Nanditha\Desktop\capstone\pjt-2\dataset\blob_detection\ankith-code\pranav-2.mp4')
fps    = cap.get(cv2.CAP_PROP_FPS)
w, h   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter('stable_tracking.mp4', fourcc, fps, (w,h))

frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret: break

    gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kps       = detector.detect(gray)
    det_pts   = [(kp.pt[0], kp.pt[1]) for kp in kps]

    tracker.update(det_pts, frame_num)
    pts = tracker.get_positions()

    # draw
    valid = 0
    for i, p in enumerate(pts):
        if p is None: continue
        x,y = int(p[0]), int(p[1])
        cv2.circle(frame,(x,y),4,(0,255,0),-1)
        cv2.putText(frame,str(i),(x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
        valid += 1

    cv2.putText(frame,
        f"Frame {frame_num+1} | Valid: {valid}",
        (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    out.write(frame)
    cv2.imshow("Stable Keypoint Tracking", frame)
    if cv2.waitKey(1)&0xFF in (27, ord('q')): break

    frame_num += 1

cap.release()
out.release()
cv2.destroyAllWindows()
