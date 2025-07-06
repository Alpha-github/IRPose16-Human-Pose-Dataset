### === FILE: ir_keypoint_tracking_kalman.py ===
import numpy as np
import cv2
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

##### KALMAN FILTER #########
# class KF2D:
#     def __init__(self):
#         self.kf = KalmanFilter(dim_x=4, dim_z=2)
#         self.kf.F = np.array([[1, 0, 1, 0],
#                               [0, 1, 0, 1],
#                               [0, 0, 1, 0],
#                               [0, 0, 0, 1]])
#         self.kf.H = np.array([[1, 0, 0, 0],
#                               [0, 1, 0, 0]])
#         self.kf.R *= 0.5
#         self.kf.P *= 10
#         self.kf.Q *= 0.01

#     def initialize(self, pt):
#         self.kf.x = np.array([[pt[0]], [pt[1]], [0], [0]])

#     def predict(self):
#         self.kf.predict()
#         return self.kf.x[:2].reshape(-1)

#     def correct(self, pt):
#         self.kf.update(np.array([[pt[0]], [pt[1]]]))
#         return self.kf.x[:2].reshape(-1)


# class KeypointTracker:
#     def __init__(self, max_points=16, max_miss=50, max_distance=50):
#         self.max_points = max_points
#         self.trackers = [None] * max_points
#         self.disappeared = [0] * max_points
#         self.max_miss = max_miss
#         self.max_distance = max_distance

#     def update(self, detections, frame_num):
#         predicted_positions = []
#         for tracker in self.trackers:
#             if tracker is not None:
#                 predicted_positions.append(tracker.predict())
#             else:
#                 predicted_positions.append(None)

#         if not detections:
#             # No detections: increase disappeared count
#             for i, tracker in enumerate(self.trackers):
#                 if tracker is not None:
#                     self.disappeared[i] += 1
#                     if self.disappeared[i] > self.max_miss:
#                         self.trackers[i] = None
#             return

#         det_array = np.array(detections)
#         cost_matrix = np.full((self.max_points, len(det_array)), np.inf)

#         for i, pred in enumerate(predicted_positions):
#             if pred is not None:
#                 for j, det in enumerate(det_array):
#                     cost_matrix[i, j] = np.linalg.norm(pred - det)

#         row_ind, col_ind = linear_sum_assignment(cost_matrix)

#         assigned_detections = set()
#         assigned_trackers = set()

#         # Update matched trackers
#         for i, j in zip(row_ind, col_ind):
#             if cost_matrix[i, j] < self.max_distance:
#                 self.trackers[i].correct(det_array[j])
#                 self.disappeared[i] = 0
#                 assigned_detections.add(j)
#                 assigned_trackers.add(i)
#             else:
#                 # Distance too large: mark tracker as unmatched
#                 pass

#         # Increment disappeared count for unmatched trackers
#         for i in range(self.max_points):
#             if i not in assigned_trackers and self.trackers[i] is not None:
#                 self.disappeared[i] += 1
#                 if self.disappeared[i] > self.max_miss:
#                     self.trackers[i] = None

#         # Initialize new trackers for unmatched detections
#         for j, det in enumerate(det_array):
#             if j not in assigned_detections:
#                 for i in range(self.max_points):
#                     if self.trackers[i] is None:
#                         kf = KF2D()
#                         kf.initialize(det)
#                         self.trackers[i] = kf
#                         self.disappeared[i] = 0
#                         break

#     def get_positions(self):
#         positions = []
#         for tracker in self.trackers:
#             if tracker is not None:
#                 pos = tracker.predict()
#                 positions.append(pos)
#             else:
#                 positions.append(None)
#         return positions


class KF2D:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        self.kf.R *= 0.3
        self.kf.P *= 10
        self.kf.Q *= 0.01
        self.last_position = None

    def initialize(self, pt):
        self.kf.x = np.array([[pt[0]], [pt[1]], [0], [0]])
        self.last_position = pt

    def predict(self):
        self.kf.predict()
        self.last_position = self.kf.x[:2].reshape(-1)
        return self.last_position

    def correct(self, pt):
        self.kf.update(np.array([[pt[0]], [pt[1]]]))
        self.last_position = self.kf.x[:2].reshape(-1)
        return self.last_position


class KeypointTracker:
    def __init__(self, num_keypoints=18, max_miss=30, max_distance=50):
        self.num_keypoints = num_keypoints
        self.max_miss = max_miss
        self.max_distance = max_distance

        self.trackers = [None] * num_keypoints
        self.disappeared = [0] * num_keypoints
        self.initialized = False

    def update(self, detections, frame_num):
        if not self.initialized:
            for i in range(min(len(detections), self.num_keypoints)):
                self.trackers[i] = KF2D()
                self.trackers[i].initialize(detections[i])
                self.disappeared[i] = 0
            self.initialized = True
            return

        for i in range(self.num_keypoints):
            tracker = self.trackers[i]
            if tracker is None:
                continue

            pred = tracker.predict()
            best_det = None
            best_dist = self.max_distance

            for det in detections:
                dist = np.linalg.norm(pred - det)
                if dist < best_dist:
                    best_det = det
                    best_dist = dist

            if best_det is not None:
                tracker.correct(best_det)
                self.disappeared[i] = 0
            else:
                self.disappeared[i] += 1

            if self.disappeared[i] > self.max_miss:
                self.trackers[i] = None

        # Reinitialize lost trackers if we have extra detections (only if really needed)
        unmatched_dets = [det for det in detections if not any(
            np.linalg.norm(det - tracker.last_position) < self.max_distance
            for tracker in self.trackers if tracker is not None
        )]

        for i in range(self.num_keypoints):
            if self.trackers[i] is None and unmatched_dets:
                self.trackers[i] = KF2D()
                self.trackers[i].initialize(unmatched_dets.pop(0))
                self.disappeared[i] = 0

    def get_positions(self):
        return [kf.predict() if kf else None for kf in self.trackers]

### === FILE: main_tracking_script.py ===
import cv2
import numpy as np
import os
from datetime import datetime
from ir_keypoint_tracking_kalman import KeypointTracker
from utils_homography import *  # assume detect_blob, detect_aruco, keypoints_classifier, etc. are defined

def setup_video_streams(col_video, ir_video):
    col = cv2.VideoCapture(col_video)
    ir = cv2.VideoCapture(ir_video)
    col_fps = int(col.get(cv2.CAP_PROP_FPS))
    ir_fps = int(ir.get(cv2.CAP_PROP_FPS))
    col_frame_count = int(col.get(cv2.CAP_PROP_FRAME_COUNT))
    ir_frame_count = int(ir.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Color FPS: {col_fps}, IR FPS: {ir_fps}, Frames: {col_frame_count} vs {ir_frame_count}")
    return col, ir, col_fps, col_frame_count, ir_frame_count

def sync_video_streams(col, ir, col_frame_count, ir_frame_count):
    skip_frames = abs(col_frame_count - ir_frame_count)
    capture = col if col_frame_count > ir_frame_count else ir
    for _ in range(skip_frames):
        capture.read()

def initialize_blob_detector():
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.minThreshold = 192
    params.maxThreshold = 255
    params.filterByArea = True
    params.minArea = 1
    params.maxArea = 10000
    params.filterByCircularity = True
    params.minCircularity = 0.01
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    params.filterByConvexity = True
    params.minConvexity = 0.1

    return cv2.SimpleBlobDetector_create(params)

def draw_keypoints_on_frame(col_frame, keypoint_ids, mapped_keypoints):
    for id in range(len(keypoint_ids)):
        x, y = (int(mapped_keypoints[id, 0]), int(mapped_keypoints[id, 1]))
        cv2.circle(col_frame, (x, y), 5, (0, 255, 255), -1)
        cv2.putText(col_frame, str(keypoint_ids[id]), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

def temp_func(gray):
    gamma = 1
    look_up = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    enhanced = cv2.LUT(gray, look_up)

    # Step 3: Threshold to isolate bright centers
    _, thresh = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY)

    # Optional: Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    return processed

def process_frame_pair(col_frame, ir_frame, matrix, tracker, detector, detect_keypoints, frame_num):
    ir_frame = undistort_image(ir_frame, camera_matrix, dist_coeffs)
    gray = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY)

    gray = temp_func(gray)
    cv2.imshow("Gray",gray)

    keypoints = detector.detect(gray)
    det_pts = [(kp.pt[0], kp.pt[1]) for kp in keypoints]

    tracker.update(det_pts, frame_num)
    positions = tracker.get_positions()

    keypoint_ids, trans_keypoints = [], []
    for i, pt in enumerate(positions):
        if pt is not None:
            keypoint_ids.append(i)
            trans_keypoints.append(pt)

    trans_keypoints = np.array(trans_keypoints).reshape(-1, 1, 2)
    aligned_img = cv2.warpPerspective(ir_frame, matrix, (col_frame.shape[1], col_frame.shape[0]))

    if detect_keypoints:
        mapped_keypoints = cv2.perspectiveTransform(trans_keypoints, matrix).reshape(-1, 2)
        draw_keypoints_on_frame(col_frame, keypoint_ids, mapped_keypoints)

    overlay = cv2.addWeighted(col_frame, 1, aligned_img, 0, 0)
    return overlay

def video_overlay_2(col_video, ir_video, matrix, detect_keypoints=False):
    col, ir, col_fps, col_frame_count, ir_frame_count = setup_video_streams(col_video, ir_video)
    sync_video_streams(col, ir, col_frame_count, ir_frame_count)

    detector = initialize_blob_detector()
    tracker = KeypointTracker()
    frame_num = 0

    while col.isOpened() and ir.isOpened():
        ret1, col_frame = col.read()
        ret2, ir_frame = ir.read()
        if not ret1 or not ret2:
            break

        overlay = process_frame_pair(col_frame, ir_frame, matrix, tracker, detector, detect_keypoints, frame_num)
        cv2.imshow('Live Feed', overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_num += 1

    col.release()
    ir.release()
    cv2.destroyAllWindows()

# ---- Main Entry ----
camera_matrix = np.array([[1.80376226e+03,0.00000000e+00,3.16914171e+02],
 [0.00000000e+00,4.31043121e+03,2.87652674e+02],
 [0.00000000e+00,0.00000000e+00,1.00000000e+00]])
dist_coeffs = np.array([[-3.49027957e+00,3.13640856e+01,8.12342618e-02,3.06527531e-02,-3.86407135e+02]])

subject = input("Enter Subject/Person Name: ")
folder_path = os.path.join("Recordings", subject)

if subject in os.listdir("Recordings"):
    recs = os.listdir(folder_path)
    recs.sort()
    color_videos = [f for f in recs if f.startswith("color_video")]
    ir_videos = [f for f in recs if f.startswith("ir_video")]

    def extract_dt(filename):
        dt_str = filename.split("_video_")[1].replace(".avi", "")
        return datetime.strptime(dt_str, "%m-%d-%y_%H-%M-%S")

    latest_color = max(color_videos, key=extract_dt)
    latest_ir = max(ir_videos, key=extract_dt)
    color_video_path = os.path.join(folder_path, latest_color)
    ir_video_path = os.path.join(folder_path, latest_ir)
else:
    raise Exception("Subject not found in Recordings")

col_image = cv2.imread(r"Custom_Results/sample_test_Color.png")
ir_image = cv2.imread(r"Custom_Results/sample_test_IR.png", cv2.IMREAD_GRAYSCALE)
marked_positions,corners,ids = detect_aruco(col_image)
aruco_ref_pts = np.array([marked_positions[i] for i in range(4)])

ir_image = undistort_image(ir_image, camera_matrix, dist_coeffs)
keypoints = detect_blob(ir_image)
for i in keypoints:
    cv2.circle(ir_image, (int(i.pt[0]), int(i.pt[1])), 2, (0, 255, 0), -1)
ir_ref_pts, body_points = keypoints_classifier(keypoints)
ir_image = preprocess_ir_data(ir_image, 255)
ir_image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
im_with_keypoints = draw_blob(ir_image, keypoints, ir_ref_pts)
irtorgb_aligned_img, irtorgb_overlay, M = homography_transform(ir_image, col_image, ir_ref_pts, aruco_ref_pts)

video_overlay_2(color_video_path, ir_video_path, M, detect_keypoints=True)
cv2.waitKey(0)
cv2.destroyAllWindows()