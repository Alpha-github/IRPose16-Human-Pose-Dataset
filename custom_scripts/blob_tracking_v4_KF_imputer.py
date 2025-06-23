import numpy as np
from pykalman import KalmanFilter
from tqdm import tqdm
import time

import numpy as np

NUM_KEYPOINTS = 16
exclude_top = 2  # Number of keypoints to exclude from the top, adjust as needed

class SimpleKalmanImputer:
    def __init__(self, num_keypoints=NUM_KEYPOINTS):
        self.num_keypoints = num_keypoints

    def kalman_smooth_1d(self, data, Q=1e-5, R=1e-1):
        n = len(data)
        xhat = np.zeros(n)
        P = np.zeros(n)
        xhatminus = np.zeros(n)
        Pminus = np.zeros(n)
        K = np.zeros(n)

        # Initialize with the first valid value
        first_valid_idx = np.where(~np.isnan(data))[0][0]
        xhat[first_valid_idx] = data[first_valid_idx]
        P[first_valid_idx] = 1.0

        for k in range(first_valid_idx + 1, n):
            # Time update
            xhatminus[k] = xhat[k - 1]
            Pminus[k] = P[k - 1] + Q

            if not np.isnan(data[k]):
                # Measurement update
                K[k] = Pminus[k] / (Pminus[k] + R)
                xhat[k] = xhatminus[k] + K[k] * (data[k] - xhatminus[k])
                P[k] = (1 - K[k]) * Pminus[k]
            else:
                # If missing, predict only
                xhat[k] = xhatminus[k]
                P[k] = Pminus[k]

        return xhat

    def fill_nan_linear(self, series):
        """Fill NaNs with linear interpolation (or ffill/bfill at boundaries)."""
        series = np.array(series)
        nans = np.isnan(series)
        not_nans = ~nans
        indices = np.arange(len(series))

        if np.sum(not_nans) == 0:
            return series  # all NaNs, cannot fill

        series[nans] = np.interp(indices[nans], indices[not_nans], series[not_nans])
        return series

    def impute_keypoints(self, framewise_keypoints):
        frame_count = len(framewise_keypoints)
        imputed_array = np.full((frame_count, self.num_keypoints, 2), np.nan)
        imputed_flags = np.full((frame_count, self.num_keypoints), False)

        for kp_idx in range(self.num_keypoints):
            x_series = np.array([
                frame[kp_idx][0] if frame[kp_idx][0] is not None else np.nan
                for frame in framewise_keypoints
            ])
            y_series = np.array([
                frame[kp_idx][1] if frame[kp_idx][1] is not None else np.nan
                for frame in framewise_keypoints
            ])


            valid = ~np.isnan(x_series) & ~np.isnan(y_series)
            if np.sum(valid) < 3:
                continue  # Skip if too little data to impute

            x_filled = self.fill_nan_linear(x_series)
            y_filled = self.fill_nan_linear(y_series)

            x_smoothed = self.kalman_smooth_1d(x_filled)
            y_smoothed = self.kalman_smooth_1d(y_filled)

            for t in range(frame_count):
                if framewise_keypoints[t][kp_idx][0] is None or framewise_keypoints[t][kp_idx][1] is None:
                    # Only replace if originally missing
                    imputed_array[t, kp_idx] = [x_smoothed[t], y_smoothed[t]]
                    imputed_flags[t, kp_idx] = True
                else:
                    # Keep original detected point
                    imputed_array[t, kp_idx] = [x_series[t], y_series[t]]

        return imputed_array, imputed_flags




import cv2
import numpy as np
import os
from datetime import datetime
from ir_keypoint_tracking_kalman import KeypointTracker
from utils_homography import *  # assume detect_blob, detect_aruco, keypoints_classifier, etc. are defined
from scipy.optimize import linear_sum_assignment

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
    frames = ir_frame_count if col_frame_count > ir_frame_count else col_frame_count
    for _ in range(skip_frames):
        capture.read()
    
    return frames

def initialize_blob_detector():
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.minThreshold = 185
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

from scipy.optimize import linear_sum_assignment
import numpy as np

def match_keypoints(prev_pts, curr_pts, max_dist=50):
    """
    Matches previous frame keypoints (with known label order) to newly detected ones,
    ignoring missing keypoints represented as [None, None].
    
    Returns a mapping from prev_index → curr_index.
    """

    # Filter out invalid points
    valid_prev = [(i, pt) for i, pt in enumerate(prev_pts) if pt[0] is not None and pt[1] is not None]
    valid_curr = [(j, pt) for j, pt in enumerate(curr_pts) if pt[0] is not None and pt[1] is not None]

    if not valid_prev or not valid_curr:
        return {}  # Nothing to match

    prev_indices, prev_coords = zip(*valid_prev)
    curr_indices, curr_coords = zip(*valid_curr)

    prev_arr = np.array(prev_coords)  # shape (m, 2)
    curr_arr = np.array(curr_coords)  # shape (n, 2)

    # Compute pairwise distance matrix
    cost_matrix = np.linalg.norm(prev_arr[:, None, :] - curr_arr[None, :, :], axis=2)
    cost_matrix[cost_matrix > max_dist] = 1e5  # Penalize distant points

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = {}
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r][c] < 1e5:
            matches[prev_indices[r]] = curr_indices[c]

    return matches  # maps prev_index (with label) → current index (in detection list)


# def match_keypoints(prev_pts, curr_pts, max_dist=50):
#     """
#     Matches previous frame keypoints (with known label order) to newly detected ones,
#     ignoring missing keypoints represented as [None, None].
    
#     Returns a mapping from prev_index → curr_index.
#     """

#     # Filter out invalid points
#     valid_prev = [(i, pt) for i, pt in enumerate(prev_pts) if pt[0] is not None and pt[1] is not None]
#     valid_curr = [(j, pt) for j, pt in enumerate(curr_pts) if pt[0] is not None and pt[1] is not None]

#     if not valid_prev or not valid_curr:
#         return {}  # Nothing to match

#     matches = {}
#     for i, (prev_idx, prev_pt) in enumerate(valid_prev):
#         min_dist = float('inf')
#         best_j = None
#         for j, (curr_idx, curr_pt) in enumerate(valid_curr):
#             dist = np.linalg.norm(np.array(prev_pt) - np.array(curr_pt))
#             if dist < min_dist and dist <= max_dist:
#                 min_dist = dist
#                 best_j = curr_idx
#         if best_j is not None:
#             matches[prev_idx] = best_j  # Allow duplicates


#     return matches  # maps prev_index (with label) → current index (in detection list)

def preprocess_and_detect(col_video, ir_video, detector, camera_matrix, dist_coeffs):
    col, ir, col_fps, col_frame_count, ir_frame_count = setup_video_streams(col_video, ir_video)
    sync_video_streams(col, ir, col_frame_count, ir_frame_count)

    framewise_keypoints = []
    color_frames = []
    ir_frames = []
    frame_count = 1

    prev_frame_labeled = None
    last_valid_frame = None

    pbar = tqdm(total=min(col_frame_count, ir_frame_count), desc="Processing Frames")
    while col.isOpened() and ir.isOpened():
        ret1, col_frame = col.read()
        ret2, ir_frame = ir.read()
        if not ret1 or not ret2:
            break

        ir_frame = undistort_image(ir_frame, camera_matrix, dist_coeffs)
        gray = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY)
        gray = temp_func(gray)

        keypoints = detector.detect(gray)
        det_pts = [(kp.pt[0], kp.pt[1]) for kp in keypoints]
        det_pts.sort(key=lambda x: x[1])  # Sort by y to stabilize

        if len(det_pts) < 16:
            print(f"\n{frame_count}: Detected only {len(det_pts)} keypoints.")
            frame_count += 1
            continue

        frame_keypoints = [[None, None] for _ in range(NUM_KEYPOINTS)]

        if prev_frame_labeled is None:
            # First frame: direct assign
            for i, pt in enumerate(det_pts[:NUM_KEYPOINTS]):
                frame_keypoints[i] = pt
            last_valid_frame = frame_keypoints.copy()

        else:
            # Check if prev_frame_labeled has all valid keypoints
            valid_prev = all(pt[0] is not None and pt[1] is not None for pt in prev_frame_labeled)

            # Use last_valid_frame if prev_frame_labeled is incomplete
            reference_frame = prev_frame_labeled if valid_prev else last_valid_frame

            if reference_frame:
                match = match_keypoints(reference_frame, det_pts)
                for prev_idx, curr_idx in match.items():
                    if prev_idx < NUM_KEYPOINTS:
                        frame_keypoints[prev_idx] = det_pts[curr_idx]

            # Update last_valid_frame if all keypoints were matched
            if all(kp[0] is not None and kp[1] is not None for kp in frame_keypoints):
                last_valid_frame = frame_keypoints.copy()

        prev_frame_labeled = frame_keypoints.copy()
        framewise_keypoints.append(frame_keypoints)
        color_frames.append(col_frame.copy())
        ir_frames.append(gray.copy())

        frame_count += 1
        pbar.update(1)

    pbar.close()
    col.release()
    ir.release()
    return framewise_keypoints, color_frames, ir_frames

def perform_kalman_imputation(framewise_keypoints):
    imputer = SimpleKalmanImputer(num_keypoints=NUM_KEYPOINTS)
    imputed_array, imputed_flags = imputer.impute_keypoints(framewise_keypoints)
    print("Imputation shape:", imputed_array.shape)
    return imputed_array, imputed_flags

def display_video_with_overlay(color_frames, ir_frames, imputed_array, imputed_flags, matrix):
    for i in range(len(color_frames)):
        col_frame = color_frames[i]
        ir_gray = ir_frames[i]
        all_points = imputed_array[i]
        flags = imputed_flags[i]

        imputed_this_frame = [str(j) for j, flag in enumerate(flags) if flag]

        for j, flag in enumerate(flags):
            if flag:
                x, y = all_points[j]
                print(f"[Frame {i+1}] Imputed Keypoint {j}: x = {x:.2f}, y = {y:.2f}")

        for j, pt in enumerate(all_points):
            if not np.isnan(pt[0]):
                mapped = cv2.perspectiveTransform(np.array([[pt]], dtype=np.float32), matrix)[0][0]
                color = (255, 0, 0) if flags[j] else (0, 255, 255)
                if imputed_this_frame:
                    color = (0, 255, 0)
                cv2.circle(col_frame, tuple(int(x) for x in mapped), 5, color, -1)
                cv2.putText(col_frame, str(j), (int(mapped[0]) + 5, int(mapped[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                continue

        ir_warp = cv2.warpPerspective(ir_gray, matrix, (col_frame.shape[1], col_frame.shape[0]))
        overlay = cv2.addWeighted(col_frame, 1, cv2.cvtColor(ir_warp, cv2.COLOR_GRAY2BGR), 0, 0)

        cv2.putText(overlay, f"Frame: {i+1}/{len(color_frames)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Live Feed", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def video_overlay_2(col_video, ir_video, matrix):
    detector = initialize_blob_detector()
    framewise_keypoints, color_frames, ir_frames = preprocess_and_detect(
        col_video, ir_video, detector, camera_matrix, dist_coeffs)
    imputed_array, imputed_flags = perform_kalman_imputation(framewise_keypoints)
    display_video_with_overlay(color_frames, ir_frames, imputed_array, imputed_flags, matrix)


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

video_overlay_2(color_video_path, ir_video_path, M)
cv2.waitKey(0)
cv2.destroyAllWindows()
