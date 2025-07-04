import numpy as np
from pykalman import KalmanFilter
from tqdm import tqdm
import time
import pandas as pd
import json
import cv2
import os
from datetime import datetime
from utils_homography import *  # assume detect_blob, detect_aruco, keypoints_classifier, etc. are defined
from scipy.optimize import linear_sum_assignment

NUM_KEYPOINTS = 16
exclude_top = 2  # Number of keypoints to exclude from the top, adjust as needed

class WindowedKalmanImputer:
    def __init__(self, window_size=5, Q=1e-5, R=1e-1):
        self.window_size = window_size  # should be odd (e.g., 5, 7)
        self.Q = Q
        self.R = R

    def kalman_smooth_1d(self, data):
        n = len(data)
        xhat = np.zeros(n)
        P = np.zeros(n)
        xhat[0] = data[0]
        P[0] = 1.0

        for k in range(1, n):
            xhat[k] = xhat[k-1]
            P[k] = P[k-1] + self.Q

            if not np.isnan(data[k]):
                K = P[k] / (P[k] + self.R)
                xhat[k] = xhat[k] + K * (data[k] - xhat[k])
                P[k] = (1 - K) * P[k]

        return xhat

    def fill_nan_linear(self, series):
        series = pd.Series(series)
        return series.interpolate(method='linear').bfill().ffill().values

    def impute_frame(self, full_data, target_frame):
        num_frames, num_keypoints, _ = full_data.shape
        half_window = self.window_size // 2
        start = max(0, target_frame - half_window)
        end = min(num_frames, target_frame + half_window + 1)
        window = full_data[start:end]

        imputed_frame = np.full((num_keypoints, 2), np.nan, dtype=np.float32)
        flags = np.zeros(num_keypoints, dtype=bool)

        for kp in range(num_keypoints):
            x_series = window[:, kp, 0]
            y_series = window[:, kp, 1]

            if np.all(np.isnan(x_series)) or np.all(np.isnan(y_series)):
                continue  # still skip if completely missing
            elif np.count_nonzero(~np.isnan(x_series)) < 2:
                x_series = np.nan_to_num(x_series, nan=np.nanmean(x_series))
                y_series = np.nan_to_num(y_series, nan=np.nanmean(y_series))

            x_filled = self.fill_nan_linear(x_series)
            y_filled = self.fill_nan_linear(y_series)

            x_smooth = self.kalman_smooth_1d(x_filled)
            y_smooth = self.kalman_smooth_1d(y_filled)

            center_idx = target_frame - start
            original_x = full_data[target_frame, kp, 0]
            original_y = full_data[target_frame, kp, 1]

            if np.isnan(original_x) or np.isnan(original_y):
                imputed_frame[kp] = [x_smooth[center_idx], y_smooth[center_idx]]
                flags[kp] = True
            else:
                imputed_frame[kp] = [original_x, original_y]

        return imputed_frame, flags

    def perform_windowed_imputation(self, framewise_keypoints):
        num_frames = len(framewise_keypoints)
        num_keypoints = len(framewise_keypoints[0])
        full_data = np.array([
            [[pt[0], pt[1]] if pt is not None else [np.nan, np.nan] for pt in frame]
            for frame in framewise_keypoints
        ], dtype=np.float32)

        imputed_array = []
        imputed_flags = []

        for t in range(num_frames):
            imputed_frame, flags = self.impute_frame(full_data, t)

            # ✅ Update full_data with imputed_frame where NaNs were present
            nan_mask = np.isnan(full_data[t, :, 0]) | np.isnan(full_data[t, :, 1])
            full_data[t][nan_mask] = imputed_frame[nan_mask]

            imputed_array.append(imputed_frame)
            imputed_flags.append(flags)

        return np.array(imputed_array), np.array(imputed_flags)



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

def match_keypoints(prev_pts, curr_pts, max_dist=50):
    """
    Label-preserving matching: Assign each label from previous frame to the nearest
    unmatched current detection, ensuring identity consistency.
    """
    matches = {}
    used_curr_indices = set()

    for label, prev_pt in enumerate(prev_pts):
        if prev_pt[0] is None or prev_pt[1] is None:
            continue

        best_dist = float('inf')
        best_idx = None

        for idx, curr_pt in enumerate(curr_pts):
            if curr_pt[0] is None or curr_pt[1] is None or idx in used_curr_indices:
                continue

            dist = np.linalg.norm(np.array(prev_pt) - np.array(curr_pt))
            if dist < best_dist and dist <= max_dist:
                best_dist = dist
                best_idx = idx

        if best_idx is not None:
            matches[label] = best_idx
            used_curr_indices.add(best_idx)

    return matches

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
        # gray = temp_func(gray)

        keypoints = detector.detect(gray)
        det_pts = [(kp.pt[0], kp.pt[1]) for kp in keypoints]
        det_pts.sort(key=lambda x: x[1])  # Sort by y to stabilize
        det_pts = det_pts[exclude_top:]  # Limit to NUM_KEYPOINTS

        frame_keypoints = [[None, None] for _ in range(NUM_KEYPOINTS)]

        if prev_frame_labeled is None:
            # First frame: direct assign
            for i, pt in enumerate(det_pts):
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
    imputer = WindowedKalmanImputer(window_size=15)
    imputed_array, imputed_flags = imputer.perform_windowed_imputation(framewise_keypoints)
    print("Imputation shape:", imputed_array.shape)
    return imputed_array, imputed_flags

def draw_keypoints(frame, points, flags, matrix):
    imputed_this_frame = [j for j, flag in enumerate(flags) if flag]

    for j, pt in enumerate(points):
        if not np.isnan(pt[0]):
            mapped = cv2.perspectiveTransform(np.array([[pt]], dtype=np.float32), matrix)[0][0]
            color = (0, 255, 0) if j in imputed_this_frame else (0, 255, 255) if not flags[j] else (255, 0, 0)
            cv2.circle(frame, tuple(int(x) for x in mapped), 5, color, -1)
            cv2.putText(frame, str(j), (int(mapped[0]) + 5, int(mapped[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

def warp_and_overlay(ir_frame, color_frame, matrix):
    ir_warp = cv2.warpPerspective(ir_frame, matrix, (color_frame.shape[1], color_frame.shape[0]))
    return cv2.addWeighted(color_frame, 1, cv2.cvtColor(ir_warp, cv2.COLOR_GRAY2BGR), 0, 0)

def display_frame(index, color_frames, ir_frames, imputed_array, imputed_flags, matrix):
    col_frame = color_frames[index].copy()
    ir_gray = ir_frames[index]
    all_points = imputed_array[index]
    flags = imputed_flags[index]

    # Print imputed keypoints for this frame
    for j, flag in enumerate(flags):
        if flag:
            x, y = all_points[j]
            if not np.isnan(x) and not np.isnan(y):
                pt = np.array([[[x, y]]], dtype=np.float32)
                mapped = cv2.perspectiveTransform(pt, matrix)[0][0]
                print(f"[Frame {index+1}] Imputed Keypoint {j}: x = {mapped[0]:.2f}, y = {mapped[1]:.2f}")

    draw_keypoints(col_frame, all_points, flags, matrix)
    overlay = warp_and_overlay(ir_gray, col_frame, matrix)

    cv2.putText(overlay, f"Frame: {index+1}/{len(color_frames)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Live Feed", overlay)
    # cv2.imshow("IR Frame", ir_gray)

def display_video_with_overlay(color_frames, ir_frames, imputed_array, imputed_flags, matrix):
    index = 0
    playing = True
    total_frames = len(color_frames)

    while True:
        if playing or cv2.getWindowProperty("Live Feed", 0) < 0:
            display_frame(index, color_frames, ir_frames, imputed_array, imputed_flags, matrix)

        key = cv2.waitKey(10 if playing else 2) & 0xFF

        if key == ord('q'):
            break
        elif key == 32:  # Space bar
            playing = not playing
        elif key == 65 or key==97:  # Left arrow ← (ASCII for left arrow)
            if not playing and index > 0:
                index -= 1
                display_frame(index, color_frames, ir_frames, imputed_array, imputed_flags, matrix)
        elif key == 68 or key==100:  # Right arrow → (ASCII for right arrow)
            if not playing and index < total_frames - 1:
                index += 1
                display_frame(index, color_frames, ir_frames, imputed_array, imputed_flags, matrix)
        elif playing:
            index += 1
            if index >= total_frames:
                break

    cv2.destroyAllWindows()

def save_imputed_keypoints_to_json(subject, imputed_array, json_datetime_stamp, matrix, color_frame_shape):
    output_dict = {}
    incomplete_frames = []

    for frame_no, keypoints in enumerate(imputed_array):
        frame_dict = {}
        all_present = True

        if len(keypoints) != NUM_KEYPOINTS:
            print(f"⚠️ Warning: Frame {frame_no} has {len(keypoints)} keypoints, expected {NUM_KEYPOINTS}.")
            incomplete_frames.append(frame_no)
            continue

        for i, (x, y) in enumerate(keypoints):
            if np.isnan(x) or np.isnan(y):
                frame_dict[i] = None
                all_present = False
            else:
                pt = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
                mapped = cv2.perspectiveTransform(pt, matrix)[0][0]
                frame_dict[i] = (float(mapped[0]), float(mapped[1]))

        if not all_present:
            incomplete_frames.append(frame_no)
        output_dict[frame_no] = frame_dict

    # Verification: Check if any frames are incomplete
    if incomplete_frames:
        print(f"⚠️ Warning: The following frames are incomplete (missing keypoints): {incomplete_frames}")
        print(f"❌ JSON will NOT be saved due to incomplete keypoints.")
        return

    # All frames complete → proceed with saving
    output_dir = os.path.join("pose_json", subject)
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{subject}_{json_datetime_stamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(output_dict, f, indent=4)

    print(f"✅ JSON saved successfully to: {filepath}")

# --- Correction Imports ---
import pickle
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

class KeypointAutoencoder(nn.Module):
    def __init__(self, input_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def load_validators():
    ae = KeypointAutoencoder()
    ae.load_state_dict(torch.load("keypoint_validator_models/ae.pth"))
    ae.eval()

    with open("keypoint_validator_models/pca.pkl", "rb") as f:
        pca = pickle.load(f)
    with open("keypoint_validator_models/gmm.pkl", "rb") as f:
        gmm = pickle.load(f)

    return ae, pca, gmm

def validate_and_correct_kps(imputed_array, imputed_flags, ae, pca, gmm, threshold=50):
    corrected = imputed_array.copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae.to(device)

    for i, (frame, flags) in enumerate(zip(imputed_array, imputed_flags)):
        flat = frame.flatten()
        x = torch.tensor(flat, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            ae_pred = ae(x).cpu().numpy().reshape(-1, 2)

        pca_pred = pca.inverse_transform(pca.transform([flat])).reshape(-1, 2)
        likelihood = gmm.score_samples([flat])[0]

        for j in range(NUM_KEYPOINTS):
            if flags[j]:
                ae_dist = np.linalg.norm(corrected[i, j] - ae_pred[j])
                pca_dist = np.linalg.norm(corrected[i, j] - pca_pred[j])
                if ae_dist > threshold or pca_dist > threshold or likelihood < -200:
                    corrected[i, j] = ae_pred[j]  # or average(ae_pred[j], pca_pred[j])
    return corrected

def video_overlay_2(col_video, ir_video, matrix):
    detector = initialize_blob_detector()
    framewise_keypoints, color_frames, ir_frames = preprocess_and_detect(
        col_video, ir_video, detector, camera_matrix, dist_coeffs)
    imputed_array, imputed_flags = perform_kalman_imputation(framewise_keypoints)

    ae, pca, gmm = load_validators()
    corrected_array = validate_and_correct_kps(imputed_array, imputed_flags, ae, pca, gmm)
    imputed_array = corrected_array
    
    nan_count = np.isnan(imputed_array).sum()
    if nan_count > 0:
        print(f"⚠️ Warning: {nan_count} imputed values are still NaN")
        # Print frame indices with at least one NaN
        nan_frames = []
        for i, frame in enumerate(imputed_array):
            if np.isnan(frame).any():
                nan_frames.append(i)
        print(f"Frames with NaNs: {nan_frames}")

    display_video_with_overlay(color_frames, ir_frames, imputed_array, imputed_flags, matrix)

    save_choice = input("Do you want to save the imputed keypoints as JSON? (y/n): ").strip().lower()
    if save_choice == 'y':
        # Extract timestamp from video filename
        video_basename = os.path.basename(col_video)
        try:
            dt_str = video_basename.split("_video_")[1].replace(".avi", "")
        except IndexError:
            dt_str = datetime.now().strftime("%m-%d-%y_%H-%M-%S")

        # Pass matrix and a sample color frame's shape
        save_imputed_keypoints_to_json(subject, imputed_array, dt_str, matrix, color_frames[0].shape)



# ---- Main Entry ----
camera_matrix = np.array([[1.80376226e+03,0.00000000e+00,3.16914171e+02],
 [0.00000000e+00,4.31043121e+03,2.87652674e+02],
 [0.00000000e+00,0.00000000e+00,1.00000000e+00]])
dist_coeffs = np.array([[-3.49027957e+00,3.13640856e+01,8.12342618e-02,3.06527531e-02,-3.86407135e+02]])

subject = input("Enter Subject/Person Name: ")
folder_path = os.path.join("Recordings", subject)

# if subject in os.listdir("Recordings"):
#     recs = os.listdir(folder_path)
#     recs.sort()
#     color_videos = [f for f in recs if f.startswith("color_video")]
#     ir_videos = [f for f in recs if f.startswith("ir_video")]

#     def extract_dt(filename):
#         dt_str = filename.split("_video_")[1].replace(".avi", "")
#         return datetime.strptime(dt_str, "%m-%d-%y_%H-%M-%S")

#     latest_color = max(color_videos, key=extract_dt)
#     latest_ir = max(ir_videos, key=extract_dt)
#     color_video_path = os.path.join(folder_path, latest_color)
#     ir_video_path = os.path.join(folder_path, latest_ir)
# else:
#     raise Exception("Subject not found in Recordings")

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
# video_overlay_2("Recordings\Pranav2\color_video_05-06-25_14-10-39.avi", "Recordings\Pranav2\ir_video_05-06-25_14-10-39.avi", M)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

if subject in os.listdir("Recordings"):
    recs = os.listdir(folder_path)
    recs.sort()
    color_videos = [f for f in recs if f.startswith("color_video")]
    ir_videos = [f for f in recs if f.startswith("ir_video")]

    for col,ir in zip(color_videos[:], ir_videos[:]):
        color_video_path = os.path.join(folder_path, col)
        ir_video_path = os.path.join(folder_path, ir)
        print(f"Processing: {col} and {ir}")
        video_overlay_2(color_video_path, ir_video_path, M)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
    raise Exception("Subject not found in Recordings")