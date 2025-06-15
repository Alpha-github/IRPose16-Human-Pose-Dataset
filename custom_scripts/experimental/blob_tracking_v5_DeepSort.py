# deep_sort_tracker.py

from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

class DeepSORTTracker:
    def __init__(self, max_cosine_distance=0.2, max_age=30, n_init=2):
        self.tracker = DeepSort(max_age=max_age, 
                                n_init=n_init, 
                                max_cosine_distance=max_cosine_distance)
        self.initialized = False

    def update(self, detections, frame, frame_num=None):
        """
        detections: list of (x, y) centroids (e.g. keypoints)
        frame: current frame (RGB or BGR)
        """
        final_boxes = []
        for i, det in enumerate(detections):
            if not isinstance(det, (tuple, list, np.ndarray)) or len(det) != 3:
                continue  # skip invalid detection

            x, y, size = det
            size = size  # bounding box size
            x1 = int(x - size / 2)
            y1 = int(y - size / 2)
            w = h = size

            final_boxes.append((
                [x1, y1, w, h],    # bbox
                1.0,               # confidence
                "keypt"             # label (can be just index or keypoint name)
            ))

        # Update DeepSORT with proper format
        tracks = self.tracker.update_tracks(final_boxes, frame=frame)

        outputs = []
        for track in tracks:
            # print(track.is_confirmed())
            # if not track.is_confirmed():
            #     continue
            x1, y1, x2, y2 = track.to_ltrb()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            outputs.append((track.track_id, (cx, cy)))

        return outputs


# main_tracking_script_with_deepsort.py

import cv2
import numpy as np
import os
from datetime import datetime
from utils_homography import *  # assuming your utility functions

def initialize_blob_detector():
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.minThreshold, params.maxThreshold = 190, 255
    params.filterByArea = True
    params.minArea = 1
    params.filterByCircularity, params.minCircularity = True, 0.01
    params.filterByInertia, params.minInertiaRatio = True, 0.01
    params.filterByConvexity, params.minConvexity = True, 0.1
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

def process_frame_pair(col_frame, ir_frame, matrix, detector, deepsort, frame_num):
    ir_frame = undistort_image(ir_frame, camera_matrix, dist_coeffs)
    gray = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY)

    gray = temp_func(gray)
    cv2.imshow("Gray",gray)

    keypoints = detector.detect(gray)
    det_pts = [(kp.pt[0], kp.pt[1]) for kp in keypoints]

    det_pts_with_size = [(kp.pt[0], kp.pt[1], kp.size) for kp in keypoints]

    tracks = deepsort.update(det_pts_with_size, ir_frame, frame_num)
    keypoint_ids = [tid for tid, (x, y) in tracks]
    positions = [(x, y) for tid, (x, y) in tracks]

    trans_kp = np.array(positions, dtype=np.float32).reshape(-1, 1, 2)
    aligned_ir = cv2.warpPerspective(ir_frame, matrix, (col_frame.shape[1], col_frame.shape[0]))

    mapped = cv2.perspectiveTransform(trans_kp, matrix).reshape(-1, 2)
    for tid, (mx, my) in zip(keypoint_ids, mapped):
        cv2.circle(col_frame, (int(mx), int(my)), 5, (0, 255, 255), -1)
        cv2.putText(col_frame, str(tid), (int(mx) + 10, int(my) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    overlay = cv2.addWeighted(col_frame, 1, aligned_ir, 0, 0)
    return overlay

def video_overlay_deepsort(color_video, ir_video, homography_matrix):
    cap_col = cv2.VideoCapture(color_video)
    cap_ir = cv2.VideoCapture(ir_video)
    detector = initialize_blob_detector()
    deepsort = DeepSORTTracker()
    frame_num = 0

    while True:
        ret1, col_frame = cap_col.read()
        ret2, ir_frame = cap_ir.read()
        if not ret1 or not ret2:
            break

        overlay = process_frame_pair(col_frame, ir_frame, homography_matrix,
                                     detector, deepsort, frame_num)

        cv2.imshow('DeepSORT Tracking', overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_num += 1

    cap_col.release()
    cap_ir.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
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

    video_overlay_deepsort(color_video_path, ir_video_path, M)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
