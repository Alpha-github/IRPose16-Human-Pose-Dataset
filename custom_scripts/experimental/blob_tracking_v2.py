import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# Constants
TOTAL_KEYPOINTS = 18
MAX_DISAPPEARANCE = 30
EXCLUDE_TOPMOST = 2  # Number of topmost blobs to ignore

# Blob detector setup
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

# Tracking structures
last_known_positions = {i: None for i in range(TOTAL_KEYPOINTS)}  # id: (x, y)
inactive_counters = {i: 0 for i in range(TOTAL_KEYPOINTS)}        # id: missing_frame_count
excluded_positions = []  # Topmost 2 blobs
frame_num = 0

def compute_cost_matrix(previous_points, current_points):
    cost_matrix = np.zeros((len(previous_points), len(current_points)))
    for i, (px, py) in enumerate(previous_points):
        for j, (cx, cy) in enumerate(current_points):
            cost_matrix[i, j] = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    return cost_matrix

def is_close(p1, p2, threshold=10):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < threshold

def ir_keypoint_tracking(frame, frame_num, MAX_DISAPPEARANCE, EXCLUDE_TOPMOST, detector, last_known_positions, inactive_counters, excluded_positions):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints = detector.detect(gray)

    # if len(keypoints)!=TOTAL_KEYPOINTS:
    #     print(len(keypoints), "Keypoints detected - frame:", frame_num+1)

    current_points = [(kp.pt[0], kp.pt[1]) for kp in keypoints]

    if frame_num == 0:
        # Identify topmost 2 blobs
        sorted_by_y = sorted(current_points, key=lambda pt: pt[1])
        excluded_positions = sorted_by_y[:EXCLUDE_TOPMOST]
        print(f"Excluded positions (topmost blobs): {excluded_positions}")

    # Filter out excluded positions from current_points
    filtered_points = []
    for pt in current_points:
        if not any(is_close(pt, excl_pt) for excl_pt in excluded_positions):
            filtered_points.append(pt)

    current_points = filtered_points

    active_ids = [i for i in last_known_positions if last_known_positions[i] is not None]
    active_points = [last_known_positions[i] for i in active_ids]

    matched = {}
    assigned = set()

    if active_points and current_points:
        cost_matrix = compute_cost_matrix(active_points, current_points)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 100:
                id_to_use = active_ids[r]
                matched[id_to_use] = current_points[c]
                assigned.add(c)

    # Update known positions with matched values
    for id in matched:
        last_known_positions[id] = matched[id]
        inactive_counters[id] = 0

    # Assign new IDs to unmatched detections
    unmatched_points = [pt for idx, pt in enumerate(current_points) if idx not in assigned]
    for id in range(TOTAL_KEYPOINTS):
        if id not in matched and last_known_positions[id] is None and unmatched_points:
            last_known_positions[id] = unmatched_points.pop(0)
            inactive_counters[id] = 0

    # Handle inactive
    for id in range(TOTAL_KEYPOINTS):
        if id not in matched:
            inactive_counters[id] += 1
            if inactive_counters[id] >= MAX_DISAPPEARANCE:
                last_known_positions[id] = None
    
    return frame, last_known_positions, inactive_counters, excluded_positions

# cap = cv2.VideoCapture(r'Recordings\Pranav\ir_video_04-08-25_15-20-54.avi')

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame, last_known_positions, inactive_counters, excluded_positions = ir_keypoint_tracking(
#         frame, frame_num, detector, last_known_positions, inactive_counters, excluded_positions)

#     # Draw points
#     valid_keypoints = 0
#     for id, pos in last_known_positions.items():
#         if pos is not None:
#             x, y = int(pos[0]), int(pos[1])
#             cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
#             cv2.putText(frame, str(id), (x + 10, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#             valid_keypoints += 1
    
    # if valid_keypoints < TOTAL_KEYPOINTS - EXCLUDE_TOPMOST:
    #     print(frame_num, "Valid keypoints:", valid_keypoints)
    #     # cv2.waitKey(0)  # Wait indefinitely if fewer than expected keypoints

    # # Draw total count
    # cv2.putText(frame, f"Frame {frame_num+1} Total Keypoints (excluding top 2): {valid_keypoints}",
    #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.8, (0, 255, 255), 2)

    # cv2.imshow('Tracking (excluding topmost 2 blobs)', frame)
    # if cv2.waitKey(30) & 0xFF == 27 or cv2.waitKey(30) == ord('q'):
    #     break

    # frame_num += 1

# cap.release()
# cv2.destroyAllWindows()
