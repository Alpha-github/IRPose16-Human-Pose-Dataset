import cv2
import numpy as np
import threading
import queue
from blob_tracking_v2 import ir_keypoint_tracking
from datetime import datetime
from ir_keypoint_tracking_kalman import KeypointTracker
from utils_homography import *

import os
# import inspect

# from pyorbbecsdk import Config, OBError, OBSensorType, OBFormat, Pipeline, FrameSet, VideoStreamProfile
# from pyorbbecsdk.examples.utils import frame_to_bgr_image

# from color_ir_recorder import datetime_str

# import pyorbbecsdk
# print(os.path.dirname(pyorbbecsdk.__file__))
# print(inspect.getfile(pyorbbecsdk))

camera_matrix = np.array([[1.80376226e+03,0.00000000e+00,3.16914171e+02],
 [0.00000000e+00,4.31043121e+03,2.87652674e+02],
 [0.00000000e+00,0.00000000e+00,1.00000000e+00]]
)
dist_coeffs = np.array([[-3.49027957e+00,3.13640856e+01,8.12342618e-02,3.06527531e-02,-3.86407135e+02]])

tracker = KeypointTracker()
frame_num = 0

def video_overlay(col_video,ir_video,matrix, detect_keypoints=False,output_path=None):
    col = cv2.VideoCapture(col_video)
    ir = cv2.VideoCapture(ir_video)

    col_fps = int(col.get(cv2.CAP_PROP_FPS))
    col_frame_count = int(col.get(cv2.CAP_PROP_FRAME_COUNT))
    ir_frame_count = int(ir.get(cv2.CAP_PROP_FRAME_COUNT))
    ir_fps = int(ir.get(cv2.CAP_PROP_FPS))
    print(col_fps,ir_fps, col_frame_count, ir_frame_count)

    if col_frame_count>ir_frame_count:
        skip_frames = int(col.get(cv2.CAP_PROP_FRAME_COUNT)) - int(ir.get(cv2.CAP_PROP_FRAME_COUNT))
        while skip_frames>0:
            _, _ = col.read()
            skip_frames-=1
    else:
        skip_frames = int(ir.get(cv2.CAP_PROP_FRAME_COUNT)) - int(col.get(cv2.CAP_PROP_FRAME_COUNT))
        while skip_frames>0:
            _, _ = ir.read()
            skip_frames-=1
    
    if output_path!=None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, col_fps, (int(col.get(cv2.CAP_PROP_FRAME_WIDTH)), int(col.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while(col.isOpened() or ir.isOpened()):
        ret1, col_frame = col.read()
        ret2, ir_frame = ir.read()

        if not ret1 or not ret2:
            break

        ir_frame = undistort_image(ir_frame,camera_matrix, dist_coeffs)
        ir_frame_gray = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY)
        
        if detect_keypoints:
            keypoints = detect_blob(ir_frame_gray, min_thresh=180, max_thresh=255,min_area=1,min_circularity=0.1,min_convexity=0.1,min_inertia=0.01)
            trans_keypoints = cv2.KeyPoint_convert(keypoints).reshape(-1,1,2)

        # for i in keypoints:
        #     cv2.circle(ir_frame, (int(i.pt[0]), int(i.pt[1])), 2, (0, 255, 0), -1)

        # ir_frame= cv2.drawKeypoints(ir_frame, keypoints, np.array([]), (0, 0, 255), 
        #                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Warp image2 to align with image1
        aligned_img = cv2.warpPerspective(ir_frame, matrix, (col_frame.shape[1], col_frame.shape[0]))

        if detect_keypoints:
            mapped_keypoints = np.array(cv2.perspectiveTransform(trans_keypoints, matrix).reshape(-1,2))
            for i in range(mapped_keypoints.shape[0]):
                cv2.circle(col_frame, (int(mapped_keypoints[i,0]), int(mapped_keypoints[i,1])), 5, (0, 255, 255), -1)

        # Overlay images
        overlay = cv2.addWeighted(col_frame, 1, aligned_img, 0, 0)

        if output_path!=None:
            out.write(overlay)

        cv2.imshow('Live Feed', overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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
    params.minThreshold = 190
    params.maxThreshold = 255
    params.filterByArea = True
    params.minArea = 1
    params.filterByCircularity = True
    params.minCircularity = 0.01
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    params.filterByConvexity = True
    params.minConvexity = 0.1
    return cv2.SimpleBlobDetector_create(params)


def initialize_video_writer(output_path, col, fps):
    width = int(col.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(col.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def draw_keypoints_on_frame(col_frame, keypoint_ids, mapped_keypoints):
    for id in range(len(keypoint_ids)):
        x, y = (int(mapped_keypoints[id, 0]), int(mapped_keypoints[id, 1]))
        cv2.circle(col_frame, (x, y), 5, (0, 255, 255), -1)
        cv2.putText(col_frame, str(keypoint_ids[id]), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


def process_frame_pair(col_frame, ir_frame, matrix, tracker, detector, detect_keypoints, frame_num):
    ir_frame = undistort_image(ir_frame, camera_matrix, dist_coeffs)

    gray = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY)
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


def video_overlay_2(col_video, ir_video, matrix, detect_keypoints=False, output_path=None):
    col, ir, col_fps, col_frame_count, ir_frame_count = setup_video_streams(col_video, ir_video)
    sync_video_streams(col, ir, col_frame_count, ir_frame_count)

    if output_path:
        out = initialize_video_writer(output_path, col, col_fps)
    else:
        out = None

    detector = initialize_blob_detector()
    tracker = KeypointTracker()
    frame_num = 0

    while col.isOpened() and ir.isOpened():
        ret1, col_frame = col.read()
        ret2, ir_frame = ir.read()
        if not ret1 or not ret2:
            break

        overlay = process_frame_pair(col_frame, ir_frame, matrix, tracker, detector, detect_keypoints, frame_num)

        if out:
            out.write(overlay)

        cv2.imshow('Live Feed', overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_num += 1

    col.release()
    ir.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

################################################################################

subject = input("Enter Subject/Person Name: ")

# Folder path
folder_path = os.path.join("Recordings", subject)

# Check if the folder exists
if subject in os.listdir("Recordings"):
    recs = os.listdir(folder_path)
    recs.sort()

    # Separate color and IR videos
    color_videos = [f for f in recs if f.startswith("color_video")]
    ir_videos = [f for f in recs if f.startswith("ir_video")]

    # Extract datetime strings and convert to datetime objects
    def extract_dt(filename):
        dt_str = filename.split("_video_")[1].replace(".avi", "")
        return datetime.strptime(dt_str, "%m-%d-%y_%H-%M-%S")

    # Get latest video by datetime
    latest_color = max(color_videos, key=extract_dt)
    latest_ir = max(ir_videos, key=extract_dt)

    # Full paths
    color_video_path = os.path.join(folder_path, latest_color)
    ir_video_path = os.path.join(folder_path, latest_ir)

    print("Latest Color Video Path:", color_video_path)
    print("Latest IR Video Path:", ir_video_path)
else:
    print("Subject not found in Recordings.")

col_image = cv2.imread(r"Custom_Results\sample_test_Color.png")
ir_image = cv2.imread(r"Custom_Results\sample_test_IR.png", cv2.IMREAD_GRAYSCALE)

marked_positions,corners,ids = detect_aruco(col_image)
print(marked_positions)

aruco_ref_pts = []
try:
    for i in range(4):
        aruco_ref_pts.append(marked_positions[i])
except Exception as e:
    print("Keypoint Missing!!")
    print(e)
aruco_ref_pts = np.array(aruco_ref_pts)

# col_image = draw_aruco(col_image, corners, ids)
# cv2.imshow("Image", col_image)

ir_image = undistort_image(ir_image, camera_matrix, dist_coeffs)


keypoints = detect_blob(ir_image)

for i in keypoints:
    cv2.circle(ir_image, (int(i.pt[0]), int(i.pt[1])), 2, (0, 255, 0), -1)

ir_ref_pts,body_points = keypoints_classifier(keypoints)
# print(ir_ref_pts)

ir_image = preprocess_ir_data(ir_image, 255)
ir_image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)


# for i in body_points:
#     cv2.drawMarker(ir_image, (i[0], i[1]), (0, 0, 255), markerType=cv2.MARKER_STAR, thickness=1,markerSize=10)
#     print(i)

im_with_keypoints = draw_blob(ir_image, keypoints, ir_ref_pts)
# cv2.imshow("Keypoints", im_with_keypoints)


# rgbtoir_aligned_img, rgbtoir_overlay = homography_transform(col_image, ir_image, aruco_ref_pts, ir_ref_pts, overlay_perc=0.5)
irtorgb_aligned_img, irtorgb_overlay, M = homography_transform(ir_image, col_image, ir_ref_pts, aruco_ref_pts)
print(col_image.shape)


# video_overlay_2(r"Recordings\Pranav2\color_video_05-06-25_13-24-35.avi",r"Recordings\Pranav2\ir_video_05-06-25_13-24-35.avi",M,detect_keypoints=True)
video_overlay_2(color_video_path,ir_video_path,M,detect_keypoints=True)
#----------------------------------------------------------------------

# Show results
# cv2.imshow("IR scaled to RGB",irtorgb_overlay)
# cv2.imshow("RGB scaled to IR",rgbtoir_overlay)

# cv2.imshow("temp", overlay)
# cv2.imwrite("Custom_Results/rgb_scaled_to_ir_image.png", rgbtoir_overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
