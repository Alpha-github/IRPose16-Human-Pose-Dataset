import cv2
import numpy as np
import threading
import queue
from blob_tracking_v2 import ir_keypoint_tracking
from datetime import datetime

import os
# import inspect

# from pyorbbecsdk import Config, OBError, OBSensorType, OBFormat, Pipeline, FrameSet, VideoStreamProfile
# from pyorbbecsdk.examples.utils import frame_to_bgr_image

# from color_ir_recorder import datetime_str

# import pyorbbecsdk
# print(os.path.dirname(pyorbbecsdk.__file__))
# print(inspect.getfile(pyorbbecsdk))

color_queue = queue.Queue()
ir_queue = queue.Queue()
stop_event = threading.Event()

camera_matrix = np.array([[1.80376226e+03,0.00000000e+00,3.16914171e+02],
 [0.00000000e+00,4.31043121e+03,2.87652674e+02],
 [0.00000000e+00,0.00000000e+00,1.00000000e+00]]
)
dist_coeffs = np.array([[-3.49027957e+00,3.13640856e+01,8.12342618e-02,3.06527531e-02,-3.86407135e+02]])

def preprocess_ir_data(ir_data, max_data):
    # Convert IR data to float32 for precise processing
    ir_data = ir_data.astype(np.float32)

    # Logarithmic compression to handle brightness extremes
    ir_data = np.log1p(ir_data) / np.log1p(max_data) * 255
    ir_data = np.clip(ir_data, 0, 255).astype(np.uint8)

    # Apply CLAHE to improve contrast in dim areas
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(6, 6))
    ir_data = clahe.apply(ir_data)

    # --- Enhance whites more meaningfully ---
    # Convert to float for smooth power-law adjustment
    float_ir = ir_data.astype(np.float32) / 255.0

    # Create a mask for whites (you can tweak the threshold)
    white_mask = float_ir > 0.75

    # Apply gamma correction (>1 brightens high-intensity parts more)
    boosted_whites = float_ir ** 0.4  # gamma < 1 brightens more

    # Combine: only apply boosted whites where white_mask is True
    float_ir[white_mask] = boosted_whites[white_mask]

    # Convert back to uint8
    ir_data = (float_ir * 255).clip(0, 255).astype(np.uint8)

    # Apply sharpening (Unsharp Mask)
    blurred = cv2.GaussianBlur(ir_data, (0, 0), 2)
    ir_data = cv2.addWeighted(ir_data, 1.5, blurred, -0.5, 0)

    return ir_data

def undistort_image(image, camera_matrix, dist_coeffs):
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]
    return undistorted

def detect_aruco(image,type=cv2.aruco.DICT_6X6_250):
    aruco_dict = cv2.aruco.getPredefinedDictionary(type)
    arucoParams = cv2.aruco.DetectorParameters()
    (CORNERS, ids,_) = cv2.aruco.detectMarkers(image, aruco_dict, parameters=arucoParams)
    marker_positions = {}

    if len(CORNERS) > 0:
        ids = ids.flatten()
        for marker_corner, marker_id in zip(CORNERS, ids):
            corners = marker_corner.reshape((4, 2))
            cX, cY = np.mean(corners, axis=0).astype(int)
            marker_positions[marker_id] = (cX, cY)
    return marker_positions, CORNERS, ids

def draw_aruco(image, corners, ids):
    for (markerCorner, markerID) in zip(corners, ids):
        corners = markerCorner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners

        # convert each of the (x, y)-coordinate pairs to integers
        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))

        # draw the bounding box of the ArUCo detection
        cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
        cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
        cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
        cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

        # compute and draw the center (x, y)-coordinates of the ArUco
        # marker
        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
        cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

        cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
        print("[INFO] ArUco marker ID: {}".format(markerID))
    return image

def detect_blob(image, min_thresh=180, max_thresh=255,min_area=1,min_circularity=0.1,min_convexity=0.1,min_inertia=0.01):

    params = cv2.SimpleBlobDetector_Params()

    # Change Color
    params.filterByColor = True
    params.blobColor = 255

    # Change thresholds
    params.minThreshold = min_thresh
    params.maxThreshold = max_thresh

    # Filter by Area
    params.filterByArea = True
    params.minArea = min_area

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = min_circularity

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = min_convexity

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = min_inertia

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image)
    return keypoints

def draw_blob(image, keypoints):
    im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    for i in range(len(keypoints)):
        cv2.circle(im_with_keypoints, ir_ref_pts[i], 2, (0, 255, 0), -1)
        cv2.putText(im_with_keypoints, str(i),(ir_ref_pts[i][0], ir_ref_pts[i][1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)
    return im_with_keypoints

def keypoints_classifier(keypoints):
    # Convert keypoints to readable format
    readable_keypts = cv2.KeyPoint_convert(keypoints).astype('int16')
    # print(readable_keypts)
    top_left = min(readable_keypts, key=lambda p: (p[0] + p[1]))
    top_right = max(readable_keypts, key=lambda p: (p[0] - p[1]))  
    bottom_left = min(readable_keypts, key=lambda p: (p[0] - p[1]))  
    bottom_right = max(readable_keypts, key=lambda p: (p[0] + p[1]))

    
    ir_ref_pts = np.array([top_left, bottom_left, bottom_right, top_right])
    # print(ir_ref_pts)
    body_points = np.array([x for x in readable_keypts if x not in ir_ref_pts])
    return ir_ref_pts,body_points

def homography_transform(image1, image2, ref_pts1, ref_pts2,overlay_perc=0.3):
    # Find homography matrix
    M, _ = cv2.findHomography(ref_pts1, ref_pts2, cv2.RANSAC)

    # Warp image2 to align with image1
    aligned_img = cv2.warpPerspective(image1, M, (image2.shape[1], image2.shape[0]))

    # Overlay images
    overlay = cv2.addWeighted(image2, overlay_perc, aligned_img, 1-overlay_perc, 0)

    return aligned_img, overlay, M

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

def video_overlay_2(col_video,ir_video,matrix,detect_keypoints=False,output_path=None):
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
    
    # Constants
    TOTAL_KEYPOINTS = 18
    MAX_DISAPPEARANCE = 30
    EXCLUDE_TOPMOST = 0  # Number of topmost blobs to ignore

    # Blob detector setup
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
    detector = cv2.SimpleBlobDetector_create(params)

    # Tracking structures
    last_known_positions = {i: None for i in range(TOTAL_KEYPOINTS)}  # id: (x, y)
    inactive_counters = {i: 0 for i in range(TOTAL_KEYPOINTS)}        # id: missing_frame_count
    excluded_positions = []  # Topmost 2 blobs
    frame_num = 0

    while(col.isOpened() or ir.isOpened()):
        ret1, col_frame = col.read()
        ret2, ir_frame = ir.read()

        if not ret1 or not ret2:
            break

        ir_frame = undistort_image(ir_frame,camera_matrix, dist_coeffs)
        
        if detect_keypoints:
            _, last_known_positions, inactive_counters, excluded_positions = ir_keypoint_tracking(ir_frame, frame_num, MAX_DISAPPEARANCE, EXCLUDE_TOPMOST, detector, last_known_positions, inactive_counters, excluded_positions)
            keypoint_ids,trans_keypoints = [], []
            for key, value in last_known_positions.items():
                if value != None:
                    trans_keypoints.append(value)
                    keypoint_ids.append(key)
            trans_keypoints = np.array(trans_keypoints).reshape(-1,1,2)

        # for i in keypoints:
        #     cv2.circle(ir_frame, (int(i.pt[0]), int(i.pt[1])), 2, (0, 255, 0), -1)

        # ir_frame= cv2.drawKeypoints(ir_frame, keypoints, np.array([]), (0, 0, 255), 
        #                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Warp image2 to align with image1
        aligned_img = cv2.warpPerspective(ir_frame, matrix, (col_frame.shape[1], col_frame.shape[0]))

        if detect_keypoints:
            mapped_keypoints = np.array(cv2.perspectiveTransform(trans_keypoints, matrix).reshape(-1,2))
            valid_keypoints = 0
            for id in range(len(keypoint_ids)):
                x, y = (int(mapped_keypoints[id,0]), int(mapped_keypoints[id,1]))
                cv2.circle(col_frame, (x, y), 5, (0, 255, 255), -1)
                # cv2.circle(col_frame, (int(mapped_keypoints[i,0]), int(mapped_keypoints[i,1])), 5, (0, 255, 255), -1)
                cv2.putText(col_frame, str(keypoint_ids[id]), (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                valid_keypoints += 1

            # if valid_keypoints < TOTAL_KEYPOINTS - EXCLUDE_TOPMOST:
            #     print(frame_num, "Valid keypoints:", valid_keypoints)
        # Overlay images
        overlay = cv2.addWeighted(col_frame, 1, aligned_img, 0, 0)

        if output_path!=None:
            out.write(overlay)

        cv2.imshow('Live Feed', overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_num += 1
    col.release()
    ir.release()

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

# im_with_keypoints = draw_blob(ir_image, keypoints)
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
