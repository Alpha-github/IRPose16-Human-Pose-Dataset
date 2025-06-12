import numpy as np
import cv2

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

def draw_blob(image, keypoints, ir_ref_pts):
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