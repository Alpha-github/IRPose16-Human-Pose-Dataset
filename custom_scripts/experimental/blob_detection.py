import cv2
import numpy as np
from time import sleep

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change Color
params.filterByColor = True
params.blobColor = 255

# Change thresholds
params.minThreshold = 180
params.maxThreshold = 255

# Filter by Area
params.filterByArea = True
params.minArea = 1

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.1

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Read video
cap = cv2.VideoCapture(r"ir_video_take1.avi")

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define codec and create VideoWriter to save the output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output_with_keypoints.avi', fourcc, fps, (frame_width, frame_height))

while True:
    ret, im = cap.read()
    if not ret:
        break

    # Convert to grayscale for detection
    gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Detect blobs
    keypoints = detector.detect(gray_im)
    print(f"Number of keypoints detected: {len(keypoints)}")

    # Draw detected blobs as red circles
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255), 
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # # Save the frame with keypoints to output video
    # out.write(im_with_keypoints)

    # Display the frame
    sleep(0.01)
    cv2.imshow("Keypoints", im_with_keypoints)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
# out.release()
cv2.destroyAllWindows()

# im = cv2.imread(r"Custom_Results\FINAL_pranav_1_IR.png")
# gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# # Detect blobs
# keypoints = detector.detect(gray_im)

# # Draw detected blobs as red circles
# im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255), 
#                                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# # # Save the frame with keypoints to output video
# # out.write(im_with_keypoints)

# # Display the frame
# # sleep(0.01)
# cv2.imshow("Keypoints", im_with_keypoints)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
