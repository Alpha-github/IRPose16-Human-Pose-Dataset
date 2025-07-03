import os
import json
import cv2
import numpy as np

def load_annotations(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return {int(frame): {k: v for k, v in pts.items()} for frame, pts in data.items()}

def get_video_path(json_path):
    # Extract subject name and timestamp from JSON filename
    json_filename = os.path.basename(json_path)
    subject = json_filename.split("_")[0]
    video_path = "\\".join(json_path.split("\\")[:-1])+"\\"+json_filename.replace(".json", ".avi").replace(f"{subject}", "color_video")

    if "named_keypoints\\" in video_path:
        video_path = video_path.replace("named_keypoints\\", "")
    print(f"Video path: {video_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    return video_path

def draw_keypoints(frame, keypoints, radius=4):
    for label, coord in keypoints.items():
        if coord is not None:
            x, y = int(coord[0]), int(coord[1])
            cv2.circle(frame, (x, y), radius, (0, 255, 0), -1)
            cv2.putText(frame, str(label), (x+6, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

def display_annotated_video(json_path):
    annotations = load_annotations(json_path)
    video_path = get_video_path(json_path)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    playing = True
    frame_index = 0

    print(f"▶️ Playing: {os.path.basename(video_path)} with {len(annotations)} annotated frames")

    while True:
        # Always set frame position and read the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Cannot read frame.")
            break

        # Draw keypoints if available
        if frame_index in annotations:
            draw_keypoints(frame, annotations[frame_index])

        # Display frame number
        cv2.putText(frame, f"Frame {frame_index}/{total_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Annotated Video", frame)

        # Wait for key input — if paused, wait indefinitely
        key = cv2.waitKey(10 if playing else 0) & 0xFF

        if key == ord('q'):
            break
        elif key == 32:  # Space bar: toggle play/pause
            playing = not playing
        elif key in (ord('a'), ord('A')):  # Backward one frame
            if frame_index > 0:
                frame_index -= 1
        elif key in (ord('d'), ord('D')):  # Forward one frame
            if frame_index < total_frames - 1:
                frame_index += 1
        elif playing:
            frame_index += 1
            if frame_index >= total_frames:
                break

    cap.release()
    cv2.destroyAllWindows()

# 🔁 Loop through pose_json directory
pose_json_root = "pose_json"
subject = input("Enter subject name: ").strip().lower()
if subject in list(map(str.lower,os.listdir(pose_json_root))):
    raw_or_named = input("Do you want to view raw or named keypoints (default - raw): ").strip().lower()
    if raw_or_named == "named":
        subject_dir = os.path.join(pose_json_root, subject, "named_keypoints")
    else:
        subject_dir = os.path.join(pose_json_root, subject)

    def extract_json(filename):
        return filename.endswith(".json")
        
    files = list(filter(extract_json,sorted(os.listdir(subject_dir))))

    for i,json_file in enumerate(files):
        if json_file.endswith(".json"):
            print(f"Processing {i+1}/{len(files)}: {json_file}")

    file_no = int(input("Enter file number to display (0 for all): ").strip())
    json_file = files[file_no-1]
    json_path = os.path.join(subject_dir, json_file)
    try:
        display_annotated_video(json_path)
    except Exception as e:
        print(f"❌ Error displaying {json_file}: {e}")
