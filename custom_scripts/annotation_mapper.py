import os
import json
import cv2
import numpy as np

def load_annotations(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return {int(frame): {k: v for k, v in pts.items()} for frame, pts in data.items()}

def get_video_path(json_path):
    json_filename = os.path.basename(json_path)
    subject = json_filename.split("_")[0]
    video_path = "\\".join(json_path.split("\\")[:-1]) + "\\" + json_filename.replace(".json", ".avi").replace(f"{subject}", "color_video")
    video_path = video_path.replace("_renamed","")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    return video_path

def draw_keypoints(frame, keypoints, radius=4):
    for label, coord in keypoints.items():
        if coord is not None:
            x, y = int(coord[0]), int(coord[1])
            cv2.circle(frame, (x, y), radius, (0, 255, 0), -1)
            cv2.putText(frame, str(label), (x+6, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

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

def rename_keypoints_in_json(json_path, mapping):
    with open(json_path, "r") as f:
        data = json.load(f)

    new_data = {}
    for frame, keypoints in data.items():
        new_data[frame] = {}
        for kpt_num, coords in keypoints.items():
            new_label = mapping.get(int(kpt_num), str(kpt_num))  # fallback to number if not mapped
            new_data[frame][new_label] = coords

    new_json_path = json_path.replace(".json", "_renamed.json")
    with open(new_json_path, "w") as f:
        json.dump(new_data, f, indent=4)
    
    print(f"✅ New JSON saved as: {new_json_path}")
    return new_json_path

# 🔁 Loop through pose_json directory
pose_json_root = "pose_json"
subject = input("Enter subject name: ").strip().lower()

if subject in list(map(str.lower, os.listdir(pose_json_root))):
    subject_dir = os.path.join(pose_json_root, subject)

    files = [f for f in sorted(os.listdir(subject_dir)) if f.endswith(".json")]

    for i, json_file in enumerate(files):
        print(f"{i+1}: {json_file}")

    file_no = int(input("Enter file number to display (0 to cancel): ").strip())
    if file_no < 1 or file_no > len(files):
        print("❌ Invalid selection.")
    else:
        json_file = files[file_no - 1]
        json_path = os.path.join(subject_dir, json_file)
        display_annotated_video(json_path)

        # Step: Ask for keypoint mappings
        print("\n💬 Enter keypoint label mappings:")
        with open(json_path, "r") as f:
            sample_data = json.load(f)
            sample_frame = next(iter(sample_data.values()))
            keypoint_nums = sorted(map(int, sample_frame.keys()))
        print(f"Detected keypoints: {keypoint_nums}")
        mapping = {}
        for num in keypoint_nums:
            label = input(f"Label for keypoint {num}: ").strip()
            mapping[num] = label

        # Step: Create renamed JSON file
        new_json_path = rename_keypoints_in_json(json_path, mapping)

        # # Step: View with renamed labels
        # display_annotated_video(new_json_path)
