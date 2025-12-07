import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Configuration
ROOT_DIR = "pose_json"
OUTPUT_DIR = os.path.join(ROOT_DIR,"coco_dataset")
TRAIN_IMAGE_DIR = os.path.join(OUTPUT_DIR, "train/images")
VAL_IMAGE_DIR = os.path.join(OUTPUT_DIR, "val/images")
ANNOTATION_DIR = os.path.join(OUTPUT_DIR, "annotations")

os.makedirs(TRAIN_IMAGE_DIR, exist_ok=True)
os.makedirs(VAL_IMAGE_DIR, exist_ok=True)
os.makedirs(ANNOTATION_DIR, exist_ok=True)

# Create base COCO structures
def get_base_coco():
    return {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "person"}]
    }

coco_train = get_base_coco()
coco_val = get_base_coco()

image_id = 0
annotation_id = 0

def get_bbox_from_keypoints(keypoints):
    visible = [keypoints[i:i+3] for i in range(0, len(keypoints), 3) if keypoints[i+2] > 0]
    if not visible:
        return [0, 0, 0, 0], 0
    coords = np.array([[pt[0], pt[1]] for pt in visible])
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    area = float((x_max - x_min) * (y_max - y_min))
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)], area

def convert_keypoints_dict_to_list(keypoint_dict, keypoint_names):
    keypoints = []
    num_keypoints = 0
    for name in keypoint_names:
        if name in keypoint_dict:
            x, y = keypoint_dict[name]
            keypoints.extend([x, y, 2])  # visibility = 2
            num_keypoints += 1
        else:
            keypoints.extend([0, 0, 0])
    return keypoints, num_keypoints

def process_video(video_path, kp_path, split):
    global image_id, annotation_id

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with open(kp_path, 'r') as f:
        kp_data = json.load(f)

    sample_frame = next(iter(kp_data.values()))
    keypoint_names = list(sample_frame.keys())

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            continue

        frame_key = str(frame_idx)
        if frame_key not in kp_data:
            continue

        keypoint_dict = kp_data[frame_key]
        keypoints, num_keypoints = convert_keypoints_dict_to_list(keypoint_dict, keypoint_names)
        bbox, area = get_bbox_from_keypoints(keypoints)

        file_name = f"{image_id:012d}.jpg"
        save_path = os.path.join(TRAIN_IMAGE_DIR if split == "train" else VAL_IMAGE_DIR, file_name)
        cv2.imwrite(save_path, frame)

        image_dict = {
            "file_name": file_name,
            "height": height,
            "width": width,
            "id": image_id
        }

        ann_dict = {
            "segmentation": [],
            "keypoints": keypoints,
            "num_keypoints": num_keypoints,
            "area": area,
            "iscrowd": 0,
            "image_id": image_id,
            "bbox": bbox,
            "category_id": 1,
            "id": annotation_id
        }

        if split == "train":
            coco_train["images"].append(image_dict)
            coco_train["annotations"].append(ann_dict)
        else:
            coco_val["images"].append(image_dict)
            coco_val["annotations"].append(ann_dict)

        image_id += 1
        annotation_id += 1

    cap.release()

# Collect all (video, keypoint json) pairs
video_pairs = []

for subject in os.listdir(ROOT_DIR):
    subject_path = os.path.join(ROOT_DIR, subject)
    if not os.path.isdir(subject_path):
        continue
    kp_dir = os.path.join(subject_path, "named_keypoints")
    if not os.path.exists(kp_dir):
        continue
    for kp_file in os.listdir(kp_dir):
        if not kp_file.endswith(".json"):
            continue
        video_file = kp_file.replace(".json", ".avi").replace(subject + "_", "color_video_")
        video_path = os.path.join(subject_path, video_file)
        kp_path = os.path.join(kp_dir, kp_file)
        if os.path.exists(video_path):
            video_pairs.append((video_path, kp_path))

# Split into train and val (80-20 split)
train_pairs, val_pairs = train_test_split(video_pairs, test_size=0.2, random_state=42)

# Process videos
for video_path, kp_path in tqdm(train_pairs, desc="Processing Train"):
    process_video(video_path, kp_path, "train")

for video_path, kp_path in tqdm(val_pairs, desc="Processing Val"):
    process_video(video_path, kp_path, "val")

# Save JSON files
with open(os.path.join(ANNOTATION_DIR, "person_keypoints_train.json"), "w") as f:
    json.dump(coco_train, f, indent=4)

with open(os.path.join(ANNOTATION_DIR, "person_keypoints_val.json"), "w") as f:
    json.dump(coco_val, f, indent=4)
