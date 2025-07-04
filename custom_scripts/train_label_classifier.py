import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_keypoint_data(json_folder):
    """
    Loads all JSON files from a folder. Each file is assumed to be a dict
    with {frame_number: {label: (x, y)}}
    """
    import json
    keypoints_per_file = []

    for root, sub, files in os.walk(json_folder):
        for file in files:
            if not file.endswith(".json") or "renamed" in file or ("Aravindhan" not in file 
                                                                   and "Sidharth" not in file
                                                                   and "Mugheesh" not in file):
                continue
            print(file)
            path = os.path.join(root, file)
            with open(path, 'r') as f:
                data = json.load(f)
                frames = sorted((int(k), v) for k, v in data.items())
                keypoints_per_file.append([v for _, v in frames])

    return keypoints_per_file  # List of list of dict[label] = (x, y)

def extract_motion_features(keypoints_sequence):
    """
    Takes a sequence of frame keypoints (list of dicts)
    Returns feature array [x_prev, y_prev, dx, dy] and labels
    """
    X = []
    y = []

    for t in range(1, len(keypoints_sequence)):
        prev_frame = keypoints_sequence[t-1]
        curr_frame = keypoints_sequence[t]

        for label in prev_frame:
            prev = prev_frame[label]
            curr = curr_frame.get(label)

            if prev is None or curr is None:
                continue

            dx = curr[0] - prev[0]
            dy = curr[1] - prev[1]
            X.append([prev[0], prev[1], dx, dy])
            y.append(label)

    return np.array(X), np.array(y)

def train_classifier(json_folder, save_path="models/motion_label_classifier.pkl"):
    all_X = []
    all_y = []

    all_keypoints = load_keypoint_data(json_folder)

    for sequence in all_keypoints:
        X_seq, y_seq = extract_motion_features(sequence)
        all_X.append(X_seq)
        all_y.append(y_seq)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    print(f"Training samples: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, save_path)
    print(f"✅ Model saved to {save_path}")

if __name__ == "__main__":
    json_folder = "pose_json"  # Update with your real JSON folder path
    train_classifier(json_folder)
