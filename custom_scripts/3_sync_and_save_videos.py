import os
import cv2
import shutil

def sync_and_trim_videos(recordings_dir, trimmed_dir):
    os.makedirs(trimmed_dir, exist_ok=True)

    for subject in os.listdir(recordings_dir):
        subject_path = os.path.join(recordings_dir, subject)
        if not os.path.isdir(subject_path):
            continue

        subject_trimmed_path = os.path.join(trimmed_dir, subject)
        os.makedirs(subject_trimmed_path, exist_ok=True)

        color_videos = sorted([f for f in os.listdir(subject_path) if f.startswith("color_video")])
        ir_videos = sorted([f for f in os.listdir(subject_path) if f.startswith("ir_video")])

        for col_file, ir_file in zip(color_videos, ir_videos):
            col_path = os.path.join(subject_path, col_file)
            ir_path = os.path.join(subject_path, ir_file)

            col_cap = cv2.VideoCapture(col_path)
            ir_cap = cv2.VideoCapture(ir_path)

            col_frames = int(col_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ir_frames = int(ir_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            col_fps = int(col_cap.get(cv2.CAP_PROP_FPS))
            ir_fps = int(ir_cap.get(cv2.CAP_PROP_FPS))
            col_width = int(col_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            col_height = int(col_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ir_width = int(ir_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ir_height = int(ir_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            min_frames = min(col_frames, ir_frames)

            if col_frames > ir_frames:
                # Trim color video
                for _ in range(col_frames - ir_frames):
                    col_cap.read()
                trimmed_col_path = os.path.join(subject_trimmed_path, col_file)
                writer = cv2.VideoWriter(trimmed_col_path, fourcc, col_fps, (col_width, col_height))
                for _ in range(min_frames):
                    ret, frame = col_cap.read()
                    if not ret:
                        break
                    writer.write(frame)
                writer.release()
                print(f"✂️ Trimmed color video saved: {trimmed_col_path}")
                
                # Copy IR video as-is
                copied_ir_path = os.path.join(subject_trimmed_path, ir_file)
                shutil.copy(ir_path, copied_ir_path)
                print(f"📥 Copied IR video: {copied_ir_path}")

            elif ir_frames > col_frames:
                # Trim IR video
                for _ in range(ir_frames - col_frames):
                    ir_cap.read()
                trimmed_ir_path = os.path.join(subject_trimmed_path, ir_file)
                writer = cv2.VideoWriter(trimmed_ir_path, fourcc, ir_fps, (ir_width, ir_height))
                for _ in range(min_frames):
                    ret, frame = ir_cap.read()
                    if not ret:
                        break
                    writer.write(frame)
                writer.release()
                print(f"✂️ Trimmed IR video saved: {trimmed_ir_path}")
                
                # Copy color video as-is
                copied_col_path = os.path.join(subject_trimmed_path, col_file)
                shutil.copy(col_path, copied_col_path)
                print(f"📥 Copied color video: {copied_col_path}")

            else:
                # Videos are already in sync — just copy both
                shutil.copy(col_path, os.path.join(subject_trimmed_path, col_file))
                shutil.copy(ir_path, os.path.join(subject_trimmed_path, ir_file))
                print(f"✅ Both videos already in sync for: {col_file}, {ir_file}")

            col_cap.release()
            ir_cap.release()
        print("----------------------------------------------------------------------")
# Run it
sync_and_trim_videos("Recordings", "Synced_Recordings")
