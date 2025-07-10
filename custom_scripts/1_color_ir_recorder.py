import cv2
import numpy as np
import threading
import queue
from pyorbbecsdk import Config, OBError, OBSensorType, OBFormat, Pipeline, FrameSet, VideoStreamProfile
from utils import frame_to_bgr_image
import datetime
import os

ESC_KEY = 27
color_queue = queue.Queue()
ir_queue = queue.Queue()
stop_event = threading.Event()

datetime_str = datetime.datetime.now().strftime("%x_%X").replace("/", "-").replace(":", "-")
print("Recording started at:", datetime_str)

def preprocess_ir_data(ir_data, max_data):
    ir_data = ir_data.astype(np.float32)
    ir_data = np.log1p(ir_data) / np.log1p(max_data) * 255
    ir_data = np.clip(ir_data, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(6, 6))
    ir_data = clahe.apply(ir_data)
    blurred = cv2.GaussianBlur(ir_data, (0, 0), 2)
    ir_data = cv2.addWeighted(ir_data, 1.5, blurred, -0.5, 0)
    return ir_data  

def record_color(folder,datetime_str):
    config = Config()
    pipeline = Pipeline()
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(1920, 0, OBFormat.RGB, 30)
        # color_profile: VideoStreamProfile = profile_list.get_default_video_stream_profile()
        config.enable_stream(color_profile)
    except Exception as e:
        print(e)
        return

    pipeline.start(config)
    out = cv2.VideoWriter(f'.\\Recordings\\{folder}\\color_video_{datetime_str}.avi', cv2.VideoWriter_fourcc(*'XVID'), 30,
                           (color_profile.get_width(), color_profile.get_height()))

    while not stop_event.is_set():
        try:
            frames: FrameSet = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                print("Failed to convert frame to image")
                continue
            color_queue.put(color_image)
            out.write(color_image)
        except Exception as e:
            print(f"Error in color thread: {e}")
            break

    out.release()
    pipeline.stop()


def record_ir(folder,datetime_str):
    config = Config()
    pipeline = Pipeline()
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.IR_SENSOR)
        try:
            ir_profile = profile_list.get_video_stream_profile(640, 0, OBFormat.Y16, 30)
        except OBError as e:
            print(e)
            ir_profile = profile_list.get_default_video_stream_profile()
        config.enable_stream(ir_profile)
    except Exception as e:
        print(e)
        return
    
    pipeline.start(config)

    VIDEO_OUTPUT = f".\\Recordings\\{folder}\\ir_video_{datetime_str}.avi"
    FPS = 30
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 576
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Codec for .avi files
    out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    while not stop_event.is_set():
        try:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            ir_frame = frames.get_ir_frame()
            if ir_frame is None:
                continue
            ir_data = np.asanyarray(ir_frame.get_data())
            width = ir_frame.get_width()
            height = ir_frame.get_height()
            ir_format = ir_frame.get_format()
            if ir_format == OBFormat.Y8:
                ir_data = np.resize(ir_data, (height, width, 1))
                data_type = np.uint8
                image_dtype = cv2.CV_8UC1
                max_data = 255
            elif ir_format == OBFormat.MJPG:
                ir_data = cv2.imdecode(ir_data, cv2.IMREAD_UNCHANGED)
                data_type = np.uint8
                image_dtype = cv2.CV_8UC1
                max_data = 255
                if ir_data is None:
                    print("decode mjpeg failed")
                    continue
                ir_data = np.resize(ir_data, (height, width, 1))
            else:
                ir_data = np.frombuffer(ir_data, dtype=np.uint16)
                data_type = np.uint16
                image_dtype = cv2.CV_16UC1
                max_data = 65535
                ir_data = np.resize(ir_data, (height, width, 1))

            cv2.normalize(ir_data, ir_data, 0, max_data, cv2.NORM_MINMAX, dtype=image_dtype)
            ir_image = ir_data.astype(data_type)
            ir_image = preprocess_ir_data(ir_image, max_data)
            ir_image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2RGB)
            ir_queue.put(ir_image)
            out.write(ir_image)
        except Exception as e:
            print(f"Error in IR thread: {e}")
            break

    out.release()
    pipeline.stop()

def display_frames():
    while not stop_event.is_set():
        if not color_queue.empty():
            color_frame = color_queue.get()
            cv2.imshow("Color Viewer", color_frame)
        if not ir_queue.empty():
            ir_frame = ir_queue.get()
            cv2.imshow("Infrared Viewer", ir_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    subject = input("Enter subject name: ")
    if subject not in os.listdir(".\\Recordings"):
        create_folder = input("Folder NOT FOUND! Do you want to create subject folder (Y/N):")
        if create_folder.lower() == "y":
            os.mkdir(f".\\Recordings\{subject}")
        else:
            print("Exiting...")
            exit()
        
    t1 = threading.Thread(target=record_color, args=(subject,datetime_str,))
    t2 = threading.Thread(target=record_ir, args=(subject,datetime_str,))
    t3 = threading.Thread(target=display_frames)

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()