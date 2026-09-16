import os
import pathlib
import threading
import time
from urllib.request import urlopen

import cv2
import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()
if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath
os.chdir(os.path.dirname(os.path.abspath(__file__)))
ip = os.getenv("RC_CAR_IP", "192.168.137.220")
model_path = os.getenv("YOLO_MODEL_PATH", "best.pt")
stream = urlopen(f"http://{ip}:81/stream")
urlopen(f"http://{ip}/action?go=speed100")
model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path)

buffer = b""
frame = None
thread_frame = None
image_flag = 0
thread_image_flag = 0
car_state = "go"
yolo_state = "go"

def yolo_thread():
    global image_flag, thread_frame, thread_image_flag, yolo_state
    while True:
        if image_flag == 1 and frame is not None:
            thread_frame = frame.copy()
            detections = model(thread_frame).pandas().xyxy[0]
            if detections.empty:
                yolo_state = "go"
                urlopen(f"http://{ip}/action?go=speed100")
            else:
                for _, detection in detections.iterrows():
                    x1, y1, x2, y2 = detection[
                        ["xmin", "ymin", "xmax", "ymax"]
                    ].astype(int).values
                    label = detection["name"]
                    confidence = detection["confidence"]
                    if "stop" in label and confidence > 0.5:
                        yolo_state = "stop"
                    elif "slow" in label and confidence > 0.5:
                        yolo_state = "go"
                        urlopen(f"http://{ip}/action?go=speed80")
                    elif "Uturn" in label and confidence > 0.5:
                        yolo_state = "Uturn"
                        urlopen(f"http://{ip}/action?go=speed80")
                    color = np.random.randint(0, 256, size=3).tolist()
                    cv2.rectangle(thread_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        thread_frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )
            thread_image_flag = 1
            image_flag = 0
        time.sleep(0.2)

def image_process_thread():
    global image_flag
    while True:
        if image_flag == 1:
            if car_state == "go" and yolo_state == "go":
                urlopen(f"http://{ip}/action?go=forward")
            elif car_state == "right" and yolo_state == "go":
                urlopen(f"http://{ip}/action?go=right")
            elif car_state == "left" and yolo_state == "go":
                urlopen(f"http://{ip}/action?go=left")
            elif yolo_state == "Uturn":
                urlopen(f"http://{ip}/action?go=turn_right")
            elif yolo_state == "stop":
                urlopen(f"http://{ip}/action?go=stop")
            image_flag = 0

threading.Thread(target=yolo_thread, daemon=True).start()
threading.Thread(target=image_process_thread, daemon=True).start()

try:
    while True:
        buffer += stream.read(4096)
        head = buffer.find(b"\xff\xd8")
        end = buffer.find(b"\xff\xd9")
        if head > -1 and end > -1:
            jpg = buffer[head:end + 2]
            buffer = buffer[end + 2:]
            image = cv2.imdecode(
                np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED
            )
            frame = cv2.resize(image, (640, 480))
            height, width, _ = image.shape
            lane_roi = image[height // 2:height - 30, 10:width - 10]
            mask = cv2.inRange(
                lane_roi, np.array([0, 0, 0]), np.array([255, 255, 80])
            )
            moments = cv2.moments(mask)
            center_x = int(moments["m10"] / moments["m00"]) if moments["m00"] else 0
            center_offset = lane_roi.shape[1] // 2 - center_x
            if center_offset < -70:
                car_state = "right"
            elif center_offset > 70:
                car_state = "left"
            else:
                car_state = "go"
            image_flag = 1
            if thread_image_flag == 1 and thread_frame is not None:
                cv2.imshow("YOLO Detection", thread_frame)
                thread_image_flag = 0
            cv2.imshow("Lane Mask", mask)
            if cv2.waitKey(1) == ord("q"):
                break
finally:
    urlopen(f"http://{ip}/action?go=stop")
    cv2.destroyAllWindows()
