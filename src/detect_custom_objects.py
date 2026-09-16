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
ip = os.getenv("RC_CAR_IP", "192.168.137.91")
model_path = os.getenv("YOLO_MODEL_PATH", "best.pt")
stream = urlopen(f"http://{ip}:81/stream")
urlopen(f"http://{ip}/action?go=speed40")
model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path)

buffer = b""
frame = None
frame_lock = threading.Lock()
stop_event = threading.Event()

def stream_receiver():
    global buffer, frame
    while not stop_event.is_set():
        buffer += stream.read(4096)
        head = buffer.find(b"\xff\xd8")
        end = buffer.find(b"\xff\xd9")
        if head > -1 and end > -1:
            jpg = buffer[head:end + 2]
            buffer = buffer[end + 2:]
            image = cv2.imdecode(
                np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED
            )
            with frame_lock:
                frame = cv2.resize(image, (640, 480))

def object_detection():
    last_detection_time = 0.0
    detection_interval = 0.3
    while not stop_event.is_set():
        with frame_lock:
            current_frame = frame.copy() if frame is not None else None
        now = time.time()
        if current_frame is not None and now - last_detection_time >= detection_interval:
            detections = model(current_frame).pandas().xyxy[0]
            last_detection_time = now
            for _, detection in detections.iterrows():
                x1, y1, x2, y2 = detection[
                    ["xmin", "ymin", "xmax", "ymax"]
                ].astype(int).values
                label = detection["name"]
                confidence = detection["confidence"]
                color = np.random.randint(0, 256, size=3).tolist()
                cv2.rectangle(current_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    current_frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
            cv2.imshow("frame", current_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break
        time.sleep(detection_interval)

receiver_thread = threading.Thread(target=stream_receiver)
detection_thread = threading.Thread(target=object_detection)
try:
    receiver_thread.start()
    detection_thread.start()
    while not stop_event.is_set():
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
finally:
    stop_event.set()
    receiver_thread.join()
    detection_thread.join()
    urlopen(f"http://{ip}/action?go=stop")
    cv2.destroyAllWindows()
