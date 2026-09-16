import os
import re
from urllib.request import urlopen

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
ip = os.getenv("RC_CAR_IP", "192.168.137.91")
stream = urlopen(f"http://{ip}:81/stream")
buffer = b""
urlopen(f"http://{ip}/action?go=speed40")

image_dir = "images"
os.makedirs(image_dir, exist_ok=True)
existing_images = [
    filename for filename in os.listdir(image_dir)
    if re.match(r"image_\d+\.png", filename)
]
image_count = (
    max(int(re.findall(r"\d+", name)[0]) for name in existing_images) + 1
    if existing_images else 0
)

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
            cv2.imshow("AI CAR Streaming", image)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            if key == ord("s"):
                output = os.path.join(image_dir, f"image_{image_count}.png")
                cv2.imwrite(output, image)
                print("Saved:", output)
                image_count += 1
finally:
    urlopen(f"http://{ip}/action?go=stop")
    cv2.destroyAllWindows()
