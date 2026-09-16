import os
import threading
from urllib.request import urlopen

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from keras.models import load_model

load_dotenv()
ip = os.getenv("RC_CAR_IP", "192.168.137.35")
api_key = os.getenv("OPENWEATHER_API_KEY")
city = os.getenv("WEATHER_CITY", "Seoul")
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.getenv(
    "KERAS_MODEL_PATH", os.path.join(project_root, "models", "keras_model.h5")
)
labels_path = os.getenv(
    "KERAS_LABELS_PATH", os.path.join(project_root, "models", "labels.txt")
)

def get_current_weather(city_name):
    if not api_key:
        raise RuntimeError("Set OPENWEATHER_API_KEY before running this script.")
    response = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"q": city_name, "appid": api_key, "units": "metric", "lang": "kr"},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()

def format_weather_data(weather_data):
    rain = weather_data.get("rain", {}).get("1h", 0)
    formatted = (
        f"City: {weather_data['name']}, {weather_data['sys']['country']}\n"
        f"Temperature: {weather_data['main']['temp']} C\n"
        f"Feels like: {weather_data['main']['feels_like']} C\n"
        f"Min / Max: {weather_data['main']['temp_min']} / "
        f"{weather_data['main']['temp_max']} C\n"
        f"Humidity: {weather_data['main']['humidity']}%\n"
        f"Weather: {weather_data['weather'][0]['description']}\n"
        f"Wind: {weather_data['wind']['speed']} m/s\n"
        f"Rain in previous hour: {rain} mm"
    )
    return formatted, rain

def adjust_speed_based_on_rain(rain_mm):
    speed = 80 if rain_mm >= 5 else 100
    urlopen(f"http://{ip}/action?go=speed{speed}")
    return speed

weather = get_current_weather(city)
formatted_weather, rain = format_weather_data(weather)
print(formatted_weather)
print("Speed setting:", adjust_speed_based_on_rain(rain))

stream = urlopen(f"http://{ip}:81/stream")
model = load_model(model_path, compile=False)
with open(labels_path, "r", encoding="utf-8") as label_file:
    class_names = label_file.readlines()
buffer = b""
image = None
image_flag = 0

def image_process_thread():
    global image, image_flag
    while True:
        if image_flag == 1:
            model_input = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
            model_input = (model_input / 127.5) - 1
            prediction = model.predict(model_input, verbose=0)
            index = np.argmax(prediction)
            class_name = class_names[index][2:]
            confidence = float(prediction[0][index])
            if confidence >= 0.80:
                if "go" in class_name:
                    urlopen(f"http://{ip}/action?go=forward")
                elif "left" in class_name:
                    urlopen(f"http://{ip}/action?go=left")
                elif "right" in class_name:
                    urlopen(f"http://{ip}/action?go=right")
                print(class_name.strip(), f"{confidence * 100:.0f}%")
            image_flag = 0

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
            height, _, _ = image.shape
            image = image[height // 2:height - 30, :]
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            cv2.imshow("AI CAR Streaming", image)
            image_flag = 1
            if cv2.waitKey(1) == ord("q"):
                break
finally:
    urlopen(f"http://{ip}/action?go=stop")
    cv2.destroyAllWindows()
