import os

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Load your pre-trained model
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.getenv(
    "KERAS_MODEL_PATH", os.path.join(project_root, "models", "keras_model.h5")
)
model = load_model(model_path, compile=False)

# Load class names from labels file
labels_path = os.getenv(
    "KERAS_LABELS_PATH", os.path.join(project_root, "models", "labels.txt")
)
with open(labels_path, "r", encoding="utf-8") as labels_file:
    class_names = labels_file.readlines()

# Confirm that TensorFlow and the model have loaded correctly
print('TensorFlow is installed correctly.')
print('Model loaded from:', model_path)
print('Class names:', class_names)
