import sys
import json
import os
import numpy as np
from PIL import Image
import onnxruntime as ort
import torch

from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton,
    QLabel, QFileDialog, QVBoxLayout
)
from PySide6.QtGui import QPixmap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PT_MODEL_PATH = os.path.join(SCRIPT_DIR, "trashnet_resnet18.pt")
ONNX_MODEL_PATH = os.path.join(SCRIPT_DIR, "outputs", "trashnet_resnet18.onnx")
META_PATH = os.path.join(SCRIPT_DIR, "outputs", "metadata.json")

with open(META_PATH) as f:
    meta = json.load(f)

CLASS_NAMES = meta["class_names"]
IMG_SIZE = meta["image_size"]
MEAN = np.array(meta["imagenet_mean"])
STD = np.array(meta["imagenet_std"])


session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])


def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    img = np.array(img).astype(np.float32) / 255.0
    img = (img - MEAN) / STD

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img.astype(np.float32)


class TrashClassifierApp(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Trash Classifier")

        self.layout = QVBoxLayout()

        self.image_label = QLabel("Upload an image")
        self.layout.addWidget(self.image_label)

        self.result_label = QLabel("")
        self.layout.addWidget(self.result_label)

        self.button = QPushButton("Select Image")
        self.button.clicked.connect(self.load_image)
        self.layout.addWidget(self.button)

        self.setLayout(self.layout)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName()

        if not file_path:
            return

        # Show image
        pixmap = QPixmap(file_path)
        self.image_label.setPixmap(pixmap.scaled(300, 300))

        # Run prediction
        input_tensor = preprocess(file_path)

        outputs = session.run(None, {"image": input_tensor})
        logits = outputs[0][0]

        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        pred_idx = np.argmax(probs)
        pred_class = CLASS_NAMES[pred_idx]
        confidence = probs[pred_idx] * 100

        self.result_label.setText(
            f"Prediction: {pred_class} ({confidence:.1f}%)"
        )


app = QApplication(sys.argv)

window = TrashClassifierApp()
window.resize(400, 400)
window.show()

sys.exit(app.exec())
