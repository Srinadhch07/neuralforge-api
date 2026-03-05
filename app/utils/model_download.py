import os
import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_URL = "https://huggingface.co/sriandhch03/agrovision-ai/resolve/main/leaf_detection_model_v4.pth"

MODEL_PATH = BASE_DIR / "models" /"leaf_detection_model_v4.pth"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model …")
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("Model ready.")
    else:
        print(f"{MODEL_PATH}")
download_model()
