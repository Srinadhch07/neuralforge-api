import os
import requests
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_URL = "https://huggingface.co/sriandhch03/agrovision-ai/resolve/main/leaf_detection_model_v4.pth"
MODEL_PATH = BASE_DIR / "models" / "v1" / "leaf_detection_model_v4.pth"

def download_model():
    if not os.path.exists(MODEL_PATH):
        logger.info("Downloading model …")
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        logger.info("Model ready.")
    else:
        logger.debug(f"{MODEL_PATH}")

