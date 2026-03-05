from pathlib import Path
import logging

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

def new_model_name():
    if not MODEL_DIR.exists():
        return "leaf_detection_model_v1.pth"
    version = len([f for f in MODEL_DIR.iterdir() if f.is_file() ]) + 1
    return  f"leaf_detection_model_v{version}.pth"

def current_model_name():
    if not MODEL_DIR.exists():
        return "leaf_detection_model_v1.pth"
    version = len([f for f in MODEL_DIR.iterdir() if f.is_file()])
    if version == 0:
        version += 1
    logger.debug(f"leaf_detection_model_v{version}.pth")
    return f"leaf_detection_model_v{version}.pth"

def is_model_exists():
    exists = True  if len([f for f in MODEL_DIR.iterdir() if f.is_file()]) else False
    logger.info("Existing model detected.") if exists else logger.info("No model detected, downloading model...")
    return exists
