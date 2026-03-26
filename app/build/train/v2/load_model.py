import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from pathlib import Path
import logging
import os
from app.utils.model_version import current_model_name

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent.parent
model_versions = current_model_name()
LOAD_MODEL_VERSION = model_versions[os.getenv("LOAD_MODEL_VERSION", "large")]

MODEL_PATH = BASE_DIR / "models" / "v3" / f"{LOAD_MODEL_VERSION}"

logger.debug(f"Model path: {MODEL_PATH}")

# Load checkpoint
model = None
try:
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # Recreating  model
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully.")
except FileNotFoundError as e:
    logger.error("No Model found")



