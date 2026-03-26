import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from pathlib import Path
import logging
import os
from app.utils.model_version import current_model_name

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
model_versions = current_model_name()
LOAD_MODEL_VERSION = model_versions[os.getenv("LOAD_MODEL_VERSION", "small")]

MODEL_PATH = BASE_DIR / "models" / "v1" /f"{LOAD_MODEL_VERSION}"

logger.debug(f"Model path: {MODEL_PATH}")

# Load checkpoint
model = None
try:
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    # Recreating  model
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, checkpoint["num_classes"])

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully.")
except FileNotFoundError as e:
    logger.error("No Model found")
    



