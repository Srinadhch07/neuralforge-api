import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from pathlib import Path
import logging

from app.utils.model_version import current_model_name

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "v2" / f"{current_model_name()}"

logger.debug(f"Model path: {MODEL_PATH}")

# Load checkpoint
model = None
try:
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    # Recreating  model
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully.")
except FileNotFoundError as e:
    logger.error("No Model found")



