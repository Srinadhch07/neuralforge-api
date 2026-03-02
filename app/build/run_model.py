import  torch
from torchvision import models, transforms
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

model = None
checkpoint = None
device = None

def load_model():
    global model, checkpoint, device
    if model is None:
        from .load_model_v1 import model as m, checkpoint as c, device as d
        model = m
        checkpoint = c
        device = d

def leaf_detection_model(image_path: str):
    try:
        load_model()
        img = Image.open(image_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)
        predicted_class = predicted_class.item()
        confidence = confidence.item()
        # Get class mapping safely
        idx_to_class = {v: k for k, v in checkpoint["class_to_idx"].items()}
        logger.debug("Predicted:", idx_to_class[predicted_class])
        logger.debug(f"Confidence: {confidence*100:.2f}%")
        return idx_to_class[predicted_class], confidence
    except Exception as e:
        logging.error(f"Error details:{e}")
        return "None", 0.0