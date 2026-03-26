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

def large_model():
    global model, checkpoint, device
    if model is None:
        from app.build.train.v2.load_model import model as m , checkpoint as c, device as d
        model, checkpoint, device = m,c,d
        logger.info("Initialised Large Model")

def medium_model():
    global model, checkpoint, device
    if model is None:
        from app.build.train.v3.load_model import model as m, checkpoint as c, device as d
        model, checkpoint, device = m,c,d
        logger.info("Initialised Medium Model")
        
def small_model():
    global model, checkpoint, device
    if model is None:
        from .train.v1.load_model_v1 import model as m, checkpoint as c, device as d
        model, checkpoint, device = m,c,d
        logger.info("Initialised Small Model")

def load_model(model_type: str = "small"):
    small_model() if model_type == "small" else medium_model() if model_type == "medium" else large_model()

def leaf_detection_model(image_input, model_type: str = "small"):
    try:
        
        load_model(model_type)
        if isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            img = Image.open(image_input).convert("RGB")
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