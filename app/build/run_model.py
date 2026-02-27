import  torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from load_model import model, checkpoint,device
import logging

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

async def leaf_detection_model(image_path: str):
    try:
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
        print("Predicted:", idx_to_class[predicted_class])
        print(f"Confidence: {confidence*100:.2f}%")
        return idx_to_class, confidence
    except Exception as e:
        logging.error(f"Error details:{e}")
        return "None", 0.0