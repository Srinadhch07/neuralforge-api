import torch
import torch.nn as nn
from torchvision import models, transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load checkpoint
checkpoint = torch.load("leaf_detection_model_v1.pth", map_location=device)
# Recreate model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, checkpoint["num_classes"])

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()
print("Model loaded successfully.")



