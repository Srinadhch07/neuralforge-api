from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet18_Weights
import torch.nn as nn
import torch
import logging

from .preprocess import train_data, test_data

logger = logging.getLogger(__name__)

# Extracting dataset clasess
classes = train_data.class_to_idx
modified_class = {v:k for k,v in classes.items()}
logger.debug(f"{modified_class}")

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
logger.debug(f"classes: {train_data.classes}")
logger.debug(f"number of training images: {len(train_data)}")
logger.debug(f"number of testing images: {len(test_data)}")

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# We must update the number off layers it must equal to classes
num_features = model.fc.in_features 
num_classes = len(train_data.classes)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, num_classes)
# model.fc = nn.Linear(num_features,num_classes) 
logger.debug(model.fc)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
