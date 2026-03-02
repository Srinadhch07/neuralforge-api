from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch

from .preprocess import train_data, test_data

# Extracting dataset clasess
classes = train_data.class_to_idx
modified_class = {v:k for k,v in classes.items()}
print(f"{modified_class}")


train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
print(f"classes: {train_data.classes}")
print(f"number of training images: {len(train_data)}")
print(f"number of testing images: {len(test_data)}")

model = resnet50(weights = ResNet50_Weights.DEFAULT)

# We must update the number off layers it must equal to classes
num_features = model.fc.in_features 
num_classes = len(train_data.classes)
model.fc = nn.Linear(num_features,num_classes) 
print(model.fc)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
