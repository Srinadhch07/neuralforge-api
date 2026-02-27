

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import torch.nn as nn
from torchvision import models

# Data collection 
train_dir = "./Data/train"
test_dir = "./Data/test"

# Processing Data
train_transforms = transforms.Compose([
    transforms.Resize((128, 128)),          
    transforms.RandomHorizontalFlip(),      
    transforms.RandomRotation(10),
    transforms.ToTensor(),                  
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   
])

test_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Data transformation
train_data = datasets.ImageFolder(root=train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(root=test_dir, transform=test_transforms)

# Extracting the Dataset classes
classes = train_data.class_to_idx
modified_class = {v:k for k,v in classes.items()}
print(modified_class)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
print("Classes:", train_data.classes)
print("Number of training images:", len(train_data))
print("Number of testing images:", len(test_data))

# building model

model = models.resnet18(pretrained=True)

# We must update the number off layers it must equal to classes
num_features = model.fc.in_features 
num_classes = len(train_data.classes)
model.fc = nn.Linear(num_features,num_classes) 
print(model.fc)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Finding and Optimising loss
import torch.nn as nn
import torch.optim as optim
criterion = nn.CrossEntropyLoss()  # multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
num_epochs = 5


# Training with loss optimization
for epoch in range(num_epochs):
    model.train()  
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()       # zero gradients
        outputs = model(images)     # forward pass
        loss = criterion(outputs, labels)  # compute loss
        loss.backward()             # backward pass
        optimizer.step()            # update weights
        # statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")


# Saving only model weights, we need to load model along with its weights later
# torch.save(model.state_dict(), "leaf_detection_model_v1.pth")
torch.save({
    "model_state_dict": model.state_dict(),
    "num_classes": num_classes,
    "class_to_idx": train_data.class_to_idx
}, "leaf_detection_model_v1.pth")
print("Model saved successfully.")
