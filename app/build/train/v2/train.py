import torch
import torch.nn as nn
import torch.optim as optim
import os
from pathlib import Path

from .build_model import model, train_loader, train_data, device, num_classes
print(f'Model built')
from app.utils.model_version import new_model_name

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / f"{new_model_name()}"

criterian = nn.CrossEntropyLoss()
optmizer = optim.Adam(model.parameters(), lr=0.001)

# TODO: c30 - update the epochs to 20
num_epochs = 5
print(f"Started epochs")
for epoch in range(num_epochs):
    model.train()
    running_loss =0.0
    correct = 0
    total = 0
    #  Implementing backpropagation algorithm in neural networks
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optmizer.zero_grad()
        outputs = model(images)
        loss = criterian(outputs, labels)
        loss.backward()
        optmizer.step()
        # statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

torch.save({ "model_state_dict": model.state_dict(), "num_classes": num_classes, "class_to_idx": train_data.class_to_idx }, MODEL_PATH)

print("Model saved successfully.")



