import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import logging
import os
from collections import Counter


logger = logging.getLogger()
from app.utils.model_version import new_model_name

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = BASE_DIR / "models" / f"{new_model_name()}"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

def train_model(num_epochs: int):
    from .build_model import model, train_loader, train_data, device, num_classes
    logger.debug(f'Deep learning model built sucessfully.')

    targets = train_data.targets
    class_counts = Counter(targets)
    counts = [class_counts[i] for i in range(len(class_counts))]
    class_weights = torch.tensor([1.0/c for c in counts]).to(device)
    criterian = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)
    logger.info(f"Initialised traning epochs")
    for epoch in range(num_epochs):
        model.train()
        running_loss =0.0
        correct = 0
        total = 0
        #  Implementing backpropagation algorithm in neural networks
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterian(outputs, labels)
            loss.backward()
            optimizer.step()
            # statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

    torch.save({ "model_state_dict": model.state_dict(), "num_classes": num_classes, "class_to_idx": train_data.class_to_idx }, MODEL_PATH)
    logger.info("Model saved successfully.")
    return f"model training successfully completed."

train_model(1)

