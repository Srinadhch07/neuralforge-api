import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load checkpoint
checkpoint = torch.load("leaf_detection_model_v1.pth", map_location=device)

# Recreate model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, checkpoint["num_classes"])
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# Transform (MUST match training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load test dataset
test_data = datasets.ImageFolder(root="./data/test", transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Accuracy
accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"\nAccuracy: {accuracy*100:.2f}%\n")

# Classification report
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=test_data.classes))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))