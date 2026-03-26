import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from pathlib import Path
from app.utils.model_version import current_model_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_versions = current_model_name()
BASE_DIR = Path(__file__).resolve().parent
TEST_DATASET_PATH = BASE_DIR / "datasets" /"test"



def test_model(model_type: str):
    MODEL_TYPE = model_versions.get(model_type).lower().strip()
    version = "v1" if MODEL_TYPE == "small" else "v2" if MODEL_TYPE == "medium" else "v3"
    MODEL_PATH = BASE_DIR.parent / "models" / version / f'{MODEL_TYPE}'

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if version  ==   "v1":
        from torchvision.models import resnet18,ResNet18_Weights
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, checkpoint["num_classes"])
    elif version == "v2":
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, checkpoint["num_classes"])
    if version  ==   "v3":
        from torchvision.models import resnet50,ResNet50_Weights
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, checkpoint["num_classes"])

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load test dataset
    test_data = datasets.ImageFolder(root=TEST_DATASET_PATH, transform=transform)
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