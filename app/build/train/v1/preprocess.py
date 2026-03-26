from torchvision import datasets, transforms
from pathlib import Path
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",".."))
DATA_DIR = os.path.join(BASE_DIR, "datasets")

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

#  Preprocessing setup
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

# Data transformation
train_data = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transforms)
test_data = datasets.ImageFolder(root=TEST_DIR, transform=test_transforms)
