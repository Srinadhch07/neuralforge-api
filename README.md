
# 🌿 AgroVision AI
## A deep learning–based computer vision system that identifies plant species and detects leaf diseases from images using ResNet architecture.

AgroVision AI is a deep learning–based computer vision system designed to identify plant species and detect leaf diseases from images. The system leverages transfer learning with a ResNet architecture to deliver high-accuracy classification suitable for agricultural diagnostics and smart farming applications.

The project is built as a scalable foundation for AI-driven crop health monitoring systems.

---

##  Project Overview

AgroVision AI performs multi-class image classification where each class represents a specific plant type or plant-disease combination. Given an input image of a leaf, the system:

1. Preprocesses the image
2. Extracts visual features using a pretrained convolutional neural network
3. Classifies the image into the most probable plant or disease category
4. Returns the predicted label along with confidence score

This approach enables automated plant health diagnosis using computer vision.

---

##  Core Capabilities

* Plant species identification
* Leaf disease classification
* Confidence-based prediction
* GPU and CPU compatible inference
* Transfer learning using pretrained ImageNet weights
* Structured dataset handling with ImageFolder format
* Model evaluation using standard classification metrics

---

##  System Architecture

AgroVision AI follows a standard deep learning pipeline:

Dataset → Data Augmentation → ResNet Feature Extraction → Fully Connected Layer → Softmax Classification → Prediction Output

The system is modular and can be extended to support:

* Additional plant species
* More disease categories
* API-based deployment
* Web or mobile interfaces

---

##  Tech Stack

**Programming Language**

* Python

**Deep Learning Framework**

* PyTorch

**Computer Vision Utilities**

* Torchvision
* PIL (Python Imaging Library)

**Data Handling & Metrics**

* NumPy
* Scikit-learn

**Visualization (Training Phase)**

* Matplotlib

---

##  Dataset Design

The project uses a folder-based dataset structure compatible with PyTorch's ImageFolder utility. Each folder represents a class label.

Example structure:

* train/

  * apple_healthy/
  * apple_scab/
  * grapevine_black_rot/
  * mango_healthy/
  * paddy_bacterial_leaf_blight/

* val/

* test/

This design allows dynamic class expansion without modifying model logic.

---

##  Model Evaluation

The system evaluates performance using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

These metrics ensure balanced performance across all plant and disease categories.

---

##  Potential Applications

* Smart agriculture platforms
* Crop monitoring systems
* Agricultural research tools
* Farm advisory AI assistants
* Precision farming solutions

---

##  Future Scope

AgroVision AI can be extended to include:

* Real-time mobile inference
* Disease treatment recommendation system
* Model versioning and retraining pipeline
* API-based deployment for large-scale usage
* Integration with IoT-based farm monitoring systems

---

##  Vision

AgroVision AI aims to bridge artificial intelligence and agriculture by enabling early disease detection, improving crop yield, and supporting data-driven farming decisions.

---

**AgroVision AI — Intelligent Crop Health Monitoring with Deep Learning**
