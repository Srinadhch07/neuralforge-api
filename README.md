# 🧠 NeuralForge

> A production-ready deep learning API platform for custom image classification — without third-party AI dependency.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)
![Celery](https://img.shields.io/badge/Celery-5.x-37814A?style=flat-square)
![Redis](https://img.shields.io/badge/Redis-7.x-red?style=flat-square&logo=redis)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 📌 What is NeuralForge?

NeuralForge is a customizable deep learning API platform that lets businesses train, version, and deploy image classification models on their own data — with no dependency on OpenAI, Google, or any third-party AI API.

Built for **agritech**, **manufacturing**, and **healthcare** companies that have proprietary image data and need custom AI solutions they fully control.

---

## ✨ Features

- 🏗️ **Three model tiers** — Small, Medium, Large — pick based on your accuracy vs compute tradeoff
- ⚡ **Async training jobs** via Celery + Redis — non-blocking, queue-based job processing
- 📦 **Dataset management pipeline** — upload, organize, and version your image datasets
- 🔁 **Model versioning** — track, compare, and serve multiple trained model versions
- ☁️ **Flexible storage** — supports both cloud and local storage for datasets and model artifacts
- 🤗 **HuggingFace integration** — pretrained weights and model hub compatibility
- 📄 **Auto-generated API docs** — FastAPI's built-in Swagger UI out of the box

---

## 🏛️ Architecture

```
Client Request
      │
      ▼
  FastAPI Server
      │
      ├──── Dataset Upload ──────► Storage (Local / Cloud)
      │
      ├──── Train Job Request ───► Celery Worker (via Redis Queue)
      │                                   │
      │                                   ▼
      │                          PyTorch Training Loop
      │                          (ResNet-18 / EfficientNet-B0 / ResNet-50)
      │                                   │
      │                                   ▼
      │                          Model Artifact Storage
      │                          + Version Registry
      │
      └──── Inference Request ───► Trained Model ──► Prediction Response
```

---

## 🧩 Model Tiers

| Tier | Model | Use Case |
|------|-------|----------|
| **Small** | ResNet-18 | Fast inference, low compute, simple classification tasks |
| **Medium** | EfficientNet-B0 | Balanced accuracy and speed, most production use cases |
| **Large** | ResNet-50 | High accuracy, complex multi-class classification |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| API Framework | FastAPI |
| Task Queue | Celery |
| Message Broker | Redis |
| Deep Learning | PyTorch |
| Pretrained Models | HuggingFace Transformers / timm |
| Storage | Local FS / Cloud (S3-compatible) |
| Model Versioning | Custom registry |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Redis server running locally or via Docker
- (Optional) CUDA-compatible GPU for faster training

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/neuralforge.git
cd neuralforge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit your config
nano .env
```

Key environment variables:

```env
REDIS_URL=redis://localhost:6379/0
STORAGE_BACKEND=local          # or "s3"
MODEL_STORAGE_PATH=./models
DATASET_STORAGE_PATH=./datasets
HF_TOKEN=your_huggingface_token
```

### Running the Platform

```bash
# Start Redis (if not already running)
redis-server

# Start Celery worker
celery -A app.worker worker --loglevel=info

# Start FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: `http://localhost:8000/docs`

---

## 📡 API Overview

### Dataset Management

```http
POST   /api/v1/datasets/upload        # Upload a zipped image dataset
GET    /api/v1/datasets               # List all datasets
DELETE /api/v1/datasets/{dataset_id}  # Delete a dataset
```

### Training

```http
POST   /api/v1/train                  # Submit a training job
GET    /api/v1/jobs/{job_id}          # Check job status
GET    /api/v1/jobs                   # List all training jobs
```

**Training request example:**

```json
{
  "dataset_id": "ds_abc123",
  "model_tier": "medium",
  "epochs": 20,
  "learning_rate": 0.001,
  "batch_size": 32
}
```

### Model Management

```http
GET    /api/v1/models                      # List all trained models
GET    /api/v1/models/{model_id}           # Get model details + metrics
DELETE /api/v1/models/{model_id}           # Delete a model version
```

### Inference

```http
POST   /api/v1/predict/{model_id}          # Run inference on an image
```

**Inference request:**

```bash
curl -X POST "http://localhost:8000/api/v1/predict/model_abc123" \
  -H "X-API-Key: your_api_key" \
  -F "file=@plant_leaf.jpg"
```

**Response:**

```json
{
  "model_id": "model_abc123",
  "predictions": [
    { "label": "healthy", "confidence": 0.94 },
    { "label": "rust_disease", "confidence": 0.05 },
    { "label": "blight", "confidence": 0.01 }
  ],
  "inference_time_ms": 38
}
```

---

## 📂 Project Structure

```
neuralforge/
├── app/
│   ├── main.py              # FastAPI app entrypoint
│   ├── worker.py            # Celery worker config
│   ├── routers/
│   │   ├── datasets.py      # Dataset upload & management
│   │   ├── training.py      # Training job endpoints
│   │   ├── models.py        # Model registry endpoints
│   │   └── inference.py     # Prediction endpoints
│   ├── services/
│   │   ├── trainer.py       # PyTorch training logic
│   │   ├── model_registry.py
│   │   └── storage.py       # Storage abstraction layer
│   ├── models/
│   │   ├── resnet18.py
│   │   ├── efficientnet_b0.py
│   │   └── resnet50.py
│   └── utils/
│       ├── dataset_utils.py
│       └── versioning.py
├── tests/
├── requirements.txt
├── .env.example
└── docker-compose.yml
```

---

## 🐳 Docker (Optional)

```bash
# Start everything with Docker Compose
docker-compose up --build
```

---

## 🔐 Authentication

NeuralForge uses API key-based authentication. Pass your key in the request header:

```http
X-API-Key: your_api_key_here
```

---

## 🎯 Real-World Use Case

> **Agritech Plant Disease Detection**
>
> An agritech company uploaded 5,000 labeled leaf images across 8 disease categories. Using the Large tier (ResNet-50), they trained a custom model achieving 96% validation accuracy in under 2 hours — deployed entirely on their own infrastructure with no third-party API dependency.

---

## 🗺️ Roadmap

- [ ] Web-based dashboard for dataset and model management
- [ ] Auto hyperparameter tuning
- [ ] Multi-GPU training support
- [ ] ONNX model export
- [ ] Webhook notifications for job completion
- [ ] Role-based access control (RBAC)

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

```bash
# Run tests
pytest tests/

# Lint
flake8 app/
```

---

## 📄 License

MIT © [Srinadh](https://github.com/srinadhch07)

---

## 👤 Author

**Srinadh** — AIML Engineer  
📝 [Medium](https://medium.com/@srinadhch07) | 💼 [LinkedIn]([https://linkedin.com/in/](https://www.linkedin.com/in/srinadhch07/)) | 🐙 [GitHub](https://github.com/srinadhch07)

---

> *NeuralForge — Train on your data. Deploy on your terms.*
