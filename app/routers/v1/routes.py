from fastapi import APIRouter, UploadFile, HTTPException, Form, Request
from uuid import uuid4
import os
import logging
import shutil
from typing import Literal
from app.tasks.dl_tasks import predict_leaf_task
from app.schemas.v1.schemas import ModelType

logger = logging.getLogger(__name__)

UPLOAD_DIR = "uploads"
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter(prefix="/v1/model")

@router.post( "/predict", name = "Inference model", summary="Predict plant species and disease", description= "", status_code=202)
async def predict(request: Request, file: UploadFile, model: Literal["small","medium","large"] = Form("small")):

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, details="Only Images are allowed.")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400,"Invalid file format")
    field_id = f'{uuid4()}_{file.filename}'
    file_path = os.path.join(UPLOAD_DIR, field_id)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    task_payload = {
        "path": file_path,
        "model": model
    }
    task = predict_leaf_task.delay((task_payload))
    return { "status": True, "message": "deep learning learning route", "data": { "taskId": task.id, "status": "processing" }}


