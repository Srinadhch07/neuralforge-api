from fastapi import APIRouter, UploadFile, HTTPException
from uuid import uuid4
import os
import logging
import shutil
from celery.result import AsyncResult
from pydantic import BaseModel

from app.config.celery_app import celery
from app.tasks.dl_tasks import predict_leaf_task

logger = logging.getLogger(__name__)

UPLOAD_DIR = "uploads"
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter(prefix="/v1/model", tags=["Inference"])

@router.post( "/predict", name = "Inference model", summary="Predict plant species and disease", description= "", status_code=202
)
async def predict(file: UploadFile):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, details="Only Images are allowed.")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400,"Invalid file format")
    field_id = f'{uuid4()}_{file.filename}'
    file_path = os.path.join(UPLOAD_DIR, field_id)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    task = predict_leaf_task.delay((file_path))
    return { "status": True, "message": "deep learning learning route", "data": { "taskId": task.id, "status": "processing" }}

@router.get("/task-status/{task_id}")
def task_status(task_id: str):
    result = AsyncResult(task_id, app=celery)
    return {
        "status": True, "message": "Task details", "data" : { "task_id": task_id, "state": result.state, "result": result.result if result.successful() else None
        }
    }
