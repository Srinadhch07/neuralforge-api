from fastapi import APIRouter, UploadFile, HTTPException, File
from uuid import uuid4
import os
import logging
import shutil
from celery.result import AsyncResult
from pathlib import Path

from app.config.celery_app import celery
from app.tasks.dl_tasks import upload_dataset_task, train_model_task
from app.documentation import train_routes_documentation
from app.schemas.v2.train_schema import TrainModel as TrainModel_v2
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v2/dataset")

@router.post("/upload", name ="upload_dataset", summary="Train model", description=train_routes_documentation.upload_dataset, status_code=202)
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP files allowed")
    TEMP_UPLOAD_PATH = Path("temp_uploads").resolve()
    TEMP_UPLOAD_PATH.mkdir(parents=True, exist_ok=True)
    temp_zip_path = TEMP_UPLOAD_PATH / file.filename
    with open(temp_zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    await file.close()
    task = upload_dataset_task.delay(str(temp_zip_path.resolve()))
    return { "status": True, "message": "Uplaoding dataset", "data": { "taskId": task.id, "status": "processing" }}

@router.post('/train', name="Train model", summary="Trains the model", description="")
async def train_model(payload: TrainModel_v2):
    if payload.epochs < 1:
        raise HTTPException(404,"Epochs cannot be less than 1")
    model_name = payload.model_name.strip().replace(" ","_").lower()
    task_payload = {
        "name": model_name,
        "epochs": payload.epochs,
        "model": payload.model,
    }
    task = train_model_task.delay(task_payload)
    return { "status": True, "message": "Retraining  deep learning model", "data": { "taskId": task.id, "status": "processing" }}


    