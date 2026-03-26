from fastapi import APIRouter, UploadFile, HTTPException, Form, Request
from uuid import uuid4
import os
import logging
from typing import Literal

from app.tasks.dl_tasks import predict_leaf_task
from app.services.s3_service import upload_file_to_s3, delete_file_from_s3
# from app.schemas.v1.schemas import 
logger = logging.getLogger(__name__)

UPLOAD_DIR = "uploads"
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter(prefix="/v2/model")

@router.post( "/predict", name = "Inference model", summary="Predict plant species and disease", description= "", status_code=202, )
async def predict(request: Request, file: UploadFile, model: Literal["small","medium","large"] = Form("small")):

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, details="Only Images are allowed.")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400,"Invalid file format")
    image_url = None
    file_bytes = await file.read()
    MAX_SIZE = 20 * 1024 * 1024

    if len(file_bytes) > MAX_SIZE:
        raise  HTTPException(400, "File is too large")
    file_path = upload_file_to_s3(
        file_bytes = file_bytes,
        filename=file.filename,
        folder="uploaded_images",
        content_type=file.content_type
    )
    logger.debug(f"Image url: {file_path}")
    task_payload = {
        "path": file_path,
        "model": model
    }
    task = predict_leaf_task.delay((task_payload))
    return { "status": True, "message": "deep learning learning route", "data": { "taskId": task.id, "status": "processing" }}
