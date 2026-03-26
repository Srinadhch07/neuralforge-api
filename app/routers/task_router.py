from fastapi import APIRouter, HTTPException
import logging
from celery.result import AsyncResult

from app.config.celery_app import celery

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/task-status/{task_id}")
async def task_details(task_id : str):
    result = AsyncResult(task_id, app=celery)
    return {
        "status": True, "message": "Task details", "data" : { "task_id": task_id, "state": result.state, "result": result.result if result.successful() else None
        }
    }