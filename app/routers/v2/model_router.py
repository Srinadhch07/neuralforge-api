from fastapi import APIRouter, HTTPException

from app.utils.model_version import list_models
from app.schemas.v1.schemas import ModelType

router = APIRouter(prefix="/v2/models")

@router.post("/versions-list")
def list_model(model: ModelType):
    return {"status": True, "message": "List of version hirarchy.", "data": list_models(model.name)}
