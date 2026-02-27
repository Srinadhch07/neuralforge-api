from fastapi import APIRouter, UploadFile

router = APIRouter(prefix="/model")

@router.get("/predict")
async def predict(file: UploadFile):
    return { "status": True, "message": "Machine learning route", "data": { "Model": "In progress"}}