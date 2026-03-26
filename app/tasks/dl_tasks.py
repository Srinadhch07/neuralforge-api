from celery import states
import logging
from pathlib import Path

from app.config.celery_app import celery
from app.build.run_model import leaf_detection_model
from app.build.train.v1.train import train_model as small_model
from app.build.train.v2.train import train_model as large_model
from app.build.train.v3.train import train_model as medium_model
from app.services.dataset_services import extract_and_merge
from data.disease_data import DISEASE_DATABASE
from app.services.s3_service import load_image_from_s3


logger = logging.getLogger(__name__)

@celery.task(bind=True, autoretry_for = (Exception,), retry_backoff=10, retry_kwargs= {"max_retries":3}, name="app.tasks.dl_tasks.predict_leaf_task")
def predict_leaf_task(self,payload: dict):
    self.update_state(state = states.STARTED)

    path = payload.get("path", None)
    model = payload.get("model", None)

    image = None
    if path.startswith("http"):
        image = load_image_from_s3(path)
    else:
        image = path
    logger.debug("Analyser started")
    prediction, confidence = leaf_detection_model(image, model)
    logger.info(prediction)
    prediction = prediction.strip().split("__", 1)
    logger.info(prediction)
    crop_name = None
    disease_name = None
    if len(prediction)>=2:
        crop_name = prediction[0].lower()
        disease_name  = prediction[1].lower()
    elif len(prediction) == 1:
        crop_name,disease_name = prediction[0].lower(), prediction[0].lower() 
    else:
        crop_name,disease_name = None, None
    logger.debug(f"Inference: Crop Name - {crop_name}, Disease - Name{disease_name}")
    result = DISEASE_DATABASE.get(crop_name).get(disease_name)

    if result is None:
        result = {
            "diseases": {
                "name": disease_name,
                "probability": confidence,
                "similar_images": [],
                "disease_details": {
                    "local_name": disease_name,
                    "description": None,
                    "url":  f"https://en.wikipedia.org/wiki/{disease_name}",
                    "treatment": {
                        "chemical": [],
                        "biological": [],
                        "prevention": [],
                        "common_names": [disease_name],
                        "cause": None,
                        "language": None 
                    } 
                }
            }
        }
        
    result["diseases"]["probability"] = confidence
    logger.debug("Analyser stopped")
    return result

@celery.task(bind=True, autoretry_for = (Exception,), retry_backoff=10, retry_kwargs= {"max_retries":3}, name="app.tasks.dl_tasks.upload_dataset_task")
def upload_dataset_task(self, zip_path: str):
    self.update_state(state = states.STARTED)

    logger.info(f"Dataset is being uploaded.{zip_path}")
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"{zip_path} not found")
    try:
        message = extract_and_merge(zip_path)
        logger.debug(f"Dataset uploaded successfully")
        return {"message": message}
    finally:
        if zip_path.exists():
            zip_path.unlink()

@celery.task(bind=True, autoretry_for = (Exception,), retry_backoff=10, retry_kwargs= {"max_retries":3}, name="app.tasks.dl_tasks.train_model_task")
def train_model_task(self, payload: dict):
    self.update_state(state = states.STARTED)
    num_epochs = payload.get("epochs",1)
    model = payload.get("model","small")
    model_name = payload.get("model_name", None)
    if num_epochs <= 1:
        logger.warning(f"Training model with low epochs, Model may not accurate.")
    logger.debug("DL model training is started.")
    message = small_model(num_epochs,model, model_name) if model == "small" else  medium_model(num_epochs,model, model_name) if model == "medium" else large_model(num_epochs,model, model_name)
    result = { "message" : message }
    logger.debug("DL model traning completed")
    return result