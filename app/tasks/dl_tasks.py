from celery import states
import logging
from pathlib import Path

from app.config.celery_app import celery
from app.build.run_model import leaf_detection_model
from app.build.train.train import train_model
from app.services.dataset_services import extract_and_merge


logger = logging.getLogger(__name__)

@celery.task(bind=True, autoretry_for = (Exception,), retry_backoff=10, retry_kwargs= {"max_retries":3}, name="app.tasks.dl_tasks.predict_leaf_task")
def predict_leaf_task(self,path:str):
    self.update_state(state =states.STARTED)
    logger.debug("Analyser started")
    prediction, confidence = leaf_detection_model(path)
    result = { "prediction": prediction, "confidence": confidence }
    logger.debug("Analyser stopped")
    return result


@celery.task(bind=True, autoretry_for = (Exception,), retry_backoff=10, retry_kwargs= {"max_retries":3}, name="app.tasks.dl_tasks.upload_dataset_task")
def upload_dataset_task(self, zip_path: str):
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
def train_model_task(self, num_epochs: int):
    self.update_state(state = states.STARTED)
    logger.debug("DL model training is started.")
    message = train_model(num_epochs)
    result = { "message" : message }
    logger.debug("DL model traning completed")
    return result