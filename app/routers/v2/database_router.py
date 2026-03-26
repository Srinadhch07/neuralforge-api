from fastapi import APIRouter, HTTPException
import logging

from app.services.dataset_services import list_backup_datasets,list_database_datasets, delete_backup_database, delete_dataset, reset_database, backup_dataset, restore_database
from app.schemas.v1.database_schema import ListBackupDatasets, ListDatasets, BackupName
logger = logging.getLogger(__name__)

router = APIRouter(prefix = "/v2/database")

@router.get("/list")
def list_all_datasets():
    dataset_list = list_database_datasets()
    return { "status": True, "message": "List of datasets", "data": { "list": dataset_list if len(dataset_list) !=0 else "No datasets found"}}

@router.get("/backup/list")
def list_all_backups():
    backup_list =  list_backup_datasets()
    return { "status": True, "message": "List of datasets", "data": { "list": backup_list if len(backup_list) !=0 else "Nothing backed up yet." }}

@router.post("/backup")
def backup_current_dataset(payload: BackupName):
    message = backup_dataset(payload.backup_name)
    return { "status": True, "message": message, "data": None}

@router.get("/reset")
def reset_current_database():
    message = reset_database()
    return { "status": True, "message": message, "data": None}


@router.post("/restore/{dataset_name}")
def restore_dataset(dataset_name: ListBackupDatasets):
    logger.debug("Restore method activated")
    message = restore_database(dataset_name.value)
    return { "status": True, "message": message, "data": None}


@router.delete("/backup/{dataset_name}/delete")
def delete_backup_dataset(dataset_name: ListBackupDatasets):
    message = delete_backup_database(dataset_name.value)
    return { "status": True, "message": message, "data": None}

@router.delete("/{dataset_name}/delete")
def delete_current_dataset(dataset_name: ListDatasets):
    message = delete_dataset(dataset_name.value)
    return { "status": True, "message": message, "data": None}
    
