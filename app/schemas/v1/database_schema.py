from pydantic  import BaseModel
from typing import Literal
from enum import Enum
from app.services.dataset_services import list_backup_datasets, list_database_datasets

datasets = list_database_datasets()
backup_datasets = list_backup_datasets()

ListBackupDatasets = Enum(
    "ListBackupDatasets",
    {name: name for  name in backup_datasets},
    type=str
)
ListDatasets = Enum(
    "ListDatasets",
    {name: name for name in datasets},
    type = str
)
class BackupName(BaseModel):
    backup_name: str
    



    

