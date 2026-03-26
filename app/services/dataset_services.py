import zipfile
import shutil
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
print(logger.name)
BASE_DATASET_PATH = Path("app/build/datasets")
BACKUP_DATASET_PATH = Path("app")

TRAIN_PATH = BASE_DATASET_PATH / "train"
TEST_PATH = BASE_DATASET_PATH / "test"
BACKUP_PATH = BACKUP_DATASET_PATH / "backup_datasets"

def reset_database():
    if TRAIN_PATH.exists() and TEST_PATH.exists():
        shutil.rmtree(TRAIN_PATH)
        shutil.rmtree(TEST_PATH)
        return f"Database reset  successfully."
    else:
        logger.warning("Database not found")
        return f"Database not found"

def backup_dataset(backup_name:str ):
    BACKUP_DATASET_PATH = BACKUP_PATH / backup_name / "datasets"
    if BACKUP_DATASET_PATH.exists():
        return f'Dataset: {backup_name} is already exists'
    if not BASE_DATASET_PATH.exists():
        return f'Dataset not found'
    shutil.copytree(BASE_DATASET_PATH, BACKUP_DATASET_PATH)
    logger.info("Backup completed")
    return f'Backup completed.'

def restore_database(dataset_name:str):
        # Removing exsting database version
        if BASE_DATASET_PATH.exists():
            shutil.rmtree(BASE_DATASET_PATH)
        BASE_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        BACKUP_PATH.mkdir(parents = True, exist_ok=True)
        DATASET_PATH = BACKUP_PATH / dataset_name / "datasets"
        if DATASET_PATH.exists():
            shutil.copytree(DATASET_PATH, BASE_DATASET_PATH)
            logger.info("Database restored successfully")
            return f"Database: {dataset_name} restored successfully"
        else:
            logger.warning("Database not found")
            return f"Database not found."

def delete_backup_database(dataset_name: str):
    DATASET_PATH = BACKUP_PATH / f"{dataset_name}"
    if DATASET_PATH.exists():
        shutil.rmtree(DATASET_PATH)
        logger.debug(f"Backup dataset: {dataset_name} deleted successfully.")
        return f"Backup dataset: {dataset_name} deleted successfully."
    else:
        logger.warning("Database not found")
        return f"Database not found."
    
def delete_dataset(dataset_name: str):
    REMOVE_TRAIN_DATASET = TRAIN_PATH / dataset_name
    REMOVE_TEST_DATASET = TEST_PATH / dataset_name
    paths = [REMOVE_TRAIN_DATASET, REMOVE_TEST_DATASET]
    deleted = [shutil.rmtree(p) or True if p.exists() else False for p in paths]
    return f'Deleted {dataset_name} successfully.' if all(deleted) else "No dataset found"

def list_backup_datasets():
    if BACKUP_PATH.exists():
        backup_list = [Path(f).name for f in BACKUP_PATH.iterdir() if f.is_dir()]
        return backup_list
    else:
        return []
    
def list_database_datasets():
    if TRAIN_PATH.exists():
        dataset_list = [Path(f).name for  f in TRAIN_PATH.iterdir() if f.is_dir()]
        return dataset_list
    else:
        return []

def extract_and_merge(zip_path: Path):
    temp_extract_path = BASE_DATASET_PATH / "temp_extract"
    if temp_extract_path.exists():
        shutil.rmtree(temp_extract_path)
    temp_extract_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_path)

    subfolders = [f for f in temp_extract_path.iterdir() if f.is_dir()]

    if len(subfolders) != 1:
        shutil.rmtree(temp_extract_path)
        raise ValueError("ZIP must contain exactly one class folder")

    class_root = subfolders[0]
    class_name = class_root.name.lower()

    train_src = class_root / "train"
    test_src = class_root / "test"

    if not train_src.exists() or not test_src.exists():
        shutil.rmtree(temp_extract_path)
        raise ValueError("Class folder must contain train/ and test/")

    TRAIN_PATH.mkdir(parents=True, exist_ok=True)
    TEST_PATH.mkdir(parents=True, exist_ok=True)

    train_dest = TRAIN_PATH / class_name
    test_dest = TEST_PATH / class_name

    train_dest.mkdir(parents=True, exist_ok=True)
    test_dest.mkdir(parents=True, exist_ok=True)

    for img in train_src.iterdir():
        if img.is_file():
            shutil.copy2(img, train_dest / img.name)

    for img in test_src.iterdir():
        if img.is_file():
            shutil.copy2(img, test_dest / img.name)

    shutil.rmtree(temp_extract_path)

    return f"{class_name} merged successfully"