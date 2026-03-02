import zipfile
import shutil
from pathlib import Path

BASE_DATASET_PATH = Path("app/build/datasets")
TRAIN_PATH = BASE_DATASET_PATH / "train"
TEST_PATH = BASE_DATASET_PATH / "test"


def extract_and_merge(zip_path: Path):

    temp_extract_path = BASE_DATASET_PATH / "temp_extract"

    if temp_extract_path.exists():
        shutil.rmtree(temp_extract_path)

    temp_extract_path.mkdir(parents=True, exist_ok=True)

    # Extract zip
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