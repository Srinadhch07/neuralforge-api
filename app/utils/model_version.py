from pathlib import Path
import logging

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
SMALL_MODEL_DIR = BASE_DIR / "models" / "v1"
MEDIUM_MODEL_DIR = BASE_DIR / "models" / "v2" 
LARGE_MODEL_DIR = BASE_DIR / "models" / "v3"

SMALL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
MEDIUM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
LARGE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

def new_model_name():
    model_versions = {}
    if not SMALL_MODEL_DIR.exists():
        model_versions["small"] = "leaf_detection_model_small_v1.pth"
    if not MEDIUM_MODEL_DIR.exists():
        model_versions["medium"] = "leaf_detection_model_medium_v1.pth"
    if not LARGE_MODEL_DIR.exists():
         model_versions["large"] = "leaf_detection_model_large_v1.pth"

    small_model_version = len([f for f in SMALL_MODEL_DIR.iterdir() if f.is_file() ]) + 1
    medium_model_version = len([f for f in MEDIUM_MODEL_DIR.iterdir() if f.is_file() ]) + 1
    large_model_version = len([f for f in LARGE_MODEL_DIR.iterdir() if f.is_file() ]) + 1

    model_versions["small"] = f"leaf_detection_model_small_v{small_model_version}.pth"
    model_versions["medium"] = f"leaf_detection_model_medium_v{medium_model_version}.pth" 
    model_versions["large"] = f"leaf_detection_model_large_v{large_model_version}.pth" 
    return model_versions

def current_model_name():
    model_versions = {}
    if not SMALL_MODEL_DIR.exists():
        model_versions["small"] = "leaf_detection_model_small_v1.pth"
    if not MEDIUM_MODEL_DIR.exists():
        model_versions["medium"] = "leaf_detection_model_medium_v1.pth"
    if not LARGE_MODEL_DIR.exists():
         model_versions["large"] = "leaf_detection_model_large_v1.pth"

    small_model_version = len([f for f in SMALL_MODEL_DIR.iterdir() if f.is_file() ])
    medium_model_version = len([f for f in MEDIUM_MODEL_DIR.iterdir() if f.is_file() ])
    large_model_version = len([f for f in LARGE_MODEL_DIR.iterdir() if f.is_file() ])

    small_model_version = small_model_version + 1 if small_model_version == 0 else small_model_version
    medium_model_version = medium_model_version + 1 if medium_model_version == 0 else medium_model_version
    large_model_version = large_model_version + 1 if large_model_version == 0 else large_model_version
    

    model_versions["small"] = f"leaf_detection_model_small_v{small_model_version}.pth"
    model_versions["medium"] = f"leaf_detection_model_medium_v{medium_model_version}.pth" 
    model_versions["large"] = f"leaf_detection_model_large_v{large_model_version}.pth" 

    logger.debug(f"Current model versions: {model_versions}")
    return model_versions

def is_model_exists():
    MODEL_EXISTS = {
        "small": None,
        "medium": None,
        "large": None
    }
    MODEL_EXISTS["small"] = True  if len([f for f in SMALL_MODEL_DIR.iterdir() if f.is_file()]) else False
    MODEL_EXISTS["medium"] = True  if len([f for f in SMALL_MODEL_DIR.iterdir() if f.is_file()]) else False
    MODEL_EXISTS["large"] = True  if len([f for f in SMALL_MODEL_DIR.iterdir() if f.is_file()]) else False
    return MODEL_EXISTS

def list_models(model_type:str = None):
    model_list = {}
    model_list["small"] = [Path(p).name for p in SMALL_MODEL_DIR.iterdir() if p.is_file()]
    model_list["medium"] = [Path(p).name for p in MEDIUM_MODEL_DIR.iterdir() if p.is_file()]
    model_list["large"] = [Path(p).name for p in LARGE_MODEL_DIR.iterdir() if p.is_file()]
    if not model_type:
        return model_list
    return model_list.get(model_type)
