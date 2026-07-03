from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "supervision.db"

DEFAULT_WEIGHTS = MODELS_DIR / "yolov8s.pt"
DEFAULT_SPEED_WEIGHTS = MODELS_DIR / "yolo11s.pt"
DEFAULT_CONFIDENCE = 0.3
DEFAULT_IOU = 0.7

for directory in (UPLOAD_DIR, OUTPUT_DIR, MODELS_DIR, DATA_DIR):
    directory.mkdir(parents=True, exist_ok=True)
