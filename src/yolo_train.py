from ultralytics import YOLO
from ultralytics.utils import SETTINGS
from pathlib import Path

# Disable MLflow logging to avoid errors with invalid registry URI
SETTINGS["mlflow"] = False

# Project root
BASE_DIR = Path(__file__).resolve().parents[1]

# Use dataset's own data.yaml (Roboflow export)
DATA_YAML = BASE_DIR / "dataset" / "object_detection_Dataset" / "data.yaml"

# Where to store YOLO runs / weights
DETECT_MODELS_DIR = BASE_DIR / "models" / "detection"

if __name__ == "__main__":
    print("Using data.yaml:", DATA_YAML)

    # Use the base weights committed at the repo root; Ultralytics downloads them if missing.
    base_weights = BASE_DIR / "yolov8n.pt"
    model = YOLO(str(base_weights) if base_weights.exists() else "yolov8n.pt")   # small model, good for CPU
    run_name = "yolov8n_birddrone"

    model.train(
        data=str(DATA_YAML),
        epochs=15,          # reduce to 10 if too slow
        imgsz=640,
        batch=8,
        device="cpu",
        project=str(DETECT_MODELS_DIR),
        name=run_name,
        exist_ok=True
    )

    print("YOLOv8 training complete.")
    print("Best weights should be in:", DETECT_MODELS_DIR / run_name / "weights" / "best.pt")
