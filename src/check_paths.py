"""Sanity-check the YOLO dataset layout.

Paths are resolved relative to this file, so it works from any working
directory and on any OS:  python src/check_paths.py
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
base = BASE_DIR / "dataset" / "object_detection_Dataset"

if __name__ == "__main__":
    print("Dataset root:", base)
    if not base.exists():
        raise SystemExit(f"Dataset folder not found: {base}")

    print("Contents of object_detection_Dataset:")
    for p in sorted(base.iterdir()):
        print(" -", p.name)

    for sub in ["train", "val", "valid", "test", "images"]:
        p = base / sub
        print(f"Exists {sub}? {p.exists()}")
