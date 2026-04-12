from __future__ import annotations

from pathlib import Path


PROJECT_FILE_NAME = "project.yaml"
CODEBOOK_FILE_NAME = "codebook.yaml"
LABELS_FILE_NAME = "labels.csv"

PROJECT_DIRS = [
    "data",
    "data/videos",
    "cache",
    "cache/frames",
    "cache/audio",
    "cache/hf",
    "models",
    "models/qwen2",
    "models/internvl",
    "runs",
    "exports",
    "exports/adapters",
    "exports/reports",
]


def project_file(project_root: Path) -> Path:
    return project_root / PROJECT_FILE_NAME


def codebook_file(project_root: Path) -> Path:
    return project_root / "data" / CODEBOOK_FILE_NAME


def labels_file(project_root: Path) -> Path:
    return project_root / "data" / LABELS_FILE_NAME

