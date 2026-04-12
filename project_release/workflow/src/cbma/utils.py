from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class CheckMessage:
    severity: str
    subject: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def load_structured_text(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError as json_error:
        try:
            import yaml  # type: ignore
        except ImportError as import_error:
            raise ValueError(
                f"{path} is not valid JSON-compatible YAML. Install PyYAML for full YAML support."
            ) from import_error
        try:
            return yaml.safe_load(text)
        except Exception as yaml_error:  # pragma: no cover - depends on optional dependency
            raise ValueError(f"Failed to parse structured file: {path}") from yaml_error
        raise json_error


def load_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text_file(path: Path, content: str, force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"File already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def resolve_path(project_root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (project_root / candidate).resolve()


def is_writable_directory(path: Path) -> bool:
    path.mkdir(parents=True, exist_ok=True)
    return os.access(path, os.W_OK)


def format_messages(messages: list[CheckMessage]) -> str:
    lines = []
    for message in messages:
        lines.append(f"[{message.severity}] {message.subject}: {message.detail}")
    return "\n".join(lines)
