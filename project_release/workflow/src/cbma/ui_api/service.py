from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path

from cbma.commands import (
    _available_split_sizes,
    _doctor_report,
    _load_project_config,
    _project_paths,
    _validate_project,
    run_baseline_run,
    run_doctor,
    run_split_create,
    run_train_sweep,
    run_validate,
)
from cbma.utils import load_csv_rows, load_structured_text


NAV_ITEMS = [
    {"key": "dashboard", "label": "Dashboard", "href": "/"},
    {"key": "settings", "label": "Settings", "href": "/settings"},
    {"key": "data", "label": "Data", "href": "/data"},
    {"key": "split", "label": "Split", "href": "/split"},
    {"key": "baseline", "label": "Baseline", "href": "/baseline"},
    {"key": "train", "label": "Train", "href": "/train"},
    {"key": "runs", "label": "Runs", "href": "/runs"},
]

ACTION_REGISTRY = {
    "doctor": {
        "label": "Run Doctor",
        "runner": lambda project_root: run_doctor(project_path=str(project_root)),
    },
    "validate": {
        "label": "Run Validate",
        "runner": lambda project_root: run_validate(project_path=str(project_root)),
    },
    "split-create": {
        "label": "Create Split",
        "runner": lambda project_root: run_split_create(project_path=str(project_root)),
    },
    "baseline-dryrun": {
        "label": "Baseline Dry-Run",
        "runner": lambda project_root: run_baseline_run(project_path=str(project_root), dry_run=True),
    },
    "train-sweep-dryrun": {
        "label": "Train Sweep Dry-Run",
        "runner": lambda project_root: run_train_sweep(project_path=str(project_root), dry_run=True),
    },
}


def _status_tone(ok: bool, warnings: int = 0) -> str:
    if ok and warnings == 0:
        return "ready"
    if ok and warnings > 0:
        return "caution"
    return "blocked"


def _discover_runs(runs_dir: Path) -> list[dict]:
    if not runs_dir.exists():
        return []

    discovered: list[dict] = []
    for metadata_path in sorted(runs_dir.glob("*/baseline_run.json"), reverse=True):
        payload = load_structured_text(metadata_path)
        if isinstance(payload, dict):
            payload["run_type"] = "baseline"
            payload["metadata_path"] = str(metadata_path)
            payload["run_name"] = metadata_path.parent.name
            discovered.append(payload)
    for metadata_path in sorted(runs_dir.glob("*/train_sweep.json"), reverse=True):
        payload = load_structured_text(metadata_path)
        if isinstance(payload, dict):
            payload["run_type"] = "train-sweep"
            payload["metadata_path"] = str(metadata_path)
            payload["run_name"] = metadata_path.parent.name
            discovered.append(payload)
    discovered.sort(key=lambda item: item["run_name"], reverse=True)
    return discovered


def _labels_summary(labels_path: Path) -> dict:
    if not labels_path.exists():
        return {"count": 0, "labels": [], "missing": True}

    fieldnames, rows = load_csv_rows(labels_path)
    label_set = sorted(
        {
            int(str(row.get("label", "")).strip())
            for row in rows
            if str(row.get("label", "")).strip().isdigit()
        }
    )
    return {
        "count": len(rows),
        "labels": label_set,
        "has_split_column": "split" in fieldnames,
        "missing": False,
    }


def _codebook_summary(codebook_path: Path) -> dict:
    if not codebook_path.exists():
        return {"task_name": None, "label_count": 0, "missing": True}

    payload = load_structured_text(codebook_path)
    labels = payload.get("labels", []) if isinstance(payload, dict) else []
    return {
        "task_name": payload.get("task_name") if isinstance(payload, dict) else None,
        "label_count": len(labels) if isinstance(labels, list) else 0,
        "missing": False,
    }


def collect_project_snapshot(project_root: Path) -> dict:
    snapshot: dict = {
        "project_root": str(project_root),
        "project_exists": project_root.exists(),
        "project_name": project_root.name,
        "doctor_messages": [],
        "validate_messages": [],
        "paths": {},
        "labels": {},
        "codebook": {},
        "split": {"exists": False, "sizes": [], "summary": None},
        "runs": [],
        "status_cards": [],
        "actions": [
            {"name": "doctor", "label": "Run Doctor"},
            {"name": "validate", "label": "Run Validate"},
            {"name": "split-create", "label": "Create Split"},
            {"name": "baseline-dryrun", "label": "Baseline Dry-Run"},
            {"name": "train-sweep-dryrun", "label": "Train Sweep Dry-Run"},
        ],
    }

    if not project_root.exists():
        snapshot["status_cards"] = [
            {
                "title": "Project",
                "value": "Missing",
                "detail": "The selected project path does not exist yet.",
                "tone": "blocked",
            }
        ]
        return snapshot

    doctor_messages, doctor_exit = _doctor_report(str(project_root))
    snapshot["doctor_messages"] = [message.to_dict() for message in doctor_messages]

    validate_messages, validate_exit = _validate_project(project_root)
    snapshot["validate_messages"] = [message.to_dict() for message in validate_messages]

    config = {}
    project_paths = {}
    try:
        config = _load_project_config(project_root)
        project_paths = _project_paths(project_root, config)
    except Exception:
        config = {}
        project_paths = {}

    if config:
        snapshot["project_name"] = str(config.get("project_name") or snapshot["project_name"])
    if project_paths:
        snapshot["paths"] = {key: str(value) for key, value in project_paths.items()}
        snapshot["labels"] = _labels_summary(project_paths["labels"])
        snapshot["codebook"] = _codebook_summary(project_paths["codebook"])

        split_dir = project_paths["splits_dir"]
        split_summary_path = split_dir / "split_summary.json"
        split_summary = load_structured_text(split_summary_path) if split_summary_path.exists() else None
        snapshot["split"] = {
            "exists": split_dir.exists(),
            "dir": str(split_dir),
            "sizes": _available_split_sizes(split_dir),
            "summary": split_summary if isinstance(split_summary, dict) else None,
        }
        snapshot["runs"] = _discover_runs(project_paths["runs_dir"])

    doctor_warnings = sum(1 for message in doctor_messages if message.severity == "WARN")
    validate_warnings = sum(1 for message in validate_messages if message.severity == "WARN")
    split_ready = bool(snapshot["split"].get("sizes"))
    recent_runs = snapshot["runs"][:5]

    snapshot["status_cards"] = [
        {
            "title": "Environment",
            "value": "Ready" if doctor_exit == 0 else "Blocked",
            "detail": f"{len(doctor_messages)} checks, {doctor_warnings} warning(s).",
            "tone": _status_tone(doctor_exit == 0, doctor_warnings),
        },
        {
            "title": "Inputs",
            "value": "Valid" if validate_exit == 0 else "Needs fixes",
            "detail": f"{len(validate_messages)} validation message(s), {validate_warnings} warning(s).",
            "tone": _status_tone(validate_exit == 0, validate_warnings),
        },
        {
            "title": "Split",
            "value": "Available" if split_ready else "Not built",
            "detail": f"Train sizes: {snapshot['split'].get('sizes') or 'none'}",
            "tone": "ready" if split_ready else "blocked",
        },
        {
            "title": "Runs",
            "value": str(len(recent_runs)),
            "detail": "Recent baseline and train-sweep records.",
            "tone": "ready" if recent_runs else "caution",
        },
    ]

    snapshot["defaults"] = config.get("defaults", {}) if isinstance(config, dict) else {}
    return snapshot


def run_ui_action(project_root: Path, action_name: str) -> dict:
    action = ACTION_REGISTRY.get(action_name)
    if not action:
        return {
            "name": action_name,
            "label": action_name,
            "status": "error",
            "exit_code": 2,
            "output": "Unknown action.",
        }

    output_buffer = io.StringIO()
    with redirect_stdout(output_buffer):
        exit_code = action["runner"](project_root)
    status = "ok" if exit_code == 0 else "error"
    return {
        "name": action_name,
        "label": action["label"],
        "status": status,
        "exit_code": exit_code,
        "output": output_buffer.getvalue().strip(),
    }


def build_page_context(project_root: Path, current_page: str, page_title: str, action_result: dict | None = None) -> dict:
    snapshot = collect_project_snapshot(project_root)
    return {
        "project_root": str(project_root),
        "page_title": page_title,
        "current_page": current_page,
        "nav_items": NAV_ITEMS,
        "snapshot": snapshot,
        "action_result": action_result,
    }
