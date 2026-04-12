from __future__ import annotations

from contextlib import contextmanager
import datetime as dt
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from cbma.utils import load_structured_text, resolve_path


def _load_project_config(project_root: Path) -> dict[str, Any]:
    config_path = project_root / "project.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing file: {config_path}")

    config = load_structured_text(config_path)
    if not isinstance(config, dict):
        raise ValueError("project.yaml top-level object must be a mapping")
    return config


def _project_paths(project_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    paths = config.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("project.yaml `paths` must be a mapping")

    values = {
        "labels": str(paths.get("labels", "data/labels.csv")),
        "codebook": str(paths.get("codebook", "data/codebook.yaml")),
        "videos_dir": str(paths.get("videos_dir", "data/videos")),
        "splits_dir": str(paths.get("splits_dir", "splits/split_v1")),
        "cache_dir": str(paths.get("cache_dir", "cache")),
        "models_dir": str(paths.get("models_dir", "models")),
        "runs_dir": str(paths.get("runs_dir", "runs")),
        "exports_dir": str(paths.get("exports_dir", "exports")),
    }
    return {key: resolve_path(project_root, value) for key, value in values.items()}


def _resolve_eval_script() -> Path:
    project_release_root = Path(__file__).resolve().parents[4]
    candidate = project_release_root / "qwen2" / "src" / "qwen2" / "eval_scaling_qwen_repro.py"
    if candidate.exists():
        return candidate
    raise FileNotFoundError("Could not find the release qwen2 eval script in project_release/qwen2/src/qwen2.")


def _timestamped_run_dir(runs_dir: Path, prefix: str) -> Path:
    run_dir = runs_dir / f"{prefix}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _resolve_existing_path(raw_value: str | None, *, relative_to: Path) -> Path | None:
    if not raw_value:
        return None
    candidate = Path(raw_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (relative_to / candidate).resolve()


def _find_latest_train_sweep_run(runs_dir: Path) -> Path:
    candidates = sorted((path for path in runs_dir.glob("train-sweep-*") if path.is_dir()), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No train-sweep runs found under {runs_dir}")
    return candidates[0]


def _load_json_mapping(path: Path, description: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    payload = load_structured_text(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a mapping: {path}")
    return payload


def _resolve_source_run_dir(
    project_paths: dict[str, Path],
    run_dir: str | None,
    recommend_file: str | None,
) -> Path:
    if run_dir:
        resolved = Path(run_dir).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Train sweep run directory does not exist: {resolved}")
        return resolved

    if recommend_file:
        recommend_path = Path(recommend_file).expanduser().resolve()
        payload = _load_json_mapping(recommend_path, "recommend file")
        source_value = payload.get("source_run") or payload.get("run_dir")
        resolved = _resolve_existing_path(str(source_value), relative_to=recommend_path.parent) if source_value else recommend_path.parent
        if resolved is None or not resolved.exists():
            raise FileNotFoundError(f"Could not resolve train sweep run directory from recommend file: {recommend_path}")
        return resolved

    return _find_latest_train_sweep_run(project_paths["runs_dir"])


def _resolve_recommend_file(source_run_dir: Path, recommend_file: str | None) -> Path:
    if recommend_file:
        resolved = Path(recommend_file).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Recommend file does not exist: {resolved}")
        return resolved

    resolved = source_run_dir / "recommend_n.json"
    if not resolved.exists():
        raise FileNotFoundError(
            f"Recommend file not found: {resolved}. Provide --recommend-file or run recommend-n first."
        )
    return resolved


def _parse_recommendation(recommend_path: Path) -> tuple[int, str | None, dict[str, Any]]:
    payload = _load_json_mapping(recommend_path, "recommend file")
    raw_n = payload.get("recommended_n")
    if raw_n is None:
        raise ValueError(f"recommend file is missing `recommended_n`: {recommend_path}")

    try:
        recommended_n = int(raw_n)
    except Exception as error:
        raise ValueError(f"recommend file has non-integer `recommended_n`: {recommend_path}") from error

    metric = payload.get("metric") or payload.get("selection_metric")
    return recommended_n, str(metric) if metric is not None else None, payload


def _resolve_source_training_run(
    project_root: Path,
    source_run_dir: Path,
    recommended_n: int,
) -> tuple[dict[str, Any], Path, Path, Path]:
    source_metadata_path = source_run_dir / "train_sweep.json"
    source_metadata = _load_json_mapping(source_metadata_path, "train sweep metadata")

    artifact_root = _resolve_existing_path(str(source_metadata.get("artifact_root", "")), relative_to=source_run_dir)
    split_dir = _resolve_existing_path(str(source_metadata.get("split_dir", "")), relative_to=source_run_dir)
    model_path = _resolve_existing_path(str(source_metadata.get("model_path", "")), relative_to=source_run_dir)

    if artifact_root is None:
        raise ValueError(f"train sweep metadata is missing `artifact_root`: {source_metadata_path}")
    if split_dir is None:
        raise ValueError(f"train sweep metadata is missing `split_dir`: {source_metadata_path}")
    if model_path is None:
        raise ValueError(f"train sweep metadata is missing `model_path`: {source_metadata_path}")

    lora_path = artifact_root / "runs" / "scaling_experiments" / f"lora_5class_{recommended_n}"
    if not lora_path.exists():
        raise FileNotFoundError(
            f"No matching training result found for recommended N={recommended_n}: {lora_path}"
        )

    return source_metadata, artifact_root, split_dir, model_path


def _resolve_test_split_path(project_root: Path, config: dict[str, Any], split_dir: Path) -> tuple[Path, str]:
    defaults = config.get("defaults", {})
    baseline_defaults = defaults.get("baseline", {}) if isinstance(defaults, dict) else {}
    test_csv_name = str(baseline_defaults.get("test_csv", "test_main.csv"))
    test_split_path = split_dir / test_csv_name
    if not test_split_path.exists():
        raise FileNotFoundError(f"Test split does not exist: {test_split_path}")
    return test_split_path, test_csv_name


@contextmanager
def _temporary_env(updates: dict[str, str]):
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _temporary_sys_path(path: Path):
    path_text = str(path)
    sys.path.insert(0, path_text)
    try:
        yield
    finally:
        try:
            sys.path.remove(path_text)
        except ValueError:
            pass


def _parse_eval_report(report_path: Path) -> dict[str, float]:
    if not report_path.exists():
        raise FileNotFoundError(f"Evaluation report not found: {report_path}")

    text = report_path.read_text(encoding="utf-8")
    accuracy_match = re.search(r"Accuracy\s*:\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
    macro_f1_match = re.search(r"Macro-F1\s*:\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)

    if not accuracy_match or not macro_f1_match:
        raise ValueError(f"Could not parse accuracy or macro_f1 from report: {report_path}")

    return {
        "accuracy": float(accuracy_match.group(1)),
        "macro_f1": float(macro_f1_match.group(1)),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_eval(
    project_path: str,
    run_dir: str | None = None,
    recommend_file: str | None = None,
    output: str | None = None,
) -> int:
    project_root = Path(project_path).expanduser().resolve()
    config = _load_project_config(project_root)
    project_paths = _project_paths(project_root, config)

    source_run_dir = _resolve_source_run_dir(project_paths, run_dir=run_dir, recommend_file=recommend_file)
    resolved_recommend_file = _resolve_recommend_file(source_run_dir, recommend_file)
    recommended_n, recommend_metric, recommend_payload = _parse_recommendation(resolved_recommend_file)

    source_metadata, artifact_root, split_dir, model_path = _resolve_source_training_run(
        project_root,
        source_run_dir,
        recommended_n,
    )
    test_split_path, test_csv_name = _resolve_test_split_path(project_root, config, split_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    output_dir = Path(output).expanduser().resolve() if output else _timestamped_run_dir(project_paths["runs_dir"], "eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_eval_dir = output_dir / "raw_eval"
    frames_cache_dir = output_dir / "cache" / "frames_cache"
    raw_eval_dir.mkdir(parents=True, exist_ok=True)
    frames_cache_dir.mkdir(parents=True, exist_ok=True)

    eval_script_path = _resolve_eval_script()
    lora_runs_dir = artifact_root / "runs" / "scaling_experiments"
    lora_path = lora_runs_dir / f"lora_5class_{recommended_n}"

    env_updates = {
        "PROJECT_ROOT": str(project_root),
        "VIDEO_BASE_DIR": str(project_paths["videos_dir"]),
        "QWEN2_SPLITS_DIR": str(split_dir),
        "QWEN2_MODEL_DIR": str(model_path),
        "QWEN2_ARTIFACT_ROOT": str(artifact_root),
        "QWEN2_RUNS_DIR": str(lora_runs_dir),
        "QWEN2_FRAMES_CACHE_DIR": str(frames_cache_dir),
        "QWEN2_EVAL_OUT_DIR": str(raw_eval_dir),
    }

    with _temporary_env(env_updates), _temporary_sys_path(eval_script_path.parent):
        spec = importlib.util.spec_from_file_location("cbma_qwen2_eval_scaling", eval_script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load evaluation script: {eval_script_path}")

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as error:
            raise RuntimeError(f"Failed to import evaluation script: {error}") from error

        if not hasattr(module, "evaluate_model"):
            raise RuntimeError(f"Evaluation script does not expose evaluate_model(): {eval_script_path}")

        try:
            module.evaluate_model(recommended_n, test_csv_name=test_csv_name)
        except Exception as error:
            raise RuntimeError(f"Evaluation script failed: {error}") from error

    report_path = raw_eval_dir / f"eval_lora_5class_{recommended_n}_report.txt"
    metrics = _parse_eval_report(report_path)

    eval_result = {
        "recommended_n": recommended_n,
        "test_metric": metrics,
        "source_run": str(source_run_dir),
        "test_split": test_csv_name,
    }
    eval_metadata = {
        "project": str(project_root),
        "recommend_file": str(resolved_recommend_file),
        "recommend_metric": recommend_metric,
        "recommend_payload": recommend_payload,
        "source_run_dir": str(source_run_dir),
        "source_run_metadata": source_metadata,
        "source_lora_path": str(lora_path),
        "test_split_path": str(test_split_path),
        "model_path": str(model_path),
        "eval_script_path": str(eval_script_path),
        "output_dir": str(output_dir),
        "executed_at": dt.datetime.now().isoformat(timespec="seconds"),
    }

    _write_json(output_dir / "eval_result.json", eval_result)
    _write_json(output_dir / "eval_metadata.json", eval_metadata)

    print(f"Evaluation run completed.")
    print(f"  recommended N: {recommended_n}")
    print(f"  source run: {source_run_dir}")
    print(f"  test split: {test_split_path}")
    print(f"  accuracy: {metrics['accuracy']:.4f}")
    print(f"  macro_f1: {metrics['macro_f1']:.4f}")
    print(f"  output: {output_dir}")
    return 0
