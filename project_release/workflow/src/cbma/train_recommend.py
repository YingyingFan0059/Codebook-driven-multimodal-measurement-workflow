from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cbma.utils import load_csv_rows, load_structured_text, resolve_path


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
        "runs_dir": str(paths.get("runs_dir", "runs")),
    }
    return {key: resolve_path(project_root, value) for key, value in values.items()}


def _resolve_run_dir(project_paths: dict[str, Path], run_dir: str | None) -> Path:
    if run_dir:
        resolved = Path(run_dir).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Train sweep run directory does not exist: {resolved}")
        return resolved

    candidates = sorted((path for path in project_paths["runs_dir"].glob("train-sweep-*") if path.is_dir()), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No train-sweep runs found under {project_paths['runs_dir']}")
    return candidates[0]


def _load_mapping(path: Path, description: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    payload = load_structured_text(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a mapping: {path}")
    return payload


def _metric_candidates(configured_metric: str | None) -> list[str]:
    candidates: list[str] = []
    if configured_metric:
        candidates.append(configured_metric)
        if configured_metric.startswith("val_"):
            candidates.append(configured_metric[4:])
        else:
            candidates.append(f"val_{configured_metric}")

    candidates.extend(["val_macro_f1", "macro_f1", "macro-f1", "score", "metric_value"])

    ordered: list[str] = []
    for candidate in candidates:
        normalized = candidate.strip()
        if normalized and normalized not in ordered:
            ordered.append(normalized)
    return ordered


def _extract_n(row: dict[str, Any]) -> int | None:
    for key in ("N", "n", "train_size", "size"):
        if key not in row:
            continue
        try:
            value = int(str(row.get(key, "")).strip())
        except Exception:
            continue
        if value > 0:
            return value
    return None


def _extract_score(row: dict[str, Any], metric_candidates: list[str]) -> tuple[float | None, str | None]:
    for key in metric_candidates:
        if key not in row:
            continue
        try:
            return float(str(row.get(key, "")).strip()), key
        except Exception:
            continue
    return None, None


def _records_from_csv(path: Path, metric_candidates: list[str]) -> tuple[list[dict[str, Any]], str | None]:
    _, rows = load_csv_rows(path)
    records: list[dict[str, Any]] = []
    metric_name: str | None = None
    for row in rows:
        train_size = _extract_n(row)
        score, used_metric = _extract_score(row, metric_candidates)
        if train_size is None or score is None:
            continue
        records.append({"N": train_size, "score": score})
        if metric_name is None and used_metric is not None:
            metric_name = used_metric
    return records, metric_name


def _records_from_json(path: Path, metric_candidates: list[str]) -> tuple[list[dict[str, Any]], str | None]:
    payload = load_structured_text(path)

    candidate_rows: list[dict[str, Any]] = []
    if isinstance(payload, list):
        candidate_rows = [item for item in payload if isinstance(item, dict)]
    elif isinstance(payload, dict):
        for key in ("results", "metrics", "rows", "records", "scaling_curve"):
            value = payload.get(key)
            if isinstance(value, list):
                candidate_rows = [item for item in value if isinstance(item, dict)]
                break
        if not candidate_rows:
            for key, value in payload.items():
                if isinstance(value, dict):
                    merged = dict(value)
                    merged.setdefault("N", key)
                    candidate_rows.append(merged)

    records: list[dict[str, Any]] = []
    metric_name: str | None = None
    for row in candidate_rows:
        train_size = _extract_n(row)
        score, used_metric = _extract_score(row, metric_candidates)
        if train_size is None or score is None:
            continue
        records.append({"N": train_size, "score": score})
        if metric_name is None and used_metric is not None:
            metric_name = used_metric
    return records, metric_name


def _discover_scaling_records(
    run_dir: Path,
    configured_metric: str | None,
) -> tuple[list[dict[str, Any]], str, Path]:
    metric_candidates = _metric_candidates(configured_metric)

    candidate_paths = [
        run_dir / "scaling_curve.csv",
        run_dir / "results.csv",
        run_dir / "metrics.csv",
        run_dir / "metrics.json",
        run_dir / "results.json",
        run_dir / "scaling_results.json",
    ]

    for path in candidate_paths:
        if not path.exists():
            continue
        if path.suffix.lower() == ".csv":
            records, metric_name = _records_from_csv(path, metric_candidates)
        else:
            records, metric_name = _records_from_json(path, metric_candidates)
        if records:
            return sorted(records, key=lambda item: item["N"]), (metric_name or "val_macro_f1"), path

    expected = ", ".join(str(path.name) for path in candidate_paths)
    raise FileNotFoundError(
        f"Could not find usable sweep metrics under {run_dir}. Expected one of: {expected}"
    )


def _resolve_output_path(run_dir: Path, output: str | None) -> Path:
    if output is None:
        return run_dir / "recommend_n.json"

    resolved = Path(output).expanduser()
    if resolved.exists() and resolved.is_dir():
        return (resolved / "recommend_n.json").resolve()
    if resolved.suffix.lower() != ".json" and output.endswith(("/", "\\")):
        return (resolved / "recommend_n.json").resolve()
    return resolved.resolve()


def _safe_float(value: float) -> float:
    return round(float(value), 6)


def _build_recommendation_payload(
    run_dir: Path,
    metric_name: str,
    records: list[dict[str, Any]],
    delta: float,
) -> dict[str, Any]:
    if not records:
        raise ValueError("No valid metric records were found for recommendation.")

    deduped: dict[int, float] = {}
    for record in records:
        deduped[int(record["N"])] = float(record["score"])

    ordered = sorted(({"N": size, "score": score} for size, score in deduped.items()), key=lambda item: item["N"])
    best_score = max(item["score"] for item in ordered)
    best_candidates = [item["N"] for item in ordered if item["score"] == best_score]
    best_n = min(best_candidates)
    threshold = best_score - delta
    eligible_n = [item["N"] for item in ordered if item["score"] >= threshold]
    if not eligible_n:
        raise ValueError("Could not compute a recommendation from the supplied metric records.")

    recommended_n = min(eligible_n)
    selection_log = [
        f"metric={metric_name}, delta={delta:.4f}",
        f"best_n={best_n}, best_score={best_score:.4f}, threshold={threshold:.4f}",
        f"eligible_n={eligible_n}",
        f"selected_n={recommended_n} because it is the smallest N within delta of the best score",
    ]

    return {
        "recommended_n": recommended_n,
        "best_n": best_n,
        "best_score": _safe_float(best_score),
        "best_val_macro_f1": _safe_float(best_score) if metric_name == "val_macro_f1" else None,
        "metric": metric_name,
        "selection_metric": metric_name,
        "delta": _safe_float(delta),
        "rule": f"min N where {metric_name} >= best - delta",
        "source_run": str(run_dir),
        "run_dir": str(run_dir),
        "summary": {
            "candidate_count": len(ordered),
            "threshold": _safe_float(threshold),
            "eligible_n": eligible_n,
            "scores": [{"N": item["N"], metric_name: _safe_float(item["score"])} for item in ordered],
        },
        "selection_log": selection_log,
    }


def run_recommend_n(
    project_path: str,
    run_dir: str | None = None,
    output: str | None = None,
) -> int:
    project_root = Path(project_path).expanduser().resolve()
    config = _load_project_config(project_root)
    project_paths = _project_paths(project_root, config)
    resolved_run_dir = _resolve_run_dir(project_paths, run_dir)

    train_sweep_path = resolved_run_dir / "train_sweep.json"
    _load_mapping(train_sweep_path, "train_sweep.json")

    defaults = config.get("defaults", {})
    training_defaults = defaults.get("training", {}) if isinstance(defaults, dict) else {}
    configured_metric = str(training_defaults.get("selection_metric", "")).strip() or None
    selection_rule = training_defaults.get("selection_rule", {}) if isinstance(training_defaults, dict) else {}
    delta = float(selection_rule.get("delta", 0.01)) if isinstance(selection_rule, dict) else 0.01

    records, metric_name, metrics_source = _discover_scaling_records(resolved_run_dir, configured_metric)
    payload = _build_recommendation_payload(resolved_run_dir, metric_name, records, delta)
    payload["summary"]["metrics_source"] = str(metrics_source)

    output_path = _resolve_output_path(resolved_run_dir, output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("Recommend-n completed.")
    print(f"  run dir: {resolved_run_dir}")
    print(f"  metrics source: {metrics_source}")
    print(f"  recommended N: {payload['recommended_n']}")
    print(f"  reason: {payload['selection_log'][-1]}")
    print(f"  output: {output_path}")
    return 0
