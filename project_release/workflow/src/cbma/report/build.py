from __future__ import annotations

import csv
import datetime as dt
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
    return {"runs_dir": resolve_path(project_root, str(paths.get("runs_dir", "runs")))}


def _resolve_eval_dir(project_paths: dict[str, Path], eval_dir: str | None) -> Path:
    if eval_dir:
        resolved = Path(eval_dir).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Evaluation directory does not exist: {resolved}")
        return resolved

    candidates = sorted((path for path in project_paths["runs_dir"].glob("eval-*") if path.is_dir()), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No eval runs found under {project_paths['runs_dir']}")
    return candidates[0]


def _load_required_json(path: Path, description: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    payload = load_structured_text(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a mapping: {path}")
    return payload


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _extract_core_metrics(eval_result: dict[str, Any], eval_metadata: dict[str, Any]) -> dict[str, Any]:
    test_metric = eval_result.get("test_metric", {})
    if not isinstance(test_metric, dict):
        test_metric = {}

    return {
        "recommended_n": _safe_int(eval_result.get("recommended_n")),
        "accuracy": _safe_float(test_metric.get("accuracy")),
        "macro_f1": _safe_float(test_metric.get("macro_f1")),
        "test_split": _string_or_none(eval_metadata.get("test_split_path")) or _string_or_none(eval_result.get("test_split")),
        "source_run": _string_or_none(eval_metadata.get("source_run_dir")) or _string_or_none(eval_result.get("source_run")),
        "model_path": _string_or_none(eval_metadata.get("model_path")),
        "project_path": _string_or_none(eval_metadata.get("project")) or _string_or_none(eval_result.get("project")),
        "recommend_file": _string_or_none(eval_metadata.get("recommend_file")),
        "eval_dir": _string_or_none(eval_metadata.get("output_dir")),
    }


def _discover_optional_files(raw_eval_dir: Path) -> dict[str, Path | None]:
    discovered = {
        "predictions": None,
        "classification_report": None,
        "confusion": None,
        "error_cases": None,
    }
    if not raw_eval_dir.exists():
        return discovered

    files = [path for path in raw_eval_dir.rglob("*") if path.is_file()]
    if not files:
        return discovered

    def rank(path: Path) -> tuple[int, int, str]:
        suffix_priority = {".json": 0, ".csv": 1, ".txt": 2}
        return (suffix_priority.get(path.suffix.lower(), 9), len(path.name), path.name.lower())

    def choose(predicate) -> Path | None:
        candidates = [path for path in files if predicate(path)]
        if not candidates:
            return None
        return sorted(candidates, key=rank)[0]

    discovered["predictions"] = choose(
        lambda path: path.suffix.lower() == ".csv" and any(token in path.name.lower() for token in ("prediction", "predictions", "pred"))
    )
    discovered["classification_report"] = choose(
        lambda path: any(token in path.name.lower() for token in ("classification", "report"))
        and path.suffix.lower() in {".json", ".csv", ".txt"}
    )
    discovered["confusion"] = choose(
        lambda path: "confusion" in path.name.lower() and path.suffix.lower() in {".json", ".csv", ".txt"}
    )
    discovered["error_cases"] = choose(
        lambda path: any(token in path.name.lower() for token in ("misclassified", "bad_cases", "error_cases", "errors"))
        and path.suffix.lower() == ".csv"
    )
    return discovered


def _resolve_label_columns(fieldnames: list[str]) -> tuple[str | None, str | None]:
    gold_candidates = ["gold_label", "gold", "true_label", "gold_code", "label"]
    pred_candidates = ["pred_label", "prediction", "pred", "pred_code"]

    gold_field = next((name for name in gold_candidates if name in fieldnames), None)
    pred_field = next((name for name in pred_candidates if name in fieldnames), None)
    return gold_field, pred_field


def _normalize_label(value: Any) -> str:
    text = str(value).strip()
    try:
        return str(int(text))
    except Exception:
        return text


def _derive_from_predictions(predictions_path: Path) -> dict[str, Any]:
    fieldnames, rows = load_csv_rows(predictions_path)
    gold_field, pred_field = _resolve_label_columns(fieldnames)
    if gold_field is None or pred_field is None:
        return {"per_class": None, "confusion": None, "errors": None}

    normalized_rows: list[dict[str, str]] = []
    labels: set[str] = set()
    for row in rows:
        gold = _normalize_label(row.get(gold_field, ""))
        pred = _normalize_label(row.get(pred_field, ""))
        normalized = dict(row)
        normalized[gold_field] = gold
        normalized[pred_field] = pred
        normalized_rows.append(normalized)
        labels.add(gold)
        labels.add(pred)

    ordered_labels = sorted(labels, key=lambda item: (not item.isdigit(), item))

    per_class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    for label in ordered_labels:
        tp = sum(1 for row in normalized_rows if row[gold_field] == label and row[pred_field] == label)
        fp = sum(1 for row in normalized_rows if row[gold_field] != label and row[pred_field] == label)
        fn = sum(1 for row in normalized_rows if row[gold_field] == label and row[pred_field] != label)
        support = sum(1 for row in normalized_rows if row[gold_field] == label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class_rows.append(
            {
                "label": label,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "support": support,
            }
        )

        confusion_row: dict[str, Any] = {"gold_label": label}
        for pred_label in ordered_labels:
            confusion_row[pred_label] = sum(
                1 for row in normalized_rows if row[gold_field] == label and row[pred_field] == pred_label
            )
        confusion_rows.append(confusion_row)

    for row in normalized_rows:
        if row[gold_field] != row[pred_field]:
            error_rows.append(row)

    return {"per_class": per_class_rows, "confusion": confusion_rows, "errors": error_rows}


def _parse_per_class_from_report(report_path: Path) -> list[dict[str, Any]] | None:
    suffix = report_path.suffix.lower()
    if suffix == ".json":
        payload = load_structured_text(report_path)
        if not isinstance(payload, dict):
            return None
        rows: list[dict[str, Any]] = []
        for label, metrics in payload.items():
            if not isinstance(metrics, dict):
                continue
            if label.lower() in {"accuracy", "macro avg", "weighted avg", "macro_avg", "weighted_avg"}:
                continue
            if "f1-score" in metrics or "f1" in metrics:
                rows.append(
                    {
                        "label": str(label),
                        "precision": _safe_float(metrics.get("precision")),
                        "recall": _safe_float(metrics.get("recall")),
                        "f1": _safe_float(metrics.get("f1-score", metrics.get("f1"))),
                        "support": _safe_int(metrics.get("support")),
                    }
                )
        return rows or None

    if suffix == ".csv":
        fieldnames, rows = load_csv_rows(report_path)
        if not rows:
            return None
        label_field = next((name for name in ("label", "class", "class_label") if name in fieldnames), None)
        f1_field = next((name for name in ("f1", "f1-score", "f1_score") if name in fieldnames), None)
        if label_field is None or f1_field is None:
            return None
        parsed: list[dict[str, Any]] = []
        for row in rows:
            parsed.append(
                {
                    "label": row.get(label_field, ""),
                    "precision": _safe_float(row.get("precision")),
                    "recall": _safe_float(row.get("recall")),
                    "f1": _safe_float(row.get(f1_field)),
                    "support": _safe_int(row.get("support")),
                }
            )
        return parsed or None

    return None


def _parse_confusion_file(confusion_path: Path) -> tuple[list[str], list[dict[str, Any]]] | None:
    suffix = confusion_path.suffix.lower()
    if suffix == ".csv":
        fieldnames, rows = load_csv_rows(confusion_path)
        return fieldnames, rows

    if suffix == ".json":
        payload = load_structured_text(confusion_path)
        if isinstance(payload, dict):
            labels = payload.get("labels")
            matrix = payload.get("matrix")
            if isinstance(labels, list) and isinstance(matrix, list):
                rows: list[dict[str, Any]] = []
                label_names = [str(label) for label in labels]
                for index, row in enumerate(matrix):
                    if not isinstance(row, list):
                        continue
                    built = {"gold_label": label_names[index] if index < len(label_names) else str(index)}
                    for label_name, value in zip(label_names, row):
                        built[label_name] = value
                    rows.append(built)
                return ["gold_label", *label_names], rows

            nested_rows: list[dict[str, Any]] = []
            nested_labels: set[str] = set()
            for gold_label, row in payload.items():
                if not isinstance(row, dict):
                    continue
                built = {"gold_label": str(gold_label)}
                for pred_label, value in row.items():
                    nested_labels.add(str(pred_label))
                    built[str(pred_label)] = value
                nested_rows.append(built)
            if nested_rows:
                ordered = sorted(nested_labels, key=lambda item: (not item.isdigit(), item))
                for row in nested_rows:
                    for label in ordered:
                        row.setdefault(label, 0)
                return ["gold_label", *ordered], nested_rows
    return None


def _read_selection_basis(recommend_file: Path | None, eval_metadata: dict[str, Any]) -> str:
    payload: dict[str, Any] | None = None
    if recommend_file and recommend_file.exists():
        loaded = load_structured_text(recommend_file)
        if isinstance(loaded, dict):
            payload = loaded
    if payload is None:
        metadata_payload = eval_metadata.get("recommend_payload")
        if isinstance(metadata_payload, dict):
            payload = metadata_payload
    if not payload:
        return "unavailable"

    selection_log = payload.get("selection_log")
    if isinstance(selection_log, str) and selection_log.strip():
        return selection_log.strip()
    if isinstance(selection_log, list) and selection_log:
        return "; ".join(str(item) for item in selection_log[:3])

    rule = payload.get("rule")
    delta = payload.get("delta")
    metric = payload.get("metric") or payload.get("selection_metric")
    parts = []
    if rule:
        parts.append(f"rule={rule}")
    if metric:
        parts.append(f"metric={metric}")
    if delta is not None:
        parts.append(f"delta={delta}")
    return ", ".join(parts) if parts else "unavailable"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "unavailable"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join([header, divider, *body])


def run_report_build(
    project_path: str,
    eval_dir: str | None = None,
    output: str | None = None,
    format_name: str = "markdown",
    include_errors: bool = False,
    include_confusion: bool = False,
) -> int:
    del include_errors
    del include_confusion

    if format_name.lower() != "markdown":
        print(f"[ERROR] Unsupported report format: {format_name}. Only `markdown` is supported.")
        return 1

    project_root = Path(project_path).expanduser().resolve()
    try:
        config = _load_project_config(project_root)
        project_paths = _project_paths(project_root, config)
        resolved_eval_dir = _resolve_eval_dir(project_paths, eval_dir)
        eval_result_path = resolved_eval_dir / "eval_result.json"
        eval_metadata_path = resolved_eval_dir / "eval_metadata.json"
        eval_result = _load_required_json(eval_result_path, "eval_result.json")
        eval_metadata = _load_required_json(eval_metadata_path, "eval_metadata.json")
    except (FileNotFoundError, ValueError) as error:
        print(f"[ERROR] {error}")
        return 1

    report_dir = Path(output).expanduser().resolve() if output else (resolved_eval_dir / "report")
    report_dir.mkdir(parents=True, exist_ok=True)

    raw_eval_dir = resolved_eval_dir / "raw_eval"
    discovered = _discover_optional_files(raw_eval_dir)
    core = _extract_core_metrics(eval_result, eval_metadata)

    recommend_file = Path(core["recommend_file"]).resolve() if core["recommend_file"] else None
    source_run_path = Path(core["source_run"]).resolve() if core["source_run"] else None
    source_run_exists = bool(source_run_path and source_run_path.exists())

    derived = {"per_class": None, "confusion": None, "errors": None}
    if discovered["predictions"] is not None:
        derived = _derive_from_predictions(discovered["predictions"])

    per_class_rows = derived["per_class"]
    if per_class_rows is None and discovered["classification_report"] is not None:
        per_class_rows = _parse_per_class_from_report(discovered["classification_report"])

    confusion_fieldnames: list[str] | None = None
    confusion_rows: list[dict[str, Any]] | None = derived["confusion"]
    if confusion_rows:
        confusion_fieldnames = list(confusion_rows[0].keys())
    elif discovered["confusion"] is not None:
        parsed_confusion = _parse_confusion_file(discovered["confusion"])
        if parsed_confusion is not None:
            confusion_fieldnames, confusion_rows = parsed_confusion

    error_rows = derived["errors"]
    if error_rows is None and discovered["error_cases"] is not None:
        _, parsed_errors = load_csv_rows(discovered["error_cases"])
        error_rows = parsed_errors

    generated_files: list[str] = []

    metrics_payload = {
        "accuracy": core["accuracy"],
        "macro_f1": core["macro_f1"],
        "recommended_n": core["recommended_n"],
        "test_split": core["test_split"],
        "source_run": core["source_run"],
        "eval_dir": str(resolved_eval_dir),
    }
    _write_json(report_dir / "metrics.json", metrics_payload)
    generated_files.append(str(report_dir / "metrics.json"))

    if per_class_rows:
        _write_csv(report_dir / "per_class_f1.csv", ["label", "precision", "recall", "f1", "support"], per_class_rows)
        generated_files.append(str(report_dir / "per_class_f1.csv"))

    if confusion_rows and confusion_fieldnames:
        _write_csv(report_dir / "confusion_matrix.csv", confusion_fieldnames, confusion_rows)
        generated_files.append(str(report_dir / "confusion_matrix.csv"))

    if error_rows:
        error_fieldnames = list(error_rows[0].keys()) if error_rows else []
        if error_fieldnames:
            _write_csv(report_dir / "error_cases.csv", error_fieldnames, error_rows)
            generated_files.append(str(report_dir / "error_cases.csv"))

    build_timestamp = dt.datetime.now().isoformat(timespec="seconds")
    run_summary = {
        "project_path": str(project_root),
        "eval_dir": str(resolved_eval_dir),
        "report_dir": str(report_dir),
        "recommend_file": str(recommend_file) if recommend_file else None,
        "recommended_n": core["recommended_n"],
        "source_run": core["source_run"],
        "model_path": core["model_path"],
        "test_split": core["test_split"],
        "generated_files": generated_files,
        "build_timestamp": build_timestamp,
    }
    _write_json(report_dir / "run_summary.json", run_summary)
    generated_files.append(str(report_dir / "run_summary.json"))

    raw_eval_label = str(raw_eval_dir) if raw_eval_dir.exists() else "not provided"
    selection_basis = _read_selection_basis(recommend_file, eval_metadata)
    per_class_section = _markdown_table(per_class_rows, ["label", "precision", "recall", "f1", "support"]) if per_class_rows else "unavailable"
    confusion_section = str(report_dir / "confusion_matrix.csv") if confusion_rows and confusion_fieldnames else "unavailable"
    error_section = f"{report_dir / 'error_cases.csv'} ({len(error_rows)} rows)" if error_rows and (report_dir / "error_cases.csv").exists() else "unavailable"

    report_text = "\n".join(
        [
            "# CBMA Evaluation Report",
            "",
            "## 1. Overview",
            f"- project path: {project_root}",
            f"- eval run path: {resolved_eval_dir}",
            f"- source train-sweep run: {core['source_run'] or 'unavailable'}",
            f"- recommended N: {core['recommended_n'] if core['recommended_n'] is not None else 'unavailable'}",
            f"- test split: {core['test_split'] or 'unavailable'}",
            f"- model path: {core['model_path'] or 'unavailable'}",
            f"- build timestamp: {build_timestamp}",
            "",
            "## 2. Headline Metrics",
            f"- accuracy: {core['accuracy'] if core['accuracy'] is not None else 'unavailable'}",
            f"- macro_f1: {core['macro_f1'] if core['macro_f1'] is not None else 'unavailable'}",
            "",
            "## 3. Recommendation Provenance",
            f"- recommend file path: {recommend_file if recommend_file else 'unavailable'}",
            f"- selection basis: {selection_basis}",
            f"- source run path: {core['source_run'] or 'unavailable'}",
            "",
            "## 4. Evaluation Inputs",
            f"- test split path: {core['test_split'] or 'unavailable'}",
            f"- raw eval directory: {raw_eval_label}",
            f"- eval metadata summary: source_run_exists={source_run_exists}, eval_script_path={eval_metadata.get('eval_script_path', 'unavailable')}",
            "",
            "## 5. Per-class Results",
            per_class_section,
            "",
            "## 6. Confusion Structure",
            f"- confusion matrix: {confusion_section}",
            "",
            "## 7. Error Cases",
            f"- error cases: {error_section}",
            "",
            "## 8. Notes and Limitations",
            "- This report is built from existing eval outputs.",
            "- The model was not re-run during report construction.",
            "- Missing sections depend on which raw eval artifacts were available at build time.",
        ]
    ) + "\n"

    (report_dir / "report.md").write_text(report_text, encoding="utf-8")
    generated_files.append(str(report_dir / "report.md"))

    run_summary["generated_files"] = generated_files
    _write_json(report_dir / "run_summary.json", run_summary)

    print("Report build completed.")
    print(f"  eval dir: {resolved_eval_dir}")
    print(f"  report dir: {report_dir}")
    print(f"  headline metrics: accuracy={core['accuracy'] if core['accuracy'] is not None else 'unavailable'}, macro_f1={core['macro_f1'] if core['macro_f1'] is not None else 'unavailable'}")
    print(f"  raw eval: {raw_eval_label}")
    return 0
