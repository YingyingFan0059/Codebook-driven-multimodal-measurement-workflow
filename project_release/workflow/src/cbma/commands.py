from __future__ import annotations

from contextlib import contextmanager
import datetime as dt
import importlib.util
import json
import os
import platform
import random
import shutil
import sys
from pathlib import Path

from cbma.layout import (
    PROJECT_DIRS,
    codebook_file,
    labels_file,
    project_file,
)
from cbma.eval.run import run_eval
from cbma.report.build import run_report_build
from cbma.train_recommend import run_recommend_n
from cbma.templates import (
    codebook_template,
    gitignore_template,
    labels_csv_template,
    project_config_template,
    project_readme_template,
    to_json_compatible_yaml,
)
from cbma.utils import (
    CheckMessage,
    format_messages,
    is_writable_directory,
    load_csv_rows,
    load_structured_text,
    resolve_path,
    write_csv_rows,
    write_text_file,
)


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


def run_init(project_path: str, force: bool = False) -> int:
    project_root = Path(project_path).expanduser().resolve()
    project_root.mkdir(parents=True, exist_ok=True)

    for relative_dir in PROJECT_DIRS:
        (project_root / relative_dir).mkdir(parents=True, exist_ok=True)

    write_text_file(
        project_file(project_root),
        to_json_compatible_yaml(project_config_template(project_root.name)),
        force=force,
    )
    write_text_file(
        codebook_file(project_root),
        to_json_compatible_yaml(codebook_template()),
        force=force,
    )
    write_text_file(labels_file(project_root), labels_csv_template(), force=force)
    write_text_file(
        project_root / "README.md",
        project_readme_template(project_root.name),
        force=force,
    )
    write_text_file(project_root / ".gitignore", gitignore_template(), force=force)

    print(f"Initialized CBMA project at {project_root}")
    print("Created:")
    print(f"  - {project_file(project_root)}")
    print(f"  - {codebook_file(project_root)}")
    print(f"  - {labels_file(project_root)}")
    return 0


def _doctor_report(project_path: str | None) -> tuple[list[CheckMessage], int]:
    messages: list[CheckMessage] = []
    exit_code = 0

    python_ok = sys.version_info >= (3, 10)
    messages.append(
        CheckMessage(
            "OK" if python_ok else "ERROR",
            "Python",
            platform.python_version(),
        )
    )
    if not python_ok:
        exit_code = 1

    ffmpeg_path = shutil.which("ffmpeg")
    messages.append(
        CheckMessage(
            "OK" if ffmpeg_path else "WARN",
            "ffmpeg",
            ffmpeg_path or "Not found in PATH",
        )
    )

    nvidia_smi_path = shutil.which("nvidia-smi")
    messages.append(
        CheckMessage(
            "OK" if nvidia_smi_path else "WARN",
            "nvidia-smi",
            nvidia_smi_path or "Not found in PATH",
        )
    )

    try:
        import torch  # type: ignore

        messages.append(CheckMessage("OK", "torch", getattr(torch, "__version__", "unknown")))
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            details = []
            for index in range(device_count):
                details.append(torch.cuda.get_device_name(index))
            messages.append(
                CheckMessage("OK", "CUDA", f"{device_count} device(s): {', '.join(details)}")
            )
        else:
            messages.append(CheckMessage("WARN", "CUDA", "torch is installed but CUDA is not available"))
    except ImportError:
        messages.append(CheckMessage("WARN", "torch", "Not installed"))

    disk_base = Path(project_path).expanduser().resolve() if project_path else Path.cwd()
    usage = shutil.disk_usage(disk_base)
    free_gb = usage.free / (1024 ** 3)
    messages.append(CheckMessage("OK", "Disk", f"{free_gb:.1f} GB free at {disk_base}"))

    if project_path:
        project_root = Path(project_path).expanduser().resolve()
        if not project_root.exists():
            messages.append(CheckMessage("ERROR", "Project", f"Directory does not exist: {project_root}"))
            exit_code = 1
        else:
            project_config = project_file(project_root)
            messages.append(
                CheckMessage(
                    "OK" if project_config.exists() else "WARN",
                    "Project config",
                    str(project_config) if project_config.exists() else f"Missing: {project_config}",
                )
            )
            writable = is_writable_directory(project_root / "runs")
            messages.append(
                CheckMessage(
                    "OK" if writable else "ERROR",
                    "Write access",
                    str(project_root / "runs"),
                )
            )
            if not writable:
                exit_code = 1

    return messages, exit_code


def run_doctor(project_path: str | None = None, json_output: bool = False) -> int:
    messages, exit_code = _doctor_report(project_path)
    if json_output:
        payload = {
            "ok": exit_code == 0,
            "messages": [message.to_dict() for message in messages],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(format_messages(messages))
    return exit_code


def run_ui_serve(project_path: str, host: str = "127.0.0.1", port: int = 7860, reload: bool = False) -> int:
    try:
        import uvicorn  # type: ignore
    except ImportError:
        print("[ERROR] UI dependencies are not installed. Run `pip install -e .[ui]` in workflow/.")
        return 1

    project_root = Path(project_path).expanduser().resolve()
    os.environ["CBMA_UI_PROJECT"] = str(project_root)
    print(f"Serving CBMA UI for {project_root} at http://{host}:{port}")
    uvicorn.run("cbma.ui_api.app:create_app_from_env", factory=True, host=host, port=port, reload=reload)
    return 0


def _load_project_config(project_root: Path) -> dict:
    config_path = project_file(project_root)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing file: {config_path}")

    config = load_structured_text(config_path)
    if not isinstance(config, dict):
        raise ValueError("project.yaml top-level object must be a mapping")
    return config


def _project_paths(project_root: Path, config: dict) -> dict[str, Path]:
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


def _normalize_label_row(row: dict[str, str]) -> dict[str, str]:
    normalized = {key: ("" if value is None else str(value).strip()) for key, value in row.items()}
    label = str(int(normalized["label"]))
    normalized["label"] = label
    normalized["gold_label"] = label
    normalized["final_code"] = label
    normalized["split"] = normalized.get("split", "").strip().lower()
    return normalized


def _build_split_fieldnames(rows: list[dict[str, str]]) -> list[str]:
    ordered = ["video_id", "video_path", "label", "split", "gold_label", "final_code"]
    extras: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in ordered and key not in extras:
                extras.append(key)
    return ordered + extras


def _allocation_from_ratios(total: int, ratios: dict[str, float]) -> dict[str, int]:
    exact = {name: max(0.0, float(value)) * total for name, value in ratios.items()}
    counts = {name: int(value) for name, value in exact.items()}
    remaining = total - sum(counts.values())
    remainders = sorted(
        ((exact[name] - counts[name], name) for name in ratios.keys()),
        key=lambda item: item[0],
        reverse=True,
    )
    for _, name in remainders[:remaining]:
        counts[name] += 1
    return counts


def _stratified_split_rows(
    rows: list[dict[str, str]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["label"], []).append(row)

    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []

    for label, label_rows in grouped.items():
        rng = random.Random(f"{seed}:{label}")
        shuffled = list(label_rows)
        rng.shuffle(shuffled)
        counts = _allocation_from_ratios(
            len(shuffled),
            {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        )

        train_end = counts["train"]
        val_end = train_end + counts["val"]
        train_rows.extend(shuffled[:train_end])
        val_rows.extend(shuffled[train_end:val_end])
        test_rows.extend(shuffled[val_end:])

    random.Random(seed).shuffle(train_rows)
    random.Random(seed + 1).shuffle(val_rows)
    random.Random(seed + 2).shuffle(test_rows)
    return train_rows, val_rows, test_rows


def _explicit_split_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []

    split_values = {row.get("split", "") for row in rows}
    if "" in split_values and len(split_values) > 1:
        raise ValueError("labels.csv mixes explicit split values with empty split values")

    for row in rows:
        split_value = row.get("split", "")
        if split_value == "train":
            train_rows.append(row)
        elif split_value == "val":
            val_rows.append(row)
        elif split_value == "test":
            test_rows.append(row)
        else:
            raise ValueError(f"Unsupported split value: {split_value}")
    return train_rows, val_rows, test_rows


def _split_summary_payload(train_rows: list[dict[str, str]], val_rows: list[dict[str, str]], test_rows: list[dict[str, str]], candidate_sizes: list[int]) -> dict:
    def label_counts(rows: list[dict[str, str]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["label"]] = counts.get(row["label"], 0) + 1
        return counts

    return {
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "test_count": len(test_rows),
        "train_label_counts": label_counts(train_rows),
        "val_label_counts": label_counts(val_rows),
        "test_label_counts": label_counts(test_rows),
        "generated_train_sizes": candidate_sizes,
    }


def run_split_create(project_path: str, force: bool = False, json_output: bool = False) -> int:
    project_root = Path(project_path).expanduser().resolve()
    config = _load_project_config(project_root)
    paths = _project_paths(project_root, config)

    _, raw_rows = load_csv_rows(paths["labels"])
    rows = [_normalize_label_row(row) for row in raw_rows]
    if not rows:
        print("[ERROR] labels.csv has no rows to split")
        return 1

    defaults = config.get("defaults", {})
    split_strategy = defaults.get("split_strategy", {})
    training = defaults.get("training", {})

    train_ratio = float(split_strategy.get("train", 0.7))
    val_ratio = float(split_strategy.get("val", 0.1))
    test_ratio = float(split_strategy.get("test", 0.2))
    seed = int(split_strategy.get("seed", 42))

    split_dir = paths["splits_dir"]
    if split_dir.exists() and any(split_dir.iterdir()) and not force:
        print(f"[ERROR] Split directory already contains files: {split_dir}. Use --force to overwrite.")
        return 1
    split_dir.mkdir(parents=True, exist_ok=True)

    if any(row.get("split") for row in rows):
        train_rows, val_rows, test_rows = _explicit_split_rows(rows)
        mode_used = "explicit"
    else:
        train_rows, val_rows, test_rows = _stratified_split_rows(
            rows,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        mode_used = "stratified"

    if not train_rows:
        print("[ERROR] Split creation produced no training rows.")
        return 1
    if not test_rows:
        print("[ERROR] Split creation produced no test rows.")
        return 1

    candidate_sizes = sorted(
        {
            int(size)
            for size in training.get("candidate_n", [])
            if isinstance(size, int) or (isinstance(size, str) and str(size).isdigit())
        }
    )
    usable_sizes = [size for size in candidate_sizes if 0 < size <= len(train_rows)]
    if not usable_sizes:
        usable_sizes = [len(train_rows)]

    fieldnames = _build_split_fieldnames(rows)
    train_export = [{**row, "split": "train"} for row in train_rows]
    val_export = [{**row, "split": "val"} for row in val_rows]
    test_export = [{**row, "split": "test"} for row in test_rows]

    write_csv_rows(split_dir / "train_pool.csv", fieldnames, train_export)
    write_csv_rows(split_dir / "test_main.csv", fieldnames, test_export)
    if val_export:
        write_csv_rows(split_dir / "val_main.csv", fieldnames, val_export)

    for size in usable_sizes:
        write_csv_rows(split_dir / f"train_{size}.csv", fieldnames, train_export[:size])

    summary = _split_summary_payload(train_export, val_export, test_export, usable_sizes)
    summary["mode"] = mode_used
    summary["seed"] = seed
    summary["split_dir"] = str(split_dir)
    write_text_file(split_dir / "split_summary.json", json.dumps(summary, indent=2, ensure_ascii=False) + "\n", force=True)

    if json_output:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(f"Created split artifacts in {split_dir}")
        print(f"  mode: {mode_used}")
        print(f"  train: {len(train_export)}")
        print(f"  val: {len(val_export)}")
        print(f"  test: {len(test_export)}")
        print(f"  train sizes: {usable_sizes}")
    return 0


def _resolve_baseline_script() -> Path:
    project_release_root = Path(__file__).resolve().parents[3]
    candidates = [
        project_release_root / "qwen2" / "src" / "qwen2" / "eval_baselines.py",
        project_release_root / "qwen2" / "src" / "eval_baselines.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find a release qwen2 baseline script in project_release/qwen2/src.")


def _resolve_train_script() -> Path:
    project_release_root = Path(__file__).resolve().parents[3]
    candidate = project_release_root / "qwen2" / "src" / "qwen2" / "train_scaling_qwen.py"
    if candidate.exists():
        return candidate
    raise FileNotFoundError("Could not find the release qwen2 training script in project_release/qwen2/src/qwen2.")


def _timestamped_run_dir(runs_dir: Path, prefix: str) -> Path:
    run_dir = runs_dir / f"{prefix}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _default_model_path(project_paths: dict[str, Path], model_name: str) -> Path:
    models_dir = project_paths["models_dir"]
    mapping = {
        "qwen2-vl-7b-instruct": models_dir / "qwen2" / "Qwen2-VL-7B-Instruct",
    }
    return mapping.get(model_name, models_dir / model_name)


def run_baseline_run(
    project_path: str,
    methods: list[str] | None = None,
    dry_run: bool = False,
    json_output: bool = False,
    model_name: str | None = None,
    model_path_override: str | None = None,
) -> int:
    project_root = Path(project_path).expanduser().resolve()
    config = _load_project_config(project_root)
    project_paths = _project_paths(project_root, config)
    defaults = config.get("defaults", {})
    baseline_defaults = defaults.get("baseline", {})

    split_dir = project_paths["splits_dir"]
    test_csv_name = str(baseline_defaults.get("test_csv", "test_main.csv"))
    test_csv_path = split_dir / test_csv_name
    if not test_csv_path.exists():
        print(f"[ERROR] Missing split file: {test_csv_path}. Run `cbma split create` first.")
        return 1

    selected_model = model_name or str(defaults.get("model", "qwen2-vl-7b-instruct"))
    selected_methods = methods or list(defaults.get("baselines", ["zeroshot", "rule"]))
    script_path = _resolve_baseline_script()

    runs_dir = project_paths["runs_dir"]
    run_dir = _timestamped_run_dir(runs_dir, "baseline")
    eval_dir = run_dir / "baselines" / "qwen2"
    cache_dir = run_dir / "cache" / "frames_cache"
    eval_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_path = (
        Path(model_path_override).expanduser().resolve()
        if model_path_override
        else _default_model_path(project_paths, selected_model)
    )

    metadata = {
        "project": str(project_root),
        "script_path": str(script_path),
        "run_dir": str(run_dir),
        "split_dir": str(split_dir),
        "test_csv": test_csv_name,
        "methods": selected_methods,
        "model_name": selected_model,
        "model_path": str(model_path),
        "dry_run": dry_run,
    }
    write_text_file(run_dir / "baseline_run.json", json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", force=True)

    if dry_run:
        if json_output:
            print(json.dumps(metadata, indent=2, ensure_ascii=False))
        else:
            print(f"Baseline dry-run ready.")
            print(f"  script: {script_path}")
            print(f"  split: {test_csv_path}")
            print(f"  model: {model_path}")
            print(f"  methods: {selected_methods}")
            print(f"  output: {eval_dir}")
        return 0

    env_updates = {
        "PROJECT_ROOT": str(project_root),
        "VIDEO_BASE_DIR": str(project_paths["videos_dir"]),
        "QWEN2_SPLITS_DIR": str(split_dir),
        "QWEN2_MODEL_DIR": str(model_path),
        "QWEN2_FRAMES_CACHE_DIR": str(cache_dir),
        "QWEN2_BASELINE_OUT_DIR": str(eval_dir),
    }

    with _temporary_env(env_updates):
        spec = importlib.util.spec_from_file_location("cbma_qwen2_eval_baselines", script_path)
        if spec is None or spec.loader is None:
            print(f"[ERROR] Could not load baseline script: {script_path}")
            return 1

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as error:
            print(f"[ERROR] Failed to import baseline script: {error}")
            return 1

        module.BASE_VIDEO_DIR = str(project_paths["videos_dir"])
        module.SPLITS_DIR = str(split_dir)
        module.MODEL_ID = str(model_path)
        module.CACHE_DIR = str(cache_dir)
        module.EVAL_OUT_DIR = str(eval_dir)
        module.GLOBAL_VIDEO_PATHS = module.build_video_path_index(module.BASE_VIDEO_DIR)

        method_map = {
            "zeroshot": "zeroshot",
            "zero-shot": "zeroshot",
            "rule": "rule",
            "rulebased": "rulebased",
            "rule-based": "rule-based",
        }
        normalized_methods: list[str] = []
        for method in selected_methods:
            key = method.strip().lower()
            if key not in method_map:
                print(f"[ERROR] Unsupported baseline method: {method}")
                return 1
            normalized_methods.append(method_map[key])

        if hasattr(module, "run_selected_baselines"):
            try:
                module.run_selected_baselines(normalized_methods, test_csv_name=test_csv_name)
            except Exception as error:
                print(f"[ERROR] Baseline run failed: {error}")
                return 1
        else:
            selected_pairs: list[tuple[str, str]] = []
            legacy_method_map = {
                "zeroshot": ("path1_zeroshot", module.PROMPT_PATH1),
                "rule": ("path2_rulebased", module.PROMPT_PATH2),
                "rulebased": ("path2_rulebased", module.PROMPT_PATH2),
                "rule-based": ("path2_rulebased", module.PROMPT_PATH2),
            }
            for method in normalized_methods:
                selected_pairs.append(legacy_method_map[method])

            try:
                processor = module.AutoProcessor.from_pretrained(module.MODEL_ID)
                model = module.Qwen2VLForConditionalGeneration.from_pretrained(
                    module.MODEL_ID,
                    torch_dtype=module.torch.bfloat16,
                    device_map="auto",
                    attn_implementation="sdpa",
                )
                model.eval()
            except Exception as error:
                print(f"[ERROR] Failed to load model or processor: {error}")
                return 1

            try:
                for path_name, prompt_text in selected_pairs:
                    module.run_baseline_eval(model, processor, path_name, prompt_text, test_csv_name=test_csv_name)
            finally:
                del model
                if hasattr(module, "gc"):
                    module.gc.collect()
                if getattr(module.torch, "cuda", None) and module.torch.cuda.is_available():
                    module.torch.cuda.empty_cache()

    print(f"Baseline run completed. Outputs written to {eval_dir}")
    return 0


def _parse_size_list(raw_sizes: list[int] | None, fallback_sizes: object) -> list[int]:
    values = raw_sizes if raw_sizes is not None else fallback_sizes
    parsed: list[int] = []
    if not isinstance(values, list):
        return parsed
    for value in values:
        if isinstance(value, int):
            if value > 0:
                parsed.append(value)
        elif isinstance(value, str) and value.isdigit():
            parsed.append(int(value))
    return sorted(set(parsed))


def _available_split_sizes(split_dir: Path) -> list[int]:
    sizes: list[int] = []
    for path in split_dir.glob("train_*.csv"):
        stem = path.stem
        try:
            sizes.append(int(stem.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(set(sizes))


def run_train_sweep(
    project_path: str,
    sizes: list[int] | None = None,
    dry_run: bool = False,
    json_output: bool = False,
    model_name: str | None = None,
    model_path_override: str | None = None,
) -> int:
    project_root = Path(project_path).expanduser().resolve()
    config = _load_project_config(project_root)
    project_paths = _project_paths(project_root, config)
    defaults = config.get("defaults", {})
    training_defaults = defaults.get("training", {})

    split_dir = project_paths["splits_dir"]
    configured_sizes = _parse_size_list(sizes, training_defaults.get("candidate_n", []))
    available_sizes = _available_split_sizes(split_dir)
    if sizes is None:
        train_sizes = [size for size in configured_sizes if size in available_sizes] or available_sizes
    else:
        train_sizes = configured_sizes

    if not train_sizes:
        print("[ERROR] No training sizes available. Set defaults.training.candidate_n or pass --sizes.")
        return 1

    missing_split_files = [size for size in train_sizes if size not in available_sizes]
    if missing_split_files:
        print(
            "[ERROR] Missing split files for sizes: "
            + ", ".join(str(size) for size in missing_split_files)
            + ". Run `cbma split create` first."
        )
        return 1

    selected_model = model_name or str(defaults.get("model", "qwen2-vl-7b-instruct"))
    model_path = (
        Path(model_path_override).expanduser().resolve()
        if model_path_override
        else _default_model_path(project_paths, selected_model)
    )
    script_path = _resolve_train_script()

    runs_dir = project_paths["runs_dir"]
    run_dir = _timestamped_run_dir(runs_dir, "train-sweep")
    artifact_root = run_dir / "artifacts" / "qwen2"
    artifact_root.mkdir(parents=True, exist_ok=True)

    metadata = {
        "project": str(project_root),
        "script_path": str(script_path),
        "run_dir": str(run_dir),
        "artifact_root": str(artifact_root),
        "split_dir": str(split_dir),
        "sizes": train_sizes,
        "model_name": selected_model,
        "model_path": str(model_path),
        "dry_run": dry_run,
    }
    write_text_file(run_dir / "train_sweep.json", json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", force=True)

    if dry_run:
        if json_output:
            print(json.dumps(metadata, indent=2, ensure_ascii=False))
        else:
            print("Train sweep dry-run ready.")
            print(f"  script: {script_path}")
            print(f"  split dir: {split_dir}")
            print(f"  sizes: {train_sizes}")
            print(f"  model: {model_path}")
            print(f"  artifact root: {artifact_root}")
        return 0

    env_updates = {
        "PROJECT_ROOT": str(project_root),
        "VIDEO_BASE_DIR": str(project_paths["videos_dir"]),
        "QWEN2_SPLITS_DIR": str(split_dir),
        "QWEN2_MODEL_DIR": str(model_path),
        "QWEN2_ARTIFACT_ROOT": str(artifact_root),
    }

    with _temporary_env(env_updates):
        spec = importlib.util.spec_from_file_location("cbma_qwen2_train_scaling", script_path)
        if spec is None or spec.loader is None:
            print(f"[ERROR] Could not load training script: {script_path}")
            return 1

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as error:
            print(f"[ERROR] Failed to import training script: {error}")
            return 1

        if not hasattr(module, "train_scaling"):
            print(f"[ERROR] Training script does not expose train_scaling(): {script_path}")
            return 1

        try:
            for size in train_sizes:
                module.train_scaling(size)
        except Exception as error:
            print(f"[ERROR] Training sweep failed: {error}")
            return 1

    print(f"Train sweep completed. Outputs written to {artifact_root}")
    return 0


def run_eval_run(
    project_path: str,
    run_dir: str | None = None,
    recommend_file: str | None = None,
    output: str | None = None,
) -> int:
    try:
        return run_eval(
            project_path=project_path,
            run_dir=run_dir,
            recommend_file=recommend_file,
            output=output,
        )
    except FileNotFoundError as error:
        print(f"[ERROR] {error}")
        return 1
    except ValueError as error:
        print(f"[ERROR] {error}")
        return 1
    except RuntimeError as error:
        print(f"[ERROR] {error}")
        return 1


def run_train_recommend_n(
    project_path: str,
    run_dir: str | None = None,
    output: str | None = None,
) -> int:
    try:
        return run_recommend_n(
            project_path=project_path,
            run_dir=run_dir,
            output=output,
        )
    except FileNotFoundError as error:
        print(f"[ERROR] {error}")
        return 1
    except ValueError as error:
        print(f"[ERROR] {error}")
        return 1


def run_report_build_command(
    project_path: str,
    eval_dir: str | None = None,
    output: str | None = None,
    format_name: str = "markdown",
    include_errors: bool = False,
    include_confusion: bool = False,
) -> int:
    return run_report_build(
        project_path=project_path,
        eval_dir=eval_dir,
        output=output,
        format_name=format_name,
        include_errors=include_errors,
        include_confusion=include_confusion,
    )


def _validate_project(project_root: Path) -> tuple[list[CheckMessage], int]:
    messages: list[CheckMessage] = []
    errors = 0

    if not project_root.exists():
        return [CheckMessage("ERROR", "Project", f"Directory does not exist: {project_root}")], 1

    config_path = project_file(project_root)
    if not config_path.exists():
        return [CheckMessage("ERROR", "project.yaml", f"Missing file: {config_path}")], 1

    try:
        config = load_structured_text(config_path)
    except ValueError as error:
        return [CheckMessage("ERROR", "project.yaml", str(error))], 1

    if not isinstance(config, dict):
        return [CheckMessage("ERROR", "project.yaml", "Top-level object must be a mapping")], 1

    for key in ("project_name", "paths", "defaults"):
        if key not in config:
            messages.append(CheckMessage("ERROR", "project.yaml", f"Missing required key: {key}"))
            errors += 1

    paths = config.get("paths", {})
    if not isinstance(paths, dict):
        messages.append(CheckMessage("ERROR", "project.yaml", "`paths` must be a mapping"))
        return messages, 1

    required_path_keys = ("labels", "codebook", "videos_dir")
    for key in required_path_keys:
        if key not in paths:
            messages.append(CheckMessage("ERROR", "project.yaml", f"Missing paths.{key}"))
            errors += 1

    if errors:
        return messages, 1

    labels_path = resolve_path(project_root, str(paths["labels"]))
    codebook_path = resolve_path(project_root, str(paths["codebook"]))
    videos_dir = resolve_path(project_root, str(paths["videos_dir"]))

    if not codebook_path.exists():
        messages.append(CheckMessage("ERROR", "codebook", f"Missing file: {codebook_path}"))
        errors += 1
        codebook = None
    else:
        try:
            codebook = load_structured_text(codebook_path)
        except ValueError as error:
            messages.append(CheckMessage("ERROR", "codebook", str(error)))
            errors += 1
            codebook = None

    label_id_set: set[int] = set()
    if isinstance(codebook, dict):
        raw_labels = codebook.get("labels")
        if not isinstance(raw_labels, list) or not raw_labels:
            messages.append(CheckMessage("ERROR", "codebook", "`labels` must be a non-empty list"))
            errors += 1
        else:
            seen_ids: set[int] = set()
            for index, item in enumerate(raw_labels):
                if not isinstance(item, dict):
                    messages.append(CheckMessage("ERROR", "codebook", f"labels[{index}] must be a mapping"))
                    errors += 1
                    continue
                for field in ("id", "name", "definition"):
                    if field not in item:
                        messages.append(CheckMessage("ERROR", "codebook", f"labels[{index}] missing `{field}`"))
                        errors += 1
                try:
                    label_id = int(item["id"])
                except Exception:
                    messages.append(CheckMessage("ERROR", "codebook", f"labels[{index}].id must be an integer"))
                    errors += 1
                    continue
                if label_id in seen_ids:
                    messages.append(CheckMessage("ERROR", "codebook", f"Duplicate label id: {label_id}"))
                    errors += 1
                seen_ids.add(label_id)
                label_id_set.add(label_id)

    if not labels_path.exists():
        messages.append(CheckMessage("ERROR", "labels.csv", f"Missing file: {labels_path}"))
        errors += 1
        rows: list[dict[str, str]] = []
        fieldnames: list[str] = []
    else:
        fieldnames, rows = load_csv_rows(labels_path)
        required_columns = {"video_id", "video_path", "label"}
        missing_columns = sorted(required_columns.difference(fieldnames))
        if missing_columns:
            messages.append(
                CheckMessage(
                    "ERROR",
                    "labels.csv",
                    f"Missing required columns: {', '.join(missing_columns)}",
                )
            )
            errors += 1

    if errors:
        return messages, 1

    if not rows:
        messages.append(CheckMessage("WARN", "labels.csv", "No labeled rows yet"))
        messages.append(CheckMessage("OK", "videos_dir", str(videos_dir)))
        return messages, 0

    seen_video_ids: set[str] = set()
    csv_label_set: set[int] = set()
    missing_video_paths: list[str] = []
    invalid_splits: list[str] = []

    allowed_splits = {"train", "val", "test", ""}

    for row_index, row in enumerate(rows, start=2):
        video_id = str(row.get("video_id", "")).strip()
        if not video_id:
            messages.append(CheckMessage("ERROR", "labels.csv", f"Row {row_index}: empty video_id"))
            errors += 1
            continue
        if video_id in seen_video_ids:
            messages.append(CheckMessage("ERROR", "labels.csv", f"Duplicate video_id: {video_id}"))
            errors += 1
        seen_video_ids.add(video_id)

        try:
            label = int(str(row.get("label", "")).strip())
        except ValueError:
            messages.append(CheckMessage("ERROR", "labels.csv", f"Row {row_index}: label must be an integer"))
            errors += 1
            continue

        csv_label_set.add(label)
        if label_id_set and label not in label_id_set:
            messages.append(CheckMessage("ERROR", "labels.csv", f"Row {row_index}: unknown label id {label}"))
            errors += 1

        split_value = str(row.get("split", "")).strip().lower()
        if split_value not in allowed_splits:
            invalid_splits.append(f"Row {row_index}: {split_value}")

        raw_video_path = str(row.get("video_path", "")).strip()
        if not raw_video_path:
            messages.append(CheckMessage("ERROR", "labels.csv", f"Row {row_index}: empty video_path"))
            errors += 1
            continue

        direct_path = Path(raw_video_path)
        candidate_paths = []
        if direct_path.is_absolute():
            candidate_paths.append(direct_path)
        else:
            candidate_paths.append((project_root / raw_video_path).resolve())
            candidate_paths.append((videos_dir / raw_video_path).resolve())

        if not any(candidate.exists() for candidate in candidate_paths):
            missing_video_paths.append(f"Row {row_index}: {raw_video_path}")

    if invalid_splits:
        messages.append(
            CheckMessage(
                "ERROR",
                "labels.csv",
                "Invalid split values: " + "; ".join(invalid_splits[:5]),
            )
        )
        errors += 1

    if missing_video_paths:
        preview = "; ".join(missing_video_paths[:5])
        messages.append(CheckMessage("ERROR", "video files", f"Missing files: {preview}"))
        errors += 1

    if label_id_set and csv_label_set and csv_label_set != label_id_set:
        missing_from_data = sorted(label_id_set.difference(csv_label_set))
        if missing_from_data:
            messages.append(
                CheckMessage(
                    "WARN",
                    "labels.csv",
                    f"No samples currently present for label ids: {missing_from_data}",
                )
            )

    messages.append(CheckMessage("OK", "rows", str(len(rows))))
    messages.append(CheckMessage("OK", "unique labels", str(sorted(csv_label_set))))
    messages.append(CheckMessage("OK", "videos_dir", str(videos_dir)))

    return messages, 1 if errors else 0


def run_validate(project_path: str, json_output: bool = False) -> int:
    project_root = Path(project_path).expanduser().resolve()
    messages, exit_code = _validate_project(project_root)
    if json_output:
        payload = {
            "ok": exit_code == 0,
            "project": str(project_root),
            "messages": [message.to_dict() for message in messages],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(format_messages(messages))
    return exit_code
