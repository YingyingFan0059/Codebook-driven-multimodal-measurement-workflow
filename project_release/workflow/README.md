# CBMA Workflow Core

This directory contains the maintained CLI core of CBMA V1.

The workflow layer is the primary interface. It is designed for offline execution, local filesystem outputs, and reproducible handoff across project setup, split generation, dry-run planning, evaluation, and reporting. The local UI is frozen and not part of the V1 core surface.

## Current Command Surface

- `cbma init`
- `cbma doctor`
- `cbma validate`
- `cbma split create`
- `cbma baseline run`
- `cbma train sweep`
- `cbma train recommend-n`
- `cbma eval run`
- `cbma report build`

## Project Layout

`cbma init` creates a project with this shape:

```text
my_project/
  project.yaml
  README.md
  .gitignore
  data/
    codebook.yaml
    labels.csv
    videos/
  cache/
    frames/
    audio/
    hf/
  models/
    qwen2/
    internvl/
  splits/
  runs/
  exports/
```

The minimum required inputs are:

- `project.yaml`
- `data/codebook.yaml`
- `data/labels.csv`
- local video files referenced by `labels.csv`

## Workflow Stages

### 1. Initialize

Create a local project skeleton with `cbma init`.

### 2. Validate

Use `cbma doctor` for environment inspection and `cbma validate` for project, codebook, labels, and file checks.

### 3. Split

Use `cbma split create` to build `train_pool.csv`, `test_main.csv`, optional `val_main.csv`, nested `train_<N>.csv`, and `split_summary.json`.

### 4. Baseline

Use `cbma baseline run --dry-run` first. This resolves script paths, split inputs, methods, and model paths without loading a model.

### 5. Train Sweep

Use `cbma train sweep --dry-run` first. This resolves candidate sizes, output locations, and backend script wiring.

### 6. Recommend N

Use `cbma train recommend-n` to convert a sweep run plus validation metrics into a `recommend_n.json` decision artifact.

### 7. Evaluation

Use `cbma eval run` after `recommend_n.json` has been generated and the corresponding model outputs are available locally.

### 8. Report

Use `cbma report build` to build a standardized report directory from an existing eval run. This step is CPU-only and does not call a model.

## Run Outputs

The workflow writes timestamped run directories under `runs/`:

```text
runs/
  baseline-YYYYMMDD-HHMMSS/
    baseline_run.json
  train-sweep-YYYYMMDD-HHMMSS/
    train_sweep.json
  eval-YYYYMMDD-HHMMSS/
    eval_result.json
    eval_metadata.json
    raw_eval/                  # optional
    report/
      report.md
      metrics.json
      run_summary.json
```

These directories are for traceability first. Optional downstream files depend on what the backend scripts emit.

## Dry-Run Versus Real Run

Dry-run mode:

- resolves paths
- validates expected split files
- writes metadata
- does not load models
- does not require a GPU

Real run mode:

- imports backend release scripts
- may require `torch`, `ffmpeg`, and CUDA
- depends on user-managed model paths
- is intended for AutoDL or comparable environments

## Scope Boundary

This directory is not a hosted service backend, multi-user platform, or web-first application.

`src/cbma/ui_api/` remains in the repository as a frozen experimental module and is not part of the V1 core.
