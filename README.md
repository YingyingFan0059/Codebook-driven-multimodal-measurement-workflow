# CBMA

CBMA is a codebook-driven multimodal analysis workflow for reproducible research.

CBMA is organized as a lightweight, offline, local-first research toolkit rather than a hosted AI product. It is intended for social science, communication, and multimodal research workflows that need explicit data structure, controlled evaluation, and filesystem-based reproducibility.

## What CBMA Covers

- Workflow pipeline for project setup, validation, split generation, baseline dry-run, training sweep dry-run, standardized evaluation, and report export
- Decision handoff through a `recommend_n.json` artifact for turning scaling results into a final evaluation choice
- CLI generation of `recommend_n.json` via `cbma train recommend-n`
- Standardized evaluation protocol via `cbma eval run`
- Structured report export via `cbma report build`

## Who This Is For

- Social science researchers working with codebooks and labeled media
- Communication and propaganda researchers working with short video data
- Multimodal researchers who want a reproducible offline workflow rather than a web product

## Execution Model

- Offline execution
- Local filesystem outputs
- User-managed GPU environments for real inference and training, such as AutoDL
- No cloud dependency
- No hosted API dependency

Local development, validation, dry-run, and report construction can be done without a GPU. Real baseline inference, real training, and real evaluation still require a user-provided local model path and a suitable runtime.

## Minimal Flow

1. Initialize or prepare a local project.
2. Validate codebook, labels, and file paths.
3. Create split artifacts.
4. Run baseline and train sweep in `--dry-run` mode to verify wiring.
5. Produce or provide a `recommend_n.json` decision artifact.
6. Run standardized evaluation on the fixed test split.
7. Export a structured report from the eval directory.

## Included Demo

The smallest release demo lives in `project_release/demo/`.

It is designed for a no-GPU environment and includes:

- a synthetic demo project
- fake labels and placeholder video files
- a minimal eval directory for `report build`
- a documented dry-run path

Start with `project_release/demo/README.md`.

## No-GPU Quickstart

Install the workflow package from `project_release/workflow`, then run the included demo:

```bash
cd project_release/workflow
python -m pip install -e .

cbma validate --project ../demo/demo_project
cbma split create --project ../demo/demo_project --force
cbma baseline run --project ../demo/demo_project --dry-run
cbma train sweep --project ../demo/demo_project --dry-run
cbma train recommend-n --project ../demo/demo_project --run-dir ../demo/demo_project/runs/train-sweep-sample
cbma report build --project ../demo/demo_project
```

This path does not run a model. It validates the project shape, produces split artifacts, writes dry-run metadata, and builds a report from the included synthetic eval inputs.

## Full Workflow In A User-Managed GPU Environment

```bash
cbma init my_project
cbma validate --project my_project
cbma split create --project my_project
cbma baseline run --project my_project --dry-run
cbma train sweep --project my_project --dry-run
cbma train recommend-n --project my_project --run-dir <runs_dir>/train-sweep-...
cbma eval run --project my_project --run-dir <runs_dir>/train-sweep-...
cbma report build --project my_project
```

## Output Shape

`cbma report build` writes a standardized report directory under an eval run:

```text
<runs_dir>/
  eval-.../
    eval_result.json
    eval_metadata.json
    report/
      report.md
      metrics.json
      run_summary.json
      per_class_f1.csv          # optional
      confusion_matrix.csv      # optional
      error_cases.csv           # optional
```

Example report excerpt:

```markdown
# CBMA Evaluation Report

## 2. Headline Metrics
- accuracy: 0.78
- macro_f1: 0.74

## 5. Per-class Results
unavailable

## 6. Confusion Structure
- confusion matrix: unavailable
```

The `unavailable` state is expected when the underlying eval output does not provide richer raw artifacts.

## Environment Notes

- No GPU required for `init`, `validate`, `split`, dry-run commands, or `report build`
- A GPU is typically required for real baseline inference, real LoRA training, and real standardized evaluation
- AutoDL or another user-managed GPU host is the expected environment for model execution

## Open-Source Boundaries

CBMA does not provide:

- source videos
- platform scraping
- model weights
- automatic model download
- an online service

Users are expected to provide:

- their own local data
- their own local model path
- their own GPU environment when running real inference or training

## Current Limits

- Eval provenance may be partially unavailable when the upstream eval outputs are sparse
- `raw_eval` enhancement depends on whatever the backend eval script actually produced
- Archived research directories remain in the repository for context, but they are not part of the V1 workflow core

## Repository Map

- `project_release/workflow/`: maintained CLI core
- `project_release/demo/`: synthetic release demo
- `project_release/qwen2/`: backend release scripts plus reproducibility appendix
- `project_release/qwen3/`: archived research assets
- `Qwen2/` and `Qwen3/`: legacy research directories retained as archive context
