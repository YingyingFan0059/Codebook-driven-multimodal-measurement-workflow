---
name: cbma-train-sweep-dryrun
description: Resolve the structure of a CBMA training sweep without launching training, using the existing dry-run CLI workflow.
---

# cbma-train-sweep-dryrun

## purpose

Inspect whether a scaling-oriented training sweep is ready by resolving candidate sizes, split files, model paths, and artifact locations without launching actual training.

## when to use

Use this skill when:

- split artifacts already exist
- the user wants to inspect training sweep readiness before allocating GPU time
- candidate sizes or model paths need confirmation

This skill is the correct preparation step before real training on AutoDL or another managed GPU host.

## CLI mapping

Primary command:

```powershell
cbma train sweep --project <project_path> --dry-run
```

Optional size override:

```powershell
cbma train sweep --project <project_path> --dry-run --sizes 200,400,800
```

Optional model override:

```powershell
cbma train sweep --project <project_path> --dry-run --model <model_name> --model-path <local_model_path>
```

Optional machine-readable mode:

```powershell
cbma train sweep --project <project_path> --dry-run --json
```

## inputs

Required:

- `project_path`: existing CBMA project directory with split artifacts

Optional:

- `sizes`: comma-separated training sizes
- `model_name`: CLI-level model alias override
- `model_path`: explicit local model path override
- `json_output`: request machine-readable output

## outputs

Expected run outputs:

- a timestamped train-sweep run directory under `runs/`
- `train_sweep.json` metadata

Expected dry-run reporting:

- resolved training script path
- split directory
- selected training sizes
- resolved model path
- intended artifact root

## guardrails

- Do not treat dry-run success as proof that real training will succeed.
- Do not start training, allocate GPU jobs, or modify backend training logic.
- Do not fabricate missing `train_{N}.csv` files.
- If requested sizes are unavailable, report that mismatch instead of silently changing the experiment design.
