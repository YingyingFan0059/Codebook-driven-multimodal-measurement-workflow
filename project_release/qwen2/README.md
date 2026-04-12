# Qwen2 Release Assets

This directory preserves the Qwen2-side backend assets used by the CBMA workflow.

In the release narrative, `project_release/qwen2/` is not a standalone product. It is a backend implementation source plus a reproducibility appendix for the current workflow core.

## Role In CBMA V1

- `src/qwen2/`: active backend script source used by the workflow wrappers
- `src/internvl/`: experimental archived research assets
- `splits/`: example split assets for reproducibility reference
- `env_record/`: reference-only environment snapshots
- `artifacts/`: optional appendix for run layout reference

## Open-Source Boundary

This release does not provide:

- source videos
- base model weights
- automatic model download
- a managed runtime

Users are expected to provide:

- their own local data
- their own local model path
- their own GPU runtime for real inference or training

## Path Convention

Use placeholders instead of machine-specific absolute paths:

```bash
export PROJECT_ROOT=<project_path>
export VIDEO_BASE_DIR=<project_path>/data/videos
export QWEN2_MODEL_DIR=<model_path>
export INTERNVL_MODEL_DIR=<model_path>
```

## Main Scripts

- `src/eval_baselines.py`
- `src/qwen2/train_scaling_qwen.py`
- `src/qwen2/eval_scaling_qwen_repro.py`

These scripts are kept as backend execution targets for the workflow wrappers. They are not exposed as the primary user interface.

## Appendix Status

- `artifacts/`: optional
- `env_record/`: reference only
- `splits/`: example only

The public release should not be read as a complete bundle of historical experiment outputs.
