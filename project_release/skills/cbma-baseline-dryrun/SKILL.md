---
name: cbma-baseline-dryrun
description: Resolve the inputs and run metadata for a CBMA baseline execution without loading the model, using the existing dry-run CLI path.
---

# cbma-baseline-dryrun

## purpose

Check whether a baseline run is structurally ready by resolving paths, split files, methods, model location, and output directories without starting real model inference.

## when to use

Use this skill when:

- split artifacts already exist
- the user wants to inspect baseline readiness before using GPU resources
- you need to confirm the baseline script path, test split, and local model path

This skill is especially useful before moving execution to AutoDL or another user-managed GPU environment.

## CLI mapping

Primary command:

```powershell
cbma baseline run --project <project_path> --dry-run
```

Optional method override:

```powershell
cbma baseline run --project <project_path> --dry-run --methods zeroshot,rule
```

Optional model override:

```powershell
cbma baseline run --project <project_path> --dry-run --model <model_name> --model-path <local_model_path>
```

Optional machine-readable mode:

```powershell
cbma baseline run --project <project_path> --dry-run --json
```

## inputs

Required:

- `project_path`: existing CBMA project directory with split artifacts

Optional:

- `methods`: comma-separated baseline methods
- `model_name`: CLI-level model alias override
- `model_path`: explicit local model path override
- `json_output`: request machine-readable output

## outputs

Expected run outputs:

- a timestamped baseline run directory under `runs/`
- `baseline_run.json` metadata

Expected dry-run reporting:

- resolved baseline script path
- resolved test split path
- selected methods
- resolved model path
- intended output directory

## guardrails

- Do not treat dry-run success as proof that real inference will succeed.
- Do not attempt to load the model, consume GPU memory, or launch evaluation in this skill.
- Do not add baseline methods that are not supported by the existing CLI.
- If split files are missing, report that as a blocking issue rather than trying to reconstruct them.
