# CBMA Demo

This demo is the smallest release-facing walkthrough for CBMA.

It is designed to show the workflow shape without requiring a GPU, a real model, or real videos. The included project uses fake labels, placeholder video files, and a synthetic eval directory so that `cbma report build` can run in a local CPU-only environment.

## Included Files

```text
demo/
  README.md
  demo_project/
    project.yaml
    data/
      codebook.yaml
      labels.csv
      videos/
    runs/
      train-sweep-sample/
        recommend_n.json
        train_sweep.json
      eval-20260411-sample/
        eval_result.json
        eval_metadata.json
```

## Local No-GPU Walkthrough

From `project_release/workflow/`:

```bash
python -m pip install -e .

cbma validate --project ../demo/demo_project
cbma split create --project ../demo/demo_project --force
cbma baseline run --project ../demo/demo_project --dry-run
cbma train sweep --project ../demo/demo_project --dry-run
cbma train recommend-n --project ../demo/demo_project --run-dir ../demo/demo_project/runs/train-sweep-sample
cbma report build --project ../demo/demo_project
```

What this does:

- validates the synthetic project
- creates project-local split artifacts
- writes baseline dry-run metadata
- writes train sweep dry-run metadata
- regenerates `recommend_n.json` from the included synthetic sweep metrics
- builds a report from the included synthetic eval run

No GPU is required for this path.

## Full Target Workflow

The intended full workflow surface is:

```bash
cbma init demo_project
cbma validate --project demo_project
cbma split create --project demo_project
cbma baseline run --project demo_project --dry-run
cbma train sweep --project demo_project --dry-run
cbma train recommend-n --project demo_project --run-dir <runs_dir>/train-sweep-...
cbma eval run --project demo_project
cbma report build --project demo_project
```

In the public lightweight release:

- `init`, `validate`, `split`, dry-run `baseline`, dry-run `train sweep`, and `report build` are safe to demonstrate locally
- real `eval run` still depends on local model artifacts and a user-managed runtime
- the included `eval-20260411-sample/` directory is a synthetic input for the report stage, not a real model evaluation

## Report Example

The generated report follows this structure:

```markdown
# CBMA Evaluation Report

## 2. Headline Metrics
- accuracy: 0.78
- macro_f1: 0.74

## 5. Per-class Results
unavailable

## 8. Notes and Limitations
- This report is built from existing eval outputs.
- The model was not re-run during report construction.
```

`unavailable` is normal in this demo because no `raw_eval` bundle is included.
