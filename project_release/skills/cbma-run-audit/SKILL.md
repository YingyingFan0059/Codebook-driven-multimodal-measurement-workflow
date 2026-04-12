---
name: cbma-run-audit
description: Inspect the structure and metadata of existing CBMA run directories so an agent can summarize what has already been prepared or executed.
---

# cbma-run-audit

## purpose

Review the contents of the `runs/` directory and summarize what baseline and train-sweep runs already exist, what metadata files were produced, and whether the run structure is internally consistent.

## when to use

Use this skill when:

- a user wants to know what has already been run
- you need to inspect run metadata before deciding the next CLI step
- you need to compare prepared baseline runs and training sweep runs

This skill is useful after dry-run steps and before recommending the next action.

## CLI mapping

There is no dedicated `cbma run audit` command in the current CLI.

This skill maps to filesystem inspection of the run outputs already produced by the existing CLI workflow, especially:

- `runs/baseline-*/baseline_run.json`
- `runs/train-sweep-*/train_sweep.json`

When relevant, this skill should also inspect split artifacts and referenced output directories to confirm that the run structure is coherent with the project layout.

## inputs

Required:

- `project_path`: existing CBMA project directory

Optional:

- `run_prefix`: inspect only baseline or train-sweep runs
- `latest_only`: focus on the most recent run

## outputs

Expected audit outputs:

- discovered run directories
- run types present
- metadata file presence
- key fields such as script path, model path, split path, selected methods, or selected sizes
- a short statement of what the next reasonable CLI step is

## guardrails

- Do not invent a new CLI command for auditing.
- Do not modify run directories while auditing them.
- Treat missing metadata as an audit finding, not as a reason to synthesize replacement files.
- Keep the audit descriptive; it should report existing structure rather than reinterpret results as completed evaluation.
