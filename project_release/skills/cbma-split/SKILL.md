---
name: cbma-split
description: Generate CBMA split artifacts from an already validated project using the existing CLI split workflow.
---

# cbma-split

## purpose

Create standard split artifacts for a CBMA project so that baseline and training workflows can operate on stable train, validation, and test files.

## when to use

Use this skill when:

- the project has passed validation or is close to valid
- `labels.csv` exists and contains labeled rows
- downstream baseline or training steps need split files

Do not use this skill as a substitute for data validation.

## CLI mapping

Primary command:

```powershell
cbma split create --project <project_path>
```

Optional overwrite mode:

```powershell
cbma split create --project <project_path> --force
```

Optional machine-readable mode:

```powershell
cbma split create --project <project_path> --json
```

## inputs

Required:

- `project_path`: existing CBMA project directory

Optional:

- `force`: overwrite an existing split directory
- `json_output`: request machine-readable output

## outputs

Expected filesystem outputs under the configured split directory:

- `train_pool.csv`
- `test_main.csv`
- `val_main.csv` when available
- `train_{N}.csv`
- `split_summary.json`

Expected command outcome:

- reported split mode
- counts for train, validation, and test
- generated training sizes

## guardrails

- Do not run this skill before basic project validation unless the user explicitly accepts validation risk.
- Do not change split logic outside the existing CLI implementation.
- Do not invent additional split files or alternate naming conventions.
- If split files already exist, do not overwrite them unless `--force` is intentionally requested.
