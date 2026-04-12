---
name: cbma-data-check
description: Check whether a CBMA project is structurally ready by running environment inspection and project validation against the existing CLI workflow.
---

# cbma-data-check

## purpose

Assess whether a CBMA project is ready for downstream workflow steps by checking environment prerequisites and validating project inputs.

## when to use

Use this skill when:

- a user has already initialized a CBMA project
- labels, codebook, or video paths may be incomplete
- you need to decide whether split generation or dry-run execution can proceed

This skill is appropriate before `split create`, `baseline run --dry-run`, or `train sweep --dry-run`.

## CLI mapping

Environment inspection:

```powershell
cbma doctor --project <project_path>
```

Project validation:

```powershell
cbma validate --project <project_path>
```

Optional machine-readable mode:

```powershell
cbma doctor --project <project_path> --json
cbma validate --project <project_path> --json
```

## inputs

Required:

- `project_path`: existing CBMA project directory

Optional:

- `json_output`: whether command output should be machine-readable

## outputs

Expected command outputs:

- environment checks for Python, disk, optional GPU-related tools, and project writability
- validation checks for `project.yaml`, `data/codebook.yaml`, `data/labels.csv`, labels, splits, and referenced video paths

Expected decision surface:

- whether the project is ready for the next CLI step
- which problems are blocking
- which issues are warnings rather than hard failures

## guardrails

- Do not silently repair invalid inputs; this skill is for inspection and reporting.
- Do not modify `project.yaml`, `codebook.yaml`, or `labels.csv` as part of the check.
- Do not assume GPU availability is required for validation itself.
- Treat missing files, invalid labels, and missing videos as validation issues, not as reasons to invent substitute data.
