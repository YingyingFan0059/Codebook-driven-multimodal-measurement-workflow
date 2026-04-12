---
name: cbma-project-init
description: Initialize a new CBMA local project workspace when the user needs the standard directory layout, starter config files, and a documented starting point for the CLI workflow.
---

# cbma-project-init

## purpose

Create a standard CBMA local project skeleton so that later CLI workflow steps operate on a predictable directory layout and config surface.

## when to use

Use this skill when:

- a user wants to start a new CBMA project
- a workspace does not yet contain `project.yaml`
- the next step requires a valid CBMA project layout

Do not use this skill when the project already exists and the task is validation, split generation, or run inspection.

## CLI mapping

Primary command:

```powershell
cbma init <project_path>
```

Optional overwrite path:

```powershell
cbma init <project_path> --force
```

## inputs

Required:

- `project_path`: target directory for the CBMA project

Optional:

- `force`: overwrite template files if they already exist

## outputs

Expected filesystem outputs:

- `project.yaml`
- `data/codebook.yaml`
- `data/labels.csv`
- `README.md`
- `.gitignore`
- standard project subdirectories such as `data/`, `cache/`, `models/`, `runs/`, and `exports/`

Expected command outcome:

- exit code `0` on successful initialization
- printed paths for the created template files

## guardrails

- Do not treat this skill as a data import step; it only initializes the project structure.
- Do not invent labels, videos, or annotations beyond the default templates already produced by `cbma init`.
- Do not overwrite an existing project unless `--force` is explicitly intended.
- Do not add extra files or alternate directory conventions outside the existing CBMA CLI workflow.
