# project_release

`project_release/` is the release-facing part of the repository.

It is organized around a lightweight workflow core, a synthetic demo, and preserved backend research assets needed to explain method provenance without turning the repository into a product platform.

## Structure

```text
project_release/
  README.md
  LIGHTWEIGHT_TOOLKIT_V1.md
  WORKFLOW_FIRST_V1.md
  workflow/
  demo/
  qwen2/
  qwen3/
```

## Roles

- `workflow/`: maintained CBMA V1 CLI core
- `demo/`: smallest synthetic project for release onboarding
- `qwen2/`: backend release scripts plus reproducibility appendix
- `qwen3/`: archived research assets, not part of the V1 core
- `WORKFLOW_FIRST_V1.md`: legacy design document kept for history

## Release Boundary

The main narrative for V1 is:

- workflow-first
- local-first
- reproducibility-first

The V1 release is not presented as:

- a hosted service
- a web application
- a multi-model platform
- a bundled data release
