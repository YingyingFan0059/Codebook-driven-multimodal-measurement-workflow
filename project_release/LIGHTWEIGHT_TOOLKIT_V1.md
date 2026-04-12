# LIGHTWEIGHT_TOOLKIT_V1

## Positioning

CBMA V1 is being defined as an offline, reproducible multimodal research toolkit for structured codebook-driven analysis.

It is not being defined as:

- a full product platform
- a hosted inference service
- a web-first system
- a general-purpose multi-model orchestration layer

The working assumption is that users manage their own runtime environment, including GPU access, model files, and local storage. Typical execution environments include AutoDL and comparable user-managed Linux/CUDA setups.

## V1 Scope

The intended V1 scope can be described as four research workflow layers:

### 1. Prepare

Prepare local project inputs and derived split artifacts.

This layer includes:

- project initialization
- environment inspection
- input validation
- split generation

### 2. Infer

Run baseline inference or training-side execution wrappers against release-side model scripts.

This layer includes:

- baseline resolution and execution
- training sweep resolution and execution

### 3. Analyze

Turn scaling outputs into research decisions and evaluative summaries.

This layer is intended to include:

- scaling comparison
- recommended training size selection
- formal evaluation
- structured error inspection

### 4. Report

Export reproducible outputs for inspection and dissemination.

This layer is intended to include:

- machine-readable metrics
- tabular summaries
- markdown reports
- reproducibility bundles

## Current Completion Status

The current state is uneven by design. CBMA V1 currently has a usable core around project setup and workflow scaffolding, but it does not yet close the full research loop.

### Implemented or Partially Implemented

- `cbma init`
- `cbma doctor`
- `cbma validate`
- `cbma split create`
- `cbma baseline run`
- `cbma train sweep`

What this means in practice:

- local project workspaces can be initialized
- structured input files can be validated
- split artifacts can be generated
- baseline runs can be prepared
- scaling-oriented training sweeps can be prepared

At present, the most stable part of the toolkit is the CLI workflow scaffolding and local run metadata organization.

## Frozen or Non-Core Modules

The following components are not part of the active CBMA V1 core:

### Frozen

- `project_release/workflow/src/cbma/ui_api/`

The local UI is preserved as an experiment, but it is frozen and not part of the maintained V1 research-toolkit core.

### Archived Research Assets

- `project_release/qwen3/`
- `project_release/qwen2/src/internvl/`

These remain useful as research records and reference assets, but they are outside the active V1 core narrative.

### Reproducibility Appendix

- `project_release/qwen2/splits/`
- `project_release/qwen2/env_record/`
- `project_release/qwen2/artifacts/`

These are retained for reproducibility and historical traceability rather than ongoing core development.

## What Is Not Yet Complete

Three steps remain necessary before CBMA V1 can claim a closed workflow from scaling to formal reporting:

### 1. Recommend-N

The toolkit still needs a formal mechanism for converting scaling outputs into a reproducible training-size decision.

### 2. Eval

The toolkit still needs a standardized evaluation stage that runs against a locked test split and produces stable evaluation outputs.

### 3. Report

The toolkit still needs a stable reporting layer that exports:

- metrics
- confusion summaries
- error cases
- human-readable reports

## Development Implication

The immediate implication of this V1 definition is straightforward:

- continue strengthening the CLI workflow core
- do not expand UI scope
- do not prioritize platformization
- do not treat archived research lines as active product surface

In other words, CBMA V1 should now be read as a lightweight research toolkit with explicit boundaries, not as an unfinished product platform.
