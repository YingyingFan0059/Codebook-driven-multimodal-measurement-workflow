# CBMA Workflow Core

Maintained CLI core of CBMA V1.  
CBMA V1 当前维护中的 CLI 核心。

## Overview / 概述

This directory contains the main workflow layer of CBMA. It is the primary interface for project setup, validation, split generation, dry-run planning, recommendation handoff, evaluation, and report building.

本目录包含 CBMA 的主工作流层。它是当前项目初始化、数据校验、划分生成、dry-run 规划、recommendation 交接、评估和报告构建的主要入口。

The workflow layer is designed for offline execution, local filesystem outputs, and reproducible handoff between stages. The local UI is frozen and is not part of the maintained V1 core.

workflow 层强调离线执行、本地文件系统输出，以及阶段之间可追溯的交接。仓库中的本地 UI 已被冻结，不属于当前维护的 V1 core。

## Current Command Surface / 当前命令面

- `cbma init`
- `cbma doctor`
- `cbma validate`
- `cbma split create`
- `cbma baseline run`
- `cbma train sweep`
- `cbma train recommend-n`
- `cbma eval run`
- `cbma report build`

## Project Layout / 项目结构

`cbma init` creates a project with this shape:

`cbma init` 会生成如下结构：

```text
my_project/
  project.yaml
  README.md
  .gitignore
  data/
    codebook.yaml
    labels.csv
    videos/
  cache/
    frames/
    audio/
    hf/
  models/
    qwen2/
    internvl/
  splits/
  runs/
  exports/
```

The minimum required inputs are:

最低要求的输入包括：

- `project.yaml`
- `data/codebook.yaml`
- `data/labels.csv`
- local video files referenced by `labels.csv`  
  `labels.csv` 中引用的本地视频文件

## Execution Model / 运行方式

- Offline execution  
  离线运行
- Local filesystem outputs  
  结果写入本地文件系统
- Dry-run friendly by default  
  默认优先支持 dry-run
- User-managed GPU environments for real inference and training  
  真实推理和训练依赖用户自管 GPU 环境

Dry-run steps do not load models and usually do not require a GPU. Real evaluation and real training may require `torch`, `ffmpeg`, CUDA, and user-provided model paths.

dry-run 步骤不会加载模型，通常也不需要 GPU。真实评估和真实训练可能依赖 `torch`、`ffmpeg`、CUDA，以及用户自行提供的模型路径。

## Workflow / 工作流

### 1. Initialize / 初始化

Use `cbma init` to create a local project skeleton.  
使用 `cbma init` 创建本地项目骨架。

### 2. Validate / 校验

Use `cbma doctor` for environment inspection and `cbma validate` for project, codebook, labels, and file checks.  
用 `cbma doctor` 检查环境，用 `cbma validate` 检查项目结构、codebook、标签和文件路径。

### 3. Split / 划分

Use `cbma split create` to generate `train_pool.csv`, `test_main.csv`, optional `val_main.csv`, nested `train_<N>.csv`, and `split_summary.json`.  
用 `cbma split create` 生成 `train_pool.csv`、`test_main.csv`、可选的 `val_main.csv`、嵌套的 `train_<N>.csv` 以及 `split_summary.json`。

### 4. Baseline / Baseline

Use `cbma baseline run --dry-run` first. It resolves script paths, split inputs, methods, and model paths without loading a model.  
建议先运行 `cbma baseline run --dry-run`。它会解析脚本路径、split 输入、baseline 方法和模型路径，但不会真正加载模型。

### 5. Train Sweep / 训练规模 Sweep

Use `cbma train sweep --dry-run` first. It resolves candidate sizes, output locations, and backend script wiring.  
建议先运行 `cbma train sweep --dry-run`。它会解析候选训练规模、输出位置和后端脚本连接方式。

### 6. Recommend N / 推荐训练规模

Use `cbma train recommend-n` to convert a sweep run plus validation metrics into a `recommend_n.json` decision artifact.  
使用 `cbma train recommend-n` 将某次 sweep run 和其验证指标转换成 `recommend_n.json` 决策产物。

### 7. Evaluation / 评估

Use `cbma eval run` after `recommend_n.json` has been generated and the corresponding local model outputs are available.  
在 `recommend_n.json` 已生成，且对应本地模型产物可用之后，使用 `cbma eval run` 进行标准化评估。

### 8. Report / 报告

Use `cbma report build` to build a standardized report directory from an existing eval run. This step is CPU-only and does not call a model.  
使用 `cbma report build` 从现有 eval run 构建标准化报告目录。这个步骤只做结果整理，不调用模型，也不依赖 GPU。

## Output Structure / 输出结构

The workflow writes timestamped run directories under `runs/`:

workflow 会在 `runs/` 下写入带时间戳的运行目录：

```text
runs/
  baseline-YYYYMMDD-HHMMSS/
    baseline_run.json
  train-sweep-YYYYMMDD-HHMMSS/
    train_sweep.json
    recommend_n.json           # when generated
  eval-YYYYMMDD-HHMMSS/
    eval_result.json
    eval_metadata.json
    raw_eval/                  # optional
    report/
      report.md
      metrics.json
      run_summary.json
```

These directories exist for traceability first. Optional downstream files depend on what the backend scripts actually emit.

这些目录首先是为了追溯性而存在的。更多可选文件是否出现，取决于底层后端脚本实际产出了什么。

## Open-Source Boundaries / 开源边界

This directory is not:

- a hosted service backend  
  不是托管服务后端
- a multi-user platform  
  不是多用户平台
- a web-first application  
  不是 web-first 应用

`src/cbma/ui_api/` remains in the repository as a frozen experimental module and is not part of the maintained V1 core.  
`src/cbma/ui_api/` 仍然保留在仓库中，但它只是冻结的实验模块，不属于当前维护的 V1 core。

## Current Limits / 当前限制

- `recommend-n` requires validation metrics already materialized in the sweep run directory  
  `recommend-n` 依赖 train-sweep run 目录中已经存在的验证指标聚合文件
- real evaluation still depends on backend script outputs and local model artifacts  
  真实评估仍然依赖后端脚本产物和本地模型文件
- report enhancement depends on whether `raw_eval` artifacts exist  
  report 的增强分析依赖 `raw_eval` 产物是否存在
