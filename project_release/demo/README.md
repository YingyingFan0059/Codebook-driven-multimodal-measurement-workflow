# CBMA Demo

Smallest release-facing walkthrough for CBMA.  
面向发布的最小 CBMA 演示入口。

## Overview / 概述

This demo is designed to show the CBMA workflow shape without requiring a GPU, a real model, or real videos. It uses fake labels, placeholder video files, a synthetic train-sweep sample, and a synthetic eval sample.

这个 demo 的目标是在不依赖 GPU、真实模型和真实视频的前提下，展示 CBMA 的工作流结构。它使用虚拟标签、占位视频文件、合成的 train-sweep sample，以及合成的 eval sample。

## Included Files / 包含内容

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
        train_sweep.json
        scaling_curve.csv
        recommend_n.json
      eval-20260411-sample/
        eval_result.json
        eval_metadata.json
```

The included files are intentionally minimal. They exist to demonstrate workflow structure and handoff artifacts, not to reproduce a real model experiment.

这些文件是刻意保持最小化的。它们的作用是展示工作流结构和阶段之间的交接产物，而不是复现一次真实模型实验。

## Execution Model / 运行方式

- Offline execution  
  离线运行
- Local filesystem outputs  
  结果写入本地文件系统
- No GPU required for validation, dry-run, recommend-n, or report build  
  `validate`、dry-run、`recommend-n` 和 `report build` 不需要 GPU
- Synthetic inputs only  
  仅包含合成输入

This demo is meant to verify the workflow shape, not to measure real model quality.

这个 demo 的目标是验证工作流形态，而不是衡量真实模型效果。

## No-GPU Quickstart / 无 GPU 快速开始

Run the following commands from `project_release/workflow/`:

在 `project_release/workflow/` 目录下运行：

```bash
cd project_release/workflow
python -m pip install -e .

cbma validate --project ../demo/demo_project
cbma split create --project ../demo/demo_project --force
cbma baseline run --project ../demo/demo_project --dry-run
cbma train sweep --project ../demo/demo_project --dry-run
cbma train recommend-n --project ../demo/demo_project --run-dir ../demo/demo_project/runs/train-sweep-sample
cbma report build --project ../demo/demo_project
```

This path does not run a real model. It:

- validates the synthetic project  
  校验合成项目结构
- creates project-local split artifacts  
  生成项目内的 split 产物
- writes baseline dry-run metadata  
  写出 baseline dry-run 元数据
- writes train sweep dry-run metadata  
  写出 train sweep dry-run 元数据
- regenerates `recommend_n.json` from the included synthetic sweep metrics  
  基于内置的合成 sweep 指标重新生成 `recommend_n.json`
- builds a report from the included synthetic eval inputs  
  基于内置的合成 eval 输入生成报告

## Full Workflow / 完整流程

```bash
cbma init demo_project
cbma validate --project demo_project
cbma split create --project demo_project
cbma baseline run --project demo_project --dry-run
cbma train sweep --project demo_project --dry-run
cbma train recommend-n --project demo_project --run-dir <runs_dir>/train-sweep-...
cbma eval run --project demo_project --run-dir <runs_dir>/train-sweep-...
cbma report build --project demo_project
```

In the lightweight public release:

- `init`, `validate`, `split`, dry-run `baseline`, dry-run `train sweep`, `recommend-n`, and `report build` can be demonstrated locally  
  `init`、`validate`、`split`、dry-run `baseline`、dry-run `train sweep`、`recommend-n` 和 `report build` 都可以在本地演示
- real `eval run` still depends on local model artifacts and a user-managed runtime  
  真实的 `eval run` 仍然依赖本地模型产物和用户自管运行环境
- the included `eval-20260411-sample/` directory is a synthetic report input, not a real model evaluation result  
  仓库中附带的 `eval-20260411-sample/` 是一个合成的报告输入，并不是真实模型评估结果

## Output Structure / 输出结构

The included synthetic eval sample can be turned into a report directory shaped like this:

内置的合成 eval sample 最终会生成如下结构的报告目录：

```text
runs/
  eval-20260411-sample/
    eval_result.json
    eval_metadata.json
    report/
      report.md
      metrics.json
      run_summary.json
```

## Report Example / 报告示例

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

`unavailable` is normal in this demo because no richer `raw_eval` bundle is included.  
这个 demo 中出现 `unavailable` 是正常的，因为仓库没有附带更丰富的 `raw_eval` 产物。

## Open-Source Boundaries / 开源边界

This demo does not provide:

- real videos  
  不提供真实视频
- real model weights  
  不提供真实模型权重
- real experiment outputs  
  不提供真实实验结果

It provides only synthetic project inputs for release onboarding.  
它只提供用于发布演示的合成项目输入。
