# CBMA

Codebook-driven multimodal analysis workflow for reproducible research.  
面向可复现研究的、由 codebook 驱动的多模态分析工作流。

## Overview / 概述

CBMA is a lightweight, offline, local-first research toolkit. It is designed for researchers who need a reproducible workflow for codebook-based multimodal analysis, rather than a hosted AI product or an online service.

CBMA 是一个轻量、离线、local-first 的研究工具包。它面向需要做 codebook-based multimodal analysis 的研究者，强调可复现流程，而不是在线产品或托管式 AI 服务。

## Core Capabilities / 核心能力

- Project initialization, validation, and split generation  
  项目初始化、数据校验与划分生成
- Baseline dry-run and train sweep dry-run  
  baseline dry-run 与训练规模 sweep dry-run
- `recommend_n.json` generation through `cbma train recommend-n`  
  通过 `cbma train recommend-n` 生成 `recommend_n.json`
- Standardized evaluation through `cbma eval run`  
  通过 `cbma eval run` 执行标准化评估
- Structured report export through `cbma report build`  
  通过 `cbma report build` 导出结构化报告

## Intended Users / 适用对象

- Social science researchers  
  社会科学研究者
- Communication and media researchers  
  传播学、媒体研究者
- Multimodal researchers working with labeled video data  
  处理标注视频数据的多模态研究者

## Execution Model / 运行方式

- Offline execution  
  离线运行
- Local filesystem outputs  
  结果写入本地文件系统
- User-managed GPU environments for real inference and training, such as AutoDL  
  真实推理和训练依赖用户自管 GPU 环境，例如 AutoDL
- No cloud dependency  
  不依赖云服务
- No hosted API dependency  
  不依赖托管 API

You can do project setup, validation, dry-run, and report building without a GPU. Real model inference, real LoRA training, and standardized evaluation still require a user-provided local model path and a suitable runtime.

你可以在没有 GPU 的情况下完成项目初始化、校验、dry-run 和 report 构建。真正的模型推理、LoRA 训练和标准化评估仍然需要用户自行提供本地模型路径和合适的运行环境。

## Workflow / 工作流

1. Initialize or prepare a local project.  
   初始化或准备本地项目。
2. Validate the codebook, labels, and file paths.  
   校验 codebook、标签文件和路径。
3. Create split artifacts.  
   生成数据划分产物。
4. Run baseline and train sweep in `--dry-run` mode to verify wiring.  
   先用 `--dry-run` 检查 baseline 和 train sweep 是否接通。
5. Generate `recommend_n.json`.  
   生成 `recommend_n.json`。
6. Run standardized evaluation on the fixed test split.  
   在固定 test split 上做标准化评估。
7. Export a report from the eval directory.  
   从 eval 目录导出报告。

## Demo / 演示

The smallest release demo is located in `project_release/demo/`.

最小可运行 demo 位于 `project_release/demo/`。

It includes:

- a synthetic demo project  
  一个合成的 demo 项目
- fake labels and placeholder video files  
  虚拟标签与占位视频文件
- a synthetic train-sweep sample and eval sample  
  合成的 train-sweep sample 与 eval sample
- a no-GPU walkthrough  
  一条不依赖 GPU 的演示路径

Start with `project_release/demo/README.md`.  
入口请先看 `project_release/demo/README.md`。

## No-GPU Quickstart / 无 GPU 快速开始

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

This path does not run a real model. It validates the project structure, produces split artifacts, verifies dry-run metadata, regenerates `recommend_n.json`, and builds a report from synthetic eval inputs.

这条路径不会运行真实模型。它会校验项目结构、生成 split、验证 dry-run 元数据、重新生成 `recommend_n.json`，并基于合成 eval 输入生成报告。

## Full Workflow / 完整流程

```bash
cbma init my_project
cbma validate --project my_project
cbma split create --project my_project
cbma baseline run --project my_project --dry-run
cbma train sweep --project my_project --dry-run
cbma train recommend-n --project my_project --run-dir <runs_dir>/train-sweep-...
cbma eval run --project my_project --run-dir <runs_dir>/train-sweep-...
cbma report build --project my_project
```

## Output Structure / 输出结构

`cbma report build` writes a standardized report directory under an eval run:

`cbma report build` 会在某个 eval run 下写出标准化报告目录：

```text
<runs_dir>/
  eval-.../
    eval_result.json
    eval_metadata.json
    report/
      report.md
      metrics.json
      run_summary.json
      per_class_f1.csv          # optional
      confusion_matrix.csv      # optional
      error_cases.csv           # optional
```

Example / 示例：

```markdown
# CBMA Evaluation Report

## 2. Headline Metrics
- accuracy: 0.78
- macro_f1: 0.74

## 5. Per-class Results
unavailable

## 6. Confusion Structure
- confusion matrix: unavailable
```

The `unavailable` state is expected when richer raw eval artifacts are not available.  
如果底层没有提供更丰富的 raw eval 产物，报告里出现 `unavailable` 是正常的。

## Open-Source Boundaries / 开源边界

CBMA does not provide:

- source videos  
  原始视频不随仓库提供
- platform scraping  
  不提供平台抓取
- model weights  
  不提供模型权重
- automatic model download  
  不自动下载模型
- an online service  
  不提供在线服务

Users are expected to provide:

- their own local data  
  用户自行提供本地数据
- their own local model path  
  用户自行提供本地模型路径
- their own GPU environment for real inference or training  
  用户自行提供真实推理与训练所需的 GPU 环境

## Current Limits / 当前限制

- `recommend-n` requires validation metrics already materialized in the sweep run directory  
  `recommend-n` 依赖 train-sweep run 目录中已经存在的验证指标聚合文件
- Eval provenance may be partially unavailable if upstream outputs are sparse  
  如果上游 eval 输出较少，追溯信息可能部分缺失
- `raw_eval` enhancement depends on whatever the backend eval script actually produced  
  `raw_eval` 增强分析依赖底层 eval 脚本到底产出了什么
- Archived research directories remain for context, but they are not part of the maintained V1 core  
  仓库里保留了研究归档目录作上下文参考，但它们不属于当前维护的 V1 core

## Repository Map / 仓库结构

- `project_release/workflow/`: maintained CLI core  
  当前维护的 CLI 核心
- `project_release/demo/`: synthetic release demo  
  合成 release demo
- `project_release/qwen2/`: backend release scripts plus reproducibility appendix  
  qwen2 后端脚本与复现附录
- `project_release/qwen3/`: archived research assets  
  归档研究资产
- `Qwen2/` and `Qwen3/`: legacy research directories retained for archive context  
  保留作历史归档上下文的旧研究目录
