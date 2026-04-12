# 项目整理进度备忘录（Qwen3 + MiniCPM 开源最小化）

> 目标：对 `project_douyin_mm` 进行代码资产梳理、目录重构与开源最小化清理，**仅处理 Qwen3 与 MiniCPM**，不包含 Qwen2。

---

## 一、总任务目标（回顾）

1. **项目扫描与分类**：按 training / evaluation / preprocessing / prompts / utils / configs / archive 等分类。
2. **识别开源核心代码**：筛出 Qwen3 与 MiniCPM 的训练、评估、预处理、配置/工具等核心文件。
3. **识别冗余/不宜开源**：临时脚本、调试脚本、废弃脚本、不应开源的路径/账号/缓存等。
4. **建议目录结构**：按 Part E 的 `project_release/` 结构重组。
5. **执行整理**：创建 `archive/`、移动归档候选、复制/重组到 `project_release/`、写文档。

**边界**：不运行代码、不改研究设计、不立即删文件（优先归档）、不做大规模重写。

---

## 二、已完成

### 2.1 扫描与分类（已完成）

- 已扫描 `Qwen3/` 下全部 .py / .sh，并完成 Part A～F 的梳理报告。
- 核心文件、归档候选、不宜开源文件均已列出（见首次报告）。

### 2.2 目录结构（已采纳）

- 采纳 Part E 的 `project_release/` 结构。
- 已存在或已创建的目录：`project_release/src/qwen3/`、`project_release/src/minicpm/`、`project_release/src/shared/`、`project_release/scripts/`、`project_release/configs/`、`project_release/prompts/`、`project_release/splits/split_v3/`、`project_release/archive/`（若尚未创建，下次补建）。

### 2.3 已写入 project_release 的核心文件

| 目标路径 | 状态 |
|----------|------|
| `project_release/src/qwen3/train_qwen3_omni_v3.py` | ✅ 已写入 |
| `project_release/src/qwen3/eval_qwen3_from_clips.py` | ✅ 已写入 |

---

## 三、已完成（本次续做）

### 3.1 已写入 project_release 的核心文件 ✅

| 目标路径 | 源路径（在 Qwen3/ 下） | 说明 |
|----------|------------------------|------|
| `project_release/src/qwen3/prepare_clip16_for_eval.py` | （此前已写入） | 评估前预处理：生成 16s 音视频片段 + merged_clips |
| `project_release/src/minicpm/train_minicpm_v3.py` | （此前已存在） | MiniCPM LoRA 训练主脚本 |
| `project_release/src/minicpm/eval_minicpm_v3.py` | `Qwen3/eval_minicpm_v3.py` | MiniCPM 评估脚本（含断点续评） |
| `project_release/src/shared/make_split_v3.py` | `Qwen3/make_split_v3.py` | 数据划分：codebook → train_*.csv / test_main.csv |
| `project_release/src/shared/gen_swift_train_2200.py` | `Qwen3/gen_swift_train_2200.py` | 从 train_2200.csv 生成 swift_train_2200.jsonl |
| `project_release/src/shared/fix_training_data.py` | `Qwen3/fix_training_data.py` | 修正 swift 训练数据 prompt/response 格式 |
| `project_release/scripts/run_qwen3_train.sh` | `Qwen3/run_qwen3_train.sh` | Qwen3 训练启动脚本（swift sft） |

### 3.2 归档：已复制到 project_release/archive/ ✅

以下 9 个文件已**复制**到 `project_release/archive/`（Qwen3/ 下原文件未删除，可按需手动删除）：

| 源文件（Qwen3/） | 说明 |
|------------------|------|
| `debug_len_scan_all.py` | 按片跑 swift infer 定位坏样本的调试脚本 |
| `test_qwen3_omni_clip16.py` | 先裁 clip 再推理的测试方案（与正式 eval_qwen3_from_clips 重叠） |
| `qwen3_eval_remaining.py` | 剩余集分块推理、坏样本跳过 |
| `rerun_fixed_and_merge_full_eval.py` | 坏样本修复后重跑与全量结果合并 |
| `1_collect_bad_videos_for_rebuild.py` | 收集需重建的 bad_videos 列表 |
| `clean_broken_eval.py` | 从日志救回 CSV、清理坏样本行 |
| `run_qwen3_omni_local.py` | 本地 8bit 端到端推理（非 Swift 主流程） |
| `prepare_dataset_2200.py` | 从 scaling_eval CSV 生成 swift_train.jsonl 的一次性脚本 |
| `eval_zero_shot_baseline.py` | Qwen3 zero-shot 基线 |

### 3.3 文档 ✅

| 文档 | 状态 |
|------|------|
| `project_release/README.md` | ✅ 已写：数据准备、Qwen3/MiniCPM 训练与评估、run_qwen3_train.sh 与 train_qwen3_omni_v3.py 差异、路径/环境、archive 说明；不包含 Qwen2。 |
| `project_release/splits/split_v3/README.md` | ✅ 已写：train_*.csv / test_main.csv 列与格式、split_v3 用途与类别定义。 |
| `project_release/requirements.txt` | ✅ 已写：从 requirements_qwen3_a800.txt 提炼，已去掉 `@ file:///...` 本地路径。 |

### 3.4 可选收尾（未做）

- 脱敏：当前阶段「暂时不需要脱敏」，若后续要开源，再统一将脚本中的 `PROJECT_ROOT`、模型路径等改为环境变量或 config。
- 若需从 Qwen3/ 删除已归档的 9 个脚本，可手动执行删除。

---

## 四、文件对应关系速查

### 核心代码（保留在 project_release 主结构）

- **Qwen3**：`train_qwen3_omni_v3.py`、`eval_qwen3_from_clips.py`、`prepare_clip16_for_eval.py`（← test_prepare_clip16_only.py）
- **MiniCPM**：`train_minicpm_v3.py`、`eval_minicpm_v3.py`
- **共享**：`make_split_v3.py`、`gen_swift_train_2200.py`、`fix_training_data.py`
- **脚本**：`run_qwen3_train.sh`

### 归档（移至 project_release/archive/）

- 见上方「3.2 归档」表格中的 9 个 .py 文件。

### 不处理 / 后续单独整理

- **Qwen2** 相关：`eval_qwen2_2200.py`、`train_scaling_qwen.py`、`eval_scaling_qwen.py`、`eval_baselines.py` 等，以及对比/绘图脚本中的 Qwen2 部分——仅标记「后续单独整理」，不放入本次 project_release 主流程。

---

## 五、自检建议

- 确认 `project_release/` 下核心脚本、shared、scripts、archive、splits 结构完整。
- 确认 `archive/` 仅含归档脚本，主流程以 README 为准。
- 开源前：将 `PROJECT_ROOT`、模型路径等改为环境变量或配置文件。

---

*最后更新：核心文件已全部写入；9 个归档脚本已复制至 archive/；README、splits/split_v3/README、requirements.txt 已完成。Qwen3 下原文件未删除，可按需手动清理。*
