# Qwen3-Omni 与 MiniCPM 微调/评估差异备忘录

本目录下两条模型线（**Qwen3-Omni**、**MiniCPM-o**）共用同一套数据划分（split_v3）与任务定义（5 分类），但**训练与评估的脚本、配置、流程有区别**。本文档便于复现时快速区分，避免混用脚本或路径。

---

## 一、对比总览

| 项目 | Qwen3-Omni | MiniCPM-o |
|------|------------|-----------|
| **训练入口** | `src/qwen3/train_qwen3_omni_v3.py`（Swift `sft_main` + `SftArguments`） | `src/minicpm/train_minicpm_v3.py`（Swift `SwiftSft` + `SwiftTrainArguments`） |
| **训练数据** | `splits/split_v3/swift_train_2200.jsonl` | `splits/split_v3/swift_train.jsonl` |
| **模型目录** | `models/qwen/Qwen3-Omni-30B-A3B-Instruct` | `models/openbmb/MiniCPM-o-2_6` |
| **输出目录（训练）** | `runs/qwen3_omni_lora_aligned_v2` | `runs/minicpm_o_lora_a100` |
| **视觉/长度** | FPS_MAX_FRAMES=8，MAX_PIXELS=150000，max_length=8192 | FPS_MAX_FRAMES=64，max_length=24000 |
| **LoRA 配置** | target 为 q/k/v/o_proj，rank/alpha 对齐 Qwen2-VL | target_modules="all-linear"，另有 no_split_module_classes |
| **评估入口** | `src/qwen3/eval_qwen3_from_clips.py` | `src/minicpm/eval_minicpm_v3.py` |
| **评估前预处理** | **必须先**跑 `prepare_clip16_for_eval.py`，生成 16s 片段（merged_clips_16s） | **不需要**预裁 clip，脚本内读原视频、自己抽帧与音频 |
| **评估输出目录** | `outputs/qwen3_omni_eval_clip16_wav` | `outputs/scaling_eval` |
| **推理方式** | Swift `swift infer`，输入为已生成的 clip 路径 | 脚本内用 `transformers` + `peft` 加载基座与 LoRA，按视频路径逐条推理 |

---

## 二、训练流程差异

### Qwen3-Omni

- 数据：使用 **swift_train_2200.jsonl**（可由 `gen_swift_train_2200.py` 从 `train_2200.csv` 生成；若用旧版 swift_train，需先跑 `fix_training_data.py`）。
- 脚本：`train_qwen3_omni_v3.py` 通过 Swift 4.0.1 的 `sft_main` 启动，模板为 **qwen3_omni**。
- 环境变量：`FPS_MAX_FRAMES=8`、`MAX_PIXELS=150000`、`MAX_RATIO=4`。
- 断点续训：若需要，在脚本内或 Swift 参数中配置 `resume_from_checkpoint`。

### MiniCPM-o

- 数据：使用 **swift_train.jsonl**（注意与 Qwen3 的 2200 条 jsonl 不同；若共用同一任务，需保证格式一致）。
- 脚本：`train_minicpm_v3.py` 在文件开头有一系列**补丁**（如 `torch.load` 的 `weights_only`、Swift/transformers 的 tokenizer 只读锁、`ProcessorMixin` 等），用于兼容当前环境；再调用 `SwiftSft`。
- 环境变量：`FPS_MAX_FRAMES=64`、`MAX_RATIO=4`、`USE_AUDIO_IN_VIDEO=True`。
- 断点续训：脚本内通过 `resume_from_checkpoint` 指定 checkpoint 路径（如 `v1-20260307-115831/checkpoint-1500`）。

---

## 三、评估流程差异

### Qwen3-Omni

1. **必须先**在测试集上运行 `src/qwen3/prepare_clip16_for_eval.py`，生成 16 秒音视频片段（video_clips_16s、audio_clips_16s、**merged_clips_16s**）及状态 CSV。
2. 再运行 `src/qwen3/eval_qwen3_from_clips.py`：从 **merged_clips_16s** 构建 Swift 推理用 jsonl，调用 `swift infer`，产出预测与指标。
3. 结果与日志在 **outputs/qwen3_omni_eval_clip16_wav** 下。

### MiniCPM-o

1. **无需**单独 clip 预处理；评估脚本直接根据 `test_main.csv` 与视频路径索引找到原视频。
2. 运行 `src/minicpm/eval_minicpm_v3.py`：脚本内用 **moviepy + librosa** 对每个视频抽帧、抽音频，构造多模态输入，再加载基座 + LoRA 做推理。
3. 结果写入 **outputs/scaling_eval**（如 `eval_minicpm_o_lora_predictions.csv`），支持断点续评（已完成的 video_id 会跳过）。

---

## 四、使用时的注意点

- **不要混用训练数据路径**：Qwen3 用 `swift_train_2200.jsonl`，MiniCPM 用 `swift_train.jsonl`；若你希望两边用同一批样本，需保证两份 jsonl 格式与路径一致，并分别放到脚本里配置的路径。
- **Qwen3 评估前必做 clip 生成**：未跑 `prepare_clip16_for_eval.py` 就运行 `eval_qwen3_from_clips.py` 会因缺少 merged_clips_16s 而失败。
- **MiniCPM 训练脚本依赖补丁**：若升级 Swift/transformers/PyTorch，若出现断点续传或 tokenizer 报错，需对照 `train_minicpm_v3.py` 开头的补丁逻辑做适配。
- **输出目录不同**：训练与评估的 checkpoint、预测结果、日志分别落在上述不同目录，复现或对比时注意区分。

---

*与 qwen3/README.md、各训练/评估脚本内配置保持一致；若脚本有改动请同步更新本备忘录。*
