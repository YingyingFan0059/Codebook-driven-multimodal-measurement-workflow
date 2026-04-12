# artifacts 说明

本目录保存已整理的历史实验产物，按模型线分开：

- `artifacts/qwen2/`
  - `runs/scaling_experiments/`：Qwen2-VL 不同训练规模的 LoRA adapter
  - `outputs/`：Qwen2-VL 的预测 CSV、评估报告和运行日志
- `artifacts/internvl/`
  - `runs/internvl_lora_5class_2200/`：InternVL2-8B 的 2200 样本 LoRA snapshot
  - `outputs/scaling_eval_repro/`：InternVL 评估结果与日志

这些文件用于结果追溯和复查，不是运行脚本的必要最小集。
