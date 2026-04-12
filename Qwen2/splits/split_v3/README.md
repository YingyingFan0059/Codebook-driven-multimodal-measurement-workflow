# split_v3 数据说明

本目录存放 Qwen2-VL 与 InternVL 共用的 `split_v3` 训练/测试划分。

---

## 文件组成

- `test_main.csv`：固定测试集，5000 条
- `train_200.csv` ~ `train_3000.csv`：嵌套训练集，步长 200
- `swift_train.jsonl` / `swift_train_v2.jsonl` / `swift_train_2200.jsonl`：不同阶段整理出的 Swift 训练数据

---

## 任务定义

这是一个 5 分类政务短视频分类任务，标签定义为：

- `0` OTHER_OR_UNCLEAR
- `1` PERFORMATIVE
- `2` MORAL
- `3` PROCEDURAL
- `4` TECHNICAL

训练与评估脚本统一依赖：

- `video_id`
- `gold_label` 或 `final_code`

---

## 说明

- 这些 split 与 `project_release/qwen3/splits/split_v3/` 使用的是同一任务定义，但脚本入口、模型和运行环境不同。
- 若正式开源，建议在仓库根文档中明确说明 `test_main.csv` 来自清洗后的 `codebook.xlsx`，以及原始视频文件不随仓库提供。
