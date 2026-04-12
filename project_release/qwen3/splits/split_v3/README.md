# split_v3 数据说明

本目录存放 **split_v3** 划分下的训练/测试 CSV 与 Swift 格式 jsonl，用于 Qwen3-Omni 与 MiniCPM 的训练与评估。

---

## 文件来源与生成

- **codebook**：原始标注数据来自 `scripts/01_data_prep/codebook.xlsx`。  
- **划分脚本**：`project_release/src/shared/make_split_v3.py`。  
  - 清洗：删除指定坏样本 ID、删除 Policy（gold_label=5）类别。  
  - 固定随机种子（42），按 5 分类 stratify 划分出 **5000 条测试集** → `test_main.csv`。  
  - 剩余样本作为训练母池，生成 **嵌套增量训练集**：`train_200.csv` ～ `train_3000.csv`（步长 200）。  
  - 嵌套含义：`train_2200.csv` 的前 200 条与 `train_200.csv` 完全一致，便于做 scaling 实验。

---

## 主要文件

| 文件名 | 说明 |
|--------|------|
| `test_main.csv` | 固定测试集，5000 条。用于评估 Qwen3 / MiniCPM。 |
| `train_200.csv` … `train_3000.csv` | 嵌套训练集，对应 200～3000 条训练样本。 |
| `swift_train_2200.jsonl` | 由 `gen_swift_train_2200.py` 从 `train_2200.csv` 生成，ms-swift 多模态训练格式。 |
| `swift_train.jsonl` | 若存在，可能为旧版或其它脚本生成；可用 `fix_training_data.py` 得到 `swift_train_v2.jsonl`。 |

---

## CSV 列与格式

- **必需列**：  
  - `video_id`：视频唯一标识，与文件名（如 `{video_id}.mp4`）对应。  
  - 标签列：`gold_label` 或 `final_code`，取值为 **0～4**（5 分类）。  
- **可选列**：如 `标题`、`视频链接` 等，来自原始 codebook，评估脚本不强制依赖。  
- **编码**：UTF-8；若含中文，保存时常用 `utf-8-sig` 以便 Excel 正确打开。

---

## 类别定义（0～4）

与训练/评估脚本中的 prompt 一致：

- **0** OTHER_OR_UNCLEAR（其他）：主持人播报、纯风景、视听矛盾无法归类等。  
- **1** PERFORMATIVE（绩效展示）：真实任务执行，如站岗、救灾、抓捕。  
- **2** MORAL（道德动员）：情感与精神符号，如军民互动、艰苦特写、致敬、煽情 BGM。  
- **3** PROCEDURAL（程序规范）：权力与秩序仪式，如列队、授衔、军乐。  
- **4** TECHNICAL（技术与实战）：武力/专业度展示，如武器特写、实弹演习、战术队形。

---

## 用途小结

- **训练**：Qwen3 使用 `swift_train_2200.jsonl`（或经 fix 后的 jsonl）；MiniCPM 使用 `swift_train.jsonl` 或脚本指定路径。  
- **评估**：统一使用 `test_main.csv` 作为测试集；Qwen3 评估前需先运行 `prepare_clip16_for_eval.py` 生成 16s 片段。
