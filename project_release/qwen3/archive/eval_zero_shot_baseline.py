#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
 Qwen3-Omni-30B  Zero-Shot 基线测试脚本
==========================================================================
 目的: 用原始模型(不加载LoRA)测试zero-shot分类准确率
 Prompt: 视觉-意图映射手册 (计算社会科学编码标准) + 音频辅助
==========================================================================
"""

import os
import sys
import json
import pandas as pd
import glob
import re
import time
import subprocess
import shutil
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter

# ==================== 1. 环境变量 ====================
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["FPS_MAX_FRAMES"] = "16"
os.environ["MAX_PIXELS"] = "230400"
os.environ["MAX_RATIO"] = "4"

# ==================== 2. 路径配置 ====================
PROJECT_ROOT = "/root/autodl-tmp/project_douyin_mm"
MODEL_DIR = f"{PROJECT_ROOT}/models/qwen/Qwen3-Omni-30B-A3B-Instruct"
CSV_PATH = f"{PROJECT_ROOT}/splits/split_v3/test_main.csv"
VIDEO_BASE_DIR = f"{PROJECT_ROOT}/videos/douyin/upload_pack"

EVAL_DIR = f"{PROJECT_ROOT}/outputs/zero_shot_baseline"
SWIFT_JSONL_PATH = f"{EVAL_DIR}/swift_eval_dataset.jsonl"
os.makedirs(EVAL_DIR, exist_ok=True)

# ==================== 3. Prompt 定义 ====================
# 基于用户原始编码手册，适配 Qwen3-Omni 的视听融合能力
SYSTEM_PROMPT = """你是一位受过严格训练的计算社会科学机器编码员。我们正在对政务短视频进行意图编码。请同时观看视频画面并聆听音频内容（包括旁白、对话、背景音乐），基于以下【视听-意图映射手册】，推断视频的唯一类别。
【视听-意图映射手册】
请按照以下优先级评估画面与音频特征：
- 类别 4 TECHNICAL (技术与实战)：寻找[纯粹的武力/专业度展示]。视觉特征：密集的武器装备特写、实弹演习爆炸画面、复杂的战术队形跑位、不涉及民众的体能拉练。音频特征：枪炮声、引擎轰鸣、指挥口令、无煽情旁白。
- 类别 1 PERFORMATIVE (绩效展示)：寻找[真实社会任务的执行]。视觉特征：军警在边防哨所站岗的静态画面、运送救灾物资的动态过程、实际执行抓捕或抢险的现场记录（重在"在做事"）。音频特征：现场环境音、任务对讲、新闻播报式解说。
- 类别 2 MORAL (道德动员)：寻找[情感与精神的视觉符号]。视觉特征：军人与民众的双向互动（如拥抱、送别）、艰苦环境下的特写（如汗水、冻伤的脸）、致敬烈士的画面、温情的节日元素。音频特征：煽情BGM、感人旁白、哽咽或致敬的语调。
- 类别 3 PROCEDURAL (程序规范)：寻找[权力与秩序的仪式]。视觉特征：排列整齐的会议与列队、授衔/授枪仪式中的标准动作、强调严肃纪律的视觉构图。音频特征：庄严的军乐、正式的宣誓词、仪式性口令。
- 类别 0 OTHER_OR_UNCLEAR (其他)：画面仅为主持人播报新闻、纯风景混剪、或者视听线索相互矛盾无法归入上述四类。
【判别边界提示】
- 如果画面是"刻苦训练"，若是为了展示战斗力归属 4；若特写人物咬牙坚持的痛苦与毅力，侧重情感动员，归属 2。
- 如果画面是"巡逻/站岗"，这属于日常核心职责的达成，优先归属 1，而非 3 或 4。
- 注意识别BGM滥用：如果画面是技术训练(类别4)但配了煽情音乐，以画面内容为准，仍归属 4。
输出要求：仅返回JSON对象，例如 {"label": 4}。严禁输出Markdown格式符、多余空格或解释性文本。"""

USER_PROMPT = "请根据视频画面和音频内容，按照【视听-意图映射手册】进行分类，仅返回JSON对象。"

# ==================== 4. 数据准备 ====================
def build_video_path_index(base_dir):
    path_map = {}
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(('.mp4', '.avi', '.mov')):
                path_map[os.path.splitext(f)[0]] = os.path.join(root, f)
    return path_map

GLOBAL_VIDEO_PATHS = build_video_path_index(VIDEO_BASE_DIR)
PATH_TO_VID = {v: k for k, v in GLOBAL_VIDEO_PATHS.items()}

df_original = pd.read_csv(CSV_PATH)
VID_TO_LABEL = {}
for idx, row in df_original.iterrows():
    vid = str(row.get('video_id', '')).strip()
    label = int(row['gold_label'] if 'gold_label' in df_original.columns else row['final_code'])
    VID_TO_LABEL[vid] = label

def prepare_swift_dataset():
    print("[INFO] 正在构建 Swift 推理数据集...")

    valid_records = []
    for vid, label in VID_TO_LABEL.items():
        video_path = GLOBAL_VIDEO_PATHS.get(vid)
        if video_path and os.path.exists(video_path):
            record = {
                "system": SYSTEM_PROMPT,
                "messages": [
                    {
                        "role": "user",
                        "content": f"<video>{video_path}</video>{USER_PROMPT}"
                    }
                ],
                "video_id": vid,
                "gold_label": label
            }
            valid_records.append(record)

    with open(SWIFT_JSONL_PATH, "w", encoding="utf-8") as f:
        for r in valid_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[INFO] 数据集: {len(valid_records)} 条有效视频")
    label_counts = Counter(VID_TO_LABEL[vid] for vid in VID_TO_LABEL
                          if GLOBAL_VIDEO_PATHS.get(vid) and os.path.exists(GLOBAL_VIDEO_PATHS.get(vid, '')))
    print("[INFO] 标签分布:")
    for label in sorted(label_counts.keys()):
        print(f"  类别 {label}: {label_counts[label]} 条")

    return SWIFT_JSONL_PATH


# ==================== 5. 解析模型回复 ====================
def parse_response(response):
    """
    解析模型回复，支持多种格式:
      - {"label": 4}       标准JSON
      - {"label":4}        无空格JSON
      - 纯数字: 4          直接输出
      - 文本中包含数字      回退提取
    返回 0-4 的整数，无法解析返回 -1
    """
    response = response.strip()

    # 1. 尝试直接解析完整 JSON
    try:
        obj = json.loads(response)
        if isinstance(obj, dict) and "label" in obj:
            val = int(obj["label"])
            if 0 <= val <= 4:
                return val
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # 2. 从回复中提取 JSON 片段
    json_match = re.search(r'\{[^}]*"label"\s*:\s*(\d)[^}]*\}', response)
    if json_match:
        val = int(json_match.group(1))
        if 0 <= val <= 4:
            return val

    # 3. 回退: 提取第一个 0-4 数字
    digits = re.findall(r'[0-4]', response)
    if digits:
        return int(digits[0])

    return -1


# ==================== 6. 运行推理 ====================
def run_swift_inference():
    jsonl_path = prepare_swift_dataset()

    # 清空旧结果
    infer_result_dir = os.path.join(MODEL_DIR, "infer_result")
    if os.path.exists(infer_result_dir):
        shutil.rmtree(infer_result_dir)
        print("[INFO] 已清空旧推理结果")

    # 核心: 不加载 LoRA, 使用正确的 qwen3_omni 模板
    cmd = [
        "swift", "infer",
        "--model", MODEL_DIR,
        "--template", "qwen3_omni",
        "--val_dataset", jsonl_path,
        "--quant_bits", "4",
        "--max_length", "8192",
        "--max_new_tokens", "32",
        "--model_kwargs", '{"disable_talker": true, "use_audio_in_video": true, "trust_remote_code": true}'
        # 注意: 没有 --adapters, 不加载任何 LoRA
    ]

    print("\n" + "=" * 60)
    print("  Zero-Shot 基线测试")
    print("  模型: Qwen3-Omni-30B-A3B-Instruct (原始, 无LoRA)")
    print("  模板: qwen3_omni")
    print("  Prompt: 视听-意图映射手册")
    print("  目的: 测试模型裸能力, 建立基线")
    print("=" * 60)
    print(f"\n[CMD] {' '.join(cmd)}\n")

    process = subprocess.Popen(cmd)

    result_pattern = os.path.join(infer_result_dir, "*.jsonl")
    latest_result = None
    print("[INFO] 等待推理引擎启动 (约需1-3分钟)...")

    while process.poll() is None:
        result_files = glob.glob(result_pattern)
        if result_files:
            latest_result = max(result_files, key=os.path.getctime)
            break
        time.sleep(2)

    # 实时监控
    seen_vids = set()
    y_true_rt, y_pred_rt = [], []

    if latest_result:
        print(f"[INFO] 输出文件: {latest_result}")
        print("[INFO] 启动实时监控 (每10条打印一次)...\n")
        with open(latest_result, 'r', encoding='utf-8') as f:
            while process.poll() is None:
                line = f.readline()
                if not line:
                    time.sleep(1)
                    continue

                try:
                    data = json.loads(line)
                    response = data.get('response', '')

                    vid = str(data.get('video_id', '')).strip()
                    if not vid:
                        path_match = re.search(r'<video>(.*?)</video>', line)
                        if path_match:
                            vid = str(PATH_TO_VID.get(path_match.group(1), ''))

                    if not vid or vid in seen_vids:
                        continue
                    seen_vids.add(vid)

                    gold_label = VID_TO_LABEL.get(vid)
                    if gold_label is None:
                        continue

                    pred = parse_response(response)
                    if pred == -1:
                        continue

                    y_true_rt.append(gold_label)
                    y_pred_rt.append(pred)

                    if len(y_true_rt) % 10 == 0:
                        acc = accuracy_score(y_true_rt, y_pred_rt)
                        print(f"  [实时] 完成: {len(y_true_rt)} | Acc: {acc:.4f}", flush=True)
                except Exception:
                    continue

    process.wait()
    if process.returncode != 0:
        print("\n[WARN] Swift 推理进程异常退出，尝试解析已有结果...")

    # ==================== 7. 最终结算 ====================
    if latest_result and os.path.exists(latest_result):
        y_true_final, y_pred_final = [], []
        records_to_save = []
        raw_responses = []

        with open(latest_result, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    response = data.get('response', '')

                    vid = str(data.get('video_id', '')).strip()
                    if not vid:
                        path_match = re.search(r'<video>(.*?)</video>', line)
                        if path_match:
                            vid = str(PATH_TO_VID.get(path_match.group(1), ''))

                    gold_label = VID_TO_LABEL.get(vid)
                    if gold_label is None:
                        continue

                    pred = parse_response(response)

                    raw_responses.append({
                        "video_id": vid,
                        "gold_label": gold_label,
                        "raw_response": response,
                        "parsed_pred": pred
                    })

                    if pred == -1:
                        continue

                    y_true_final.append(gold_label)
                    y_pred_final.append(pred)
                    records_to_save.append({
                        "video_id": vid,
                        "gold_label": gold_label,
                        "pred_label": pred,
                        "raw_response": response
                    })
                except Exception:
                    pass

        if len(y_true_final) > 0:
            # 保存预测结果
            csv_path = os.path.join(EVAL_DIR, "zero_shot_predictions.csv")
            pd.DataFrame(records_to_save).to_csv(csv_path, index=False, encoding="utf-8-sig")

            # 保存原始回复
            raw_path = os.path.join(EVAL_DIR, "zero_shot_raw_responses.jsonl")
            with open(raw_path, "w", encoding="utf-8") as f:
                for r in raw_responses:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            # 计算指标
            acc = accuracy_score(y_true_final, y_pred_final)
            report = classification_report(y_true_final, y_pred_final, digits=4, zero_division=0)
            cm = confusion_matrix(y_true_final, y_pred_final, labels=[0, 1, 2, 3, 4])

            pred_dist = Counter(y_pred_final)
            true_dist = Counter(y_true_final)
            unparsed = sum(1 for r in raw_responses if r["parsed_pred"] == -1)

            report_str = "=" * 60 + "\n"
            report_str += "  Zero-Shot 基线测试报告\n"
            report_str += "  模型: Qwen3-Omni-30B (原始, 无LoRA)\n"
            report_str += "  模板: qwen3_omni\n"
            report_str += "  Prompt: 视听-意图映射手册\n"
            report_str += "=" * 60 + "\n\n"

            report_str += f"总样本数: {len(raw_responses)}\n"
            report_str += f"有效预测: {len(y_true_final)}\n"
            report_str += f"无法解析: {unparsed}\n\n"

            report_str += f"Accuracy: {acc:.4f}\n\n"

            report_str += "真实标签分布:\n"
            for label in sorted(true_dist.keys()):
                report_str += f"  类别 {label}: {true_dist[label]}\n"

            report_str += "\n模型预测分布:\n"
            for label in sorted(pred_dist.keys()):
                report_str += f"  类别 {label}: {pred_dist[label]}\n"

            report_str += f"\n分类报告:\n{report}\n"

            report_str += "混淆矩阵 (行=真实, 列=预测):\n"
            report_str += "       " + "  ".join([f"P{i:d}" for i in range(5)]) + "\n"
            for i in range(5):
                row = "  ".join([f"{cm[i][j]:4d}" for j in range(5)])
                report_str += f"  T{i}   {row}\n"

            report_str += "\n" + "=" * 60 + "\n"
            report_str += "解读指南:\n"
            report_str += "  Acc > 50%: 模型有能力, 正确模板微调后应能提升到 70%+\n"
            report_str += "  Acc 30-50%: 需优化prompt/简化分类/增加few-shot\n"
            report_str += "  Acc < 30%: 任务对视频模型很难, 需重新审视方案\n"
            report_str += "  只预测少数类别: prompt需改进或任务边界模糊\n"
            report_str += "=" * 60 + "\n"

            print(report_str)

            txt_path = os.path.join(EVAL_DIR, "zero_shot_report.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(report_str)

            print(f"[SAVED] 预测CSV: {csv_path}")
            print(f"[SAVED] 原始回复: {raw_path}")
            print(f"[SAVED] 评测报告: {txt_path}")

            # 打印样例
            print("\n" + "=" * 60)
            print("  前15条预测样例")
            print("=" * 60)
            for r in records_to_save[:15]:
                mark = "O" if r["gold_label"] == r["pred_label"] else "X"
                resp_preview = r["raw_response"][:60].replace("\n", " ")
                print(f"  [{mark}] vid={r['video_id'][:20]:20s} 真实={r['gold_label']} 预测={r['pred_label']} | {resp_preview}")

        else:
            print("\n[ERROR] 未提取到有效预测结果")
            if raw_responses:
                print("[DEBUG] 前5条原始回复:")
                for r in raw_responses[:5]:
                    print(f"  vid={r['video_id']}, response='{r['raw_response'][:100]}'")
    else:
        print("\n[ERROR] 未找到推理输出文件")


if __name__ == "__main__":
    run_swift_inference()