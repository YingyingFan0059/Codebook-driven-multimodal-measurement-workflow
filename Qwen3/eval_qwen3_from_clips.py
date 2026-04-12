#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen3-Omni-30B LoRA 测试脚本（仅推理与评估）
前提：clip16 已经提前生成在 merged_clips_16s 目录下
"""

import os
import sys
import json
import glob
import time
import re
import subprocess
from collections import Counter
from typing import Optional, Tuple

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ==================== 1. 环境变量 ====================
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["FPS_MAX_FRAMES"] = "8"
os.environ["MAX_PIXELS"] = "150000"
os.environ["MAX_RATIO"] = "4"

# ==================== 2. 路径配置 ====================
PROJECT_ROOT = "/root/autodl-tmp/project_douyin_mm"
MODEL_DIR = f"{PROJECT_ROOT}/models/qwen/Qwen3-Omni-30B-A3B-Instruct"
RUN_BASE_DIR = f"{PROJECT_ROOT}/runs/qwen3_omni_lora_aligned_v2"

TRAIN_DATA = f"{PROJECT_ROOT}/splits/split_v3/swift_train_2200.jsonl"
CSV_PATH = f"{PROJECT_ROOT}/splits/split_v3/test_main.csv"

EVAL_DIR = f"{PROJECT_ROOT}/outputs/qwen3_omni_eval_clip16_wav"
MERGED_CLIP_DIR = f"{EVAL_DIR}/merged_clips_16s"

SWIFT_JSONL_PATH = f"{EVAL_DIR}/swift_eval_dataset_qwen3_clip16_wav.jsonl"
INFER_RESULT_PATH = f"{EVAL_DIR}/infer_result_qwen3_clip16_wav.jsonl"
REALTIME_LOG_PATH = f"{EVAL_DIR}/realtime_monitor.log"

os.makedirs(EVAL_DIR, exist_ok=True)

CLIP_SECONDS = 16
LORA_PATH = None
PRINT_EVERY = 50


# ==================== 3. 工具函数 ====================
def log_print(msg: str) -> None:
    print(msg, flush=True)
    with open(REALTIME_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def ensure_exists(path: str, name: str) -> None:
    if not os.path.exists(path):
        log_print(f"[ERROR] {name} 不存在: {path}")
        sys.exit(1)


def get_latest_checkpoint(base_path: str) -> Optional[str]:
    checkpoints = glob.glob(os.path.join(base_path, "v*", "checkpoint-*"))
    checkpoints = [p for p in checkpoints if os.path.isdir(p)]
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)


def load_gold_labels(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "video_id" not in df.columns:
        raise ValueError("CSV 缺少 video_id 列")
    label_col = "gold_label" if "gold_label" in df.columns else "final_code"
    df = df.copy()
    df["video_id"] = df["video_id"].astype(str).str.strip()
    df["gold_label"] = df[label_col].astype(int)
    return df[["video_id", "gold_label"]]


def load_train_prompt(train_jsonl_path: str) -> Tuple[str, str]:
    with open(train_jsonl_path, "r", encoding="utf-8") as f:
        first = json.loads(f.readline())
    system_prompt = first.get("system", "").strip()
    query_prompt = first.get("query", "").strip()
    if not system_prompt or not query_prompt:
        raise ValueError("训练集第一条样本缺少 system 或 query")
    return system_prompt, query_prompt


def build_merged_clip_index(base_dir: str) -> dict:
    path_map = {}
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(".mp4"):
                stem = os.path.splitext(f)[0]
                vid = stem.replace(f"_clip{CLIP_SECONDS}s", "")
                path_map[vid] = os.path.join(root, f)
    return path_map


def prepare_swift_dataset_from_existing_clips(
    df_gold: pd.DataFrame,
    merged_clip_paths: dict,
    train_system: str,
    train_query: str,
    jsonl_path: str,
) -> Tuple[str, int, dict]:
    valid_records = []
    stats = {
        "missing_clip": 0,
        "invalid_clip": 0,
    }

    for _, row in df_gold.iterrows():
        vid = row["video_id"]
        label = int(row["gold_label"])

        clip_path = merged_clip_paths.get(vid)
        if not clip_path or not os.path.exists(clip_path):
            stats["missing_clip"] += 1
            continue

        if os.path.getsize(clip_path) <= 0:
            stats["invalid_clip"] += 1
            continue

        record = {
            "system": train_system,
            "query": train_query,
            "videos": [clip_path],
            "video_id": vid,
            "gold_label": label,
        }
        valid_records.append(record)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in valid_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log_print(f"[INFO] Swift 推理数据集已生成: {jsonl_path}")
    log_print(f"[INFO] 有效样本数: {len(valid_records)}")
    log_print(f"[INFO] missing_clip: {stats['missing_clip']}")
    log_print(f"[INFO] invalid_clip: {stats['invalid_clip']}")

    return jsonl_path, len(valid_records), stats


def extract_label(response: str):
    if response is None:
        return None, "empty_response"

    text = str(response).strip()
    if not text:
        return None, "empty_response"

    # 去掉 markdown 代码块包裹
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    # 1) 严格 JSON
    try:
        obj = json.loads(text)

        if isinstance(obj, dict) and "label" in obj:
            val = obj["label"]
            if isinstance(val, str) and val.isdigit():
                val = int(val)
            if isinstance(val, int) and 0 <= val <= 4:
                return val, "json_label"

        if isinstance(obj, int) and 0 <= obj <= 4:
            return obj, "json_int"

    except Exception:
        pass

    # 2) 常见 label 模式
    m = re.search(r'["\']?label["\']?\s*[:：=]\s*["\']?([0-4])["\']?', text, re.IGNORECASE)
    if m:
        return int(m.group(1)), "regex_label"

    # 3) 中文/英文描述里找独立数字
    nums = re.findall(r'(?<!\d)([0-4])(?!\d)', text)
    if nums:
        return int(nums[-1]), "regex_digit"

    return None, "parse_failed"


def wait_for_result_file(result_path: str, process: subprocess.Popen, timeout_sec: int = 180):
    start = time.time()
    while time.time() - start < timeout_sec:
        if os.path.exists(result_path):
            return result_path
        if process.poll() is not None:
            break
        time.sleep(2)
    return result_path if os.path.exists(result_path) else None


def read_partial_metrics(result_path: str, vid_to_label: dict):
    y_true, y_pred = [], []
    pred_counter = Counter()
    seen_vids = set()
    total_rows = 0
    parse_failed = 0

    if not os.path.exists(result_path):
        return {
            "total_rows": 0,
            "done": 0,
            "acc": None,
            "macro_f1": None,
            "pred_counter": {},
            "parse_failed": 0,
        }

    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total_rows += 1
            try:
                data = json.loads(line)
            except Exception:
                parse_failed += 1
                continue

            response = data.get("response", "")
            videos = data.get("videos", [])
            if not videos:
                continue

            clip_path = videos[0]
            base = os.path.splitext(os.path.basename(clip_path))[0]
            vid = base.replace(f"_clip{CLIP_SECONDS}s", "")

            if not vid or vid in seen_vids or vid not in vid_to_label:
                continue

            pred_label, _ = extract_label(response)
            if pred_label is None:
                parse_failed += 1
                continue

            seen_vids.add(vid)
            y_true.append(vid_to_label[vid])
            y_pred.append(pred_label)
            pred_counter[pred_label] += 1

    if y_true:
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    else:
        acc, macro_f1 = None, None

    return {
        "total_rows": total_rows,
        "done": len(y_true),
        "acc": acc,
        "macro_f1": macro_f1,
        "pred_counter": dict(pred_counter),
        "parse_failed": parse_failed,
    }


def monitor_inference_progress(process, result_path, vid_to_label, print_every=50, poll_sec=5):
    last_bucket = -1
    while process.poll() is None:
        metrics = read_partial_metrics(result_path, vid_to_label)
        done = metrics["done"]
        if done > 0:
            bucket = done // print_every
            if bucket > last_bucket:
                last_bucket = bucket
                log_print(
                    f"[REALTIME] rows={metrics['total_rows']} | done={done} | "
                    f"acc={metrics['acc']:.4f} | macro_f1={metrics['macro_f1']:.4f} | "
                    f"parse_failed={metrics['parse_failed']} | pred_dist={metrics['pred_counter']}"
                )
        time.sleep(poll_sec)

    metrics = read_partial_metrics(result_path, vid_to_label)
    if metrics["done"] > 0 and metrics["acc"] is not None:
        log_print(
            f"[REALTIME-FINAL] rows={metrics['total_rows']} | done={metrics['done']} | "
            f"acc={metrics['acc']:.4f} | macro_f1={metrics['macro_f1']:.4f} | "
            f"parse_failed={metrics['parse_failed']} | pred_dist={metrics['pred_counter']}"
        )


# ==================== 4. 主流程 ====================
def run_swift_inference():
    if os.path.exists(REALTIME_LOG_PATH):
        os.remove(REALTIME_LOG_PATH)

    ensure_exists(PROJECT_ROOT, "PROJECT_ROOT")
    ensure_exists(MODEL_DIR, "MODEL_DIR")
    ensure_exists(RUN_BASE_DIR, "RUN_BASE_DIR")
    ensure_exists(TRAIN_DATA, "TRAIN_DATA")
    ensure_exists(CSV_PATH, "CSV_PATH")
    ensure_exists(MERGED_CLIP_DIR, "MERGED_CLIP_DIR")

    lora_path = LORA_PATH if LORA_PATH else get_latest_checkpoint(RUN_BASE_DIR)
    if lora_path is None:
        log_print(f"[ERROR] 未在 {RUN_BASE_DIR} 下找到 checkpoint-*")
        sys.exit(1)

    train_system, train_query = load_train_prompt(TRAIN_DATA)
    df_gold = load_gold_labels(CSV_PATH)
    vid_to_label = dict(zip(df_gold["video_id"], df_gold["gold_label"]))

    log_print("=" * 80)
    log_print("[INFO] Qwen3-Omni 从已生成 clip16 测试开始")
    log_print(f"[INFO] 使用 checkpoint: {lora_path}")
    log_print(f"[INFO] CLIP_SECONDS: {CLIP_SECONDS}")
    log_print(f"[INFO] MERGED_CLIP_DIR: {MERGED_CLIP_DIR}")
    log_print(f"[INFO] EVAL_DIR: {EVAL_DIR}")
    log_print("=" * 80)

    log_print("[INFO] 正在建立 merged clip 索引...")
    merged_clip_paths = build_merged_clip_index(MERGED_CLIP_DIR)
    log_print(f"[INFO] merged clip 索引数: {len(merged_clip_paths)}")

    jsonl_path, valid_count, prep_stats = prepare_swift_dataset_from_existing_clips(
        df_gold=df_gold,
        merged_clip_paths=merged_clip_paths,
        train_system=train_system,
        train_query=train_query,
        jsonl_path=SWIFT_JSONL_PATH,
    )

    if valid_count == 0:
        log_print("[ERROR] 没有有效 clip 可供推理")
        sys.exit(1)

    if os.path.exists(INFER_RESULT_PATH):
        os.remove(INFER_RESULT_PATH)

    cmd = [
        "swift", "infer",
        "--model", MODEL_DIR,
        "--adapters", lora_path,
        "--val_dataset", jsonl_path,
        "--template", "qwen3_omni",
        "--quant_bits", "4",
        "--max_length", "8192",
        "--max_new_tokens", "32",
        "--result_path", INFER_RESULT_PATH,
        "--model_kwargs",
        '{"disable_talker": true, "use_audio_in_video": true, "trust_remote_code": true, "attn_implementation": "flash_attention_2"}'
    ]

    log_print("[INFO] 即将执行命令：")
    log_print(" ".join(cmd))

    process = subprocess.Popen(cmd)

    latest_result = wait_for_result_file(INFER_RESULT_PATH, process, timeout_sec=180)
    if latest_result:
        log_print(f"[INFO] 检测到结果文件: {latest_result}")
    else:
        log_print("[WARN] 180 秒内未发现结果文件；将继续等待进程输出。")

    monitor_inference_progress(
        process=process,
        result_path=INFER_RESULT_PATH,
        vid_to_label=vid_to_label,
        print_every=PRINT_EVERY,
        poll_sec=5,
    )

    process.wait()
    if process.returncode != 0:
        log_print(f"[ERROR] Swift infer 失败，返回码: {process.returncode}")
        sys.exit(1)

    if not os.path.exists(INFER_RESULT_PATH):
        log_print("[ERROR] 推理进程已结束，但未找到结果 JSONL")
        sys.exit(1)

    records = []
    y_true, y_pred = [], []
    pred_counter = Counter()
    parse_status_counter = Counter()
    seen = set()

    with open(INFER_RESULT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            response = data.get("response", "")
            videos = data.get("videos", [])
            if not videos:
                continue

            clip_path = videos[0]
            base = os.path.splitext(os.path.basename(clip_path))[0]
            vid = base.replace(f"_clip{CLIP_SECONDS}s", "")

            if not vid or vid in seen or vid not in vid_to_label:
                continue

            pred, parse_status = extract_label(response)
            parse_status_counter[parse_status] += 1
            seen.add(vid)

            gold = vid_to_label[vid]
            records.append({
                "video_id": vid,
                "gold_label": gold,
                "pred_label": pred,
                "parse_status": parse_status,
                "raw_response": response,
                "clip_path": clip_path,
            })

            if pred is not None:
                y_true.append(gold)
                y_pred.append(pred)
                pred_counter[pred] += 1

    pred_csv_path = os.path.join(EVAL_DIR, "eval_qwen3_clip16_wav_predictions.csv")
    pd.DataFrame(records).to_csv(pred_csv_path, index=False, encoding="utf-8-sig")
    log_print(f"[INFO] 逐条预测已保存: {pred_csv_path}")

    if len(y_true) == 0:
        log_print("[ERROR] 没有任何成功解析样本，无法评估。")
        sys.exit(1)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

    report_text = []
    report_text.append("=" * 80)
    report_text.append("Qwen3-Omni 从已生成 clip16 的测试报告")
    report_text.append("=" * 80)
    report_text.append(f"Checkpoint: {lora_path}")
    report_text.append(f"Clip seconds: {CLIP_SECONDS}")
    report_text.append(f"Total test rows: {len(df_gold)}")
    report_text.append(f"Valid clip rows: {valid_count}")
    report_text.append(f"Clip stats: {prep_stats}")
    report_text.append(f"Evaluated rows: {len(y_true)}")
    report_text.append(f"Accuracy: {acc:.4f}")
    report_text.append(f"Macro-F1: {macro_f1:.4f}")
    report_text.append(f"Pred distribution: {dict(pred_counter)}")
    report_text.append(f"Parse status distribution: {dict(parse_status_counter)}")
    report_text.append("")
    report_text.append(report)
    report_text.append("")
    report_text.append("Confusion Matrix:")
    report_text.append(str(cm))
    report_text.append("=" * 80)

    report_text = "\n".join(report_text)
    log_print(report_text)

    report_path = os.path.join(EVAL_DIR, "eval_qwen3_clip16_wav_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    cm_path = os.path.join(EVAL_DIR, "eval_qwen3_clip16_wav_confusion_matrix.csv")
    pd.DataFrame(
        cm,
        index=[f"true_{i}" for i in [0, 1, 2, 3, 4]],
        columns=[f"pred_{i}" for i in [0, 1, 2, 3, 4]],
    ).to_csv(cm_path, encoding="utf-8-sig")

    pred_dist_path = os.path.join(EVAL_DIR, "eval_qwen3_clip16_wav_pred_distribution.json")
    with open(pred_dist_path, "w", encoding="utf-8") as f:
        json.dump(dict(pred_counter), f, ensure_ascii=False, indent=2)

    parse_dist_path = os.path.join(EVAL_DIR, "eval_qwen3_clip16_wav_parse_status_distribution.json")
    with open(parse_dist_path, "w", encoding="utf-8") as f:
        json.dump(dict(parse_status_counter), f, ensure_ascii=False, indent=2)

    log_print(f"[INFO] 报告已保存: {report_path}")
    log_print(f"[INFO] 混淆矩阵已保存: {cm_path}")
    log_print(f"[INFO] 预测分布已保存: {pred_dist_path}")
    log_print(f"[INFO] 解析状态分布已保存: {parse_dist_path}")


if __name__ == "__main__":
    run_swift_inference()