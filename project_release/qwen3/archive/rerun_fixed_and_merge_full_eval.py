#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
import re
import subprocess
from collections import Counter

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ==================== 1. 环境变量 ====================
os.environ.pop("OMP_NUM_THREADS", None)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["FPS_MAX_FRAMES"] = "8"
os.environ["MAX_PIXELS"] = "150000"
os.environ["MAX_RATIO"] = "4"

# ==================== 2. 路径 ====================
PROJECT_ROOT = "/root/autodl-tmp/project_douyin_mm"
EVAL_DIR = f"{PROJECT_ROOT}/outputs/qwen3_omni_eval_clip16_wav"
SNAP_DIR = f"{EVAL_DIR}/snapshot_first3000"

MODEL_DIR = f"{PROJECT_ROOT}/models/qwen/Qwen3-Omni-30B-A3B-Instruct"
RUN_BASE_DIR = f"{PROJECT_ROOT}/runs/qwen3_omni_lora_aligned_v2"
TRAIN_DATA = f"{PROJECT_ROOT}/splits/split_v3/swift_train_2200.jsonl"
TEST_CSV = f"{PROJECT_ROOT}/splits/split_v3/test_main.csv"

MERGED_CLIP_DIR = f"{EVAL_DIR}/merged_clips_16s"

# 已有结果
FIRST3000_RESULT = f"{SNAP_DIR}/infer_result_first3000.jsonl"
REMAINING_RESULT = f"{EVAL_DIR}/infer_result_qwen3_remaining.jsonl"

# 修复清单（默认用你第一个脚本输出）
FIXED_RESULT_CSV = f"{EVAL_DIR}/bad_videos_rebuild_results.csv"

# 重测输出
RERUN_JSONL = f"{EVAL_DIR}/swift_eval_dataset_fixed_rerun.jsonl"
RERUN_RESULT = f"{EVAL_DIR}/infer_result_fixed_rerun.jsonl"
RERUN_PRED_CSV = f"{EVAL_DIR}/eval_qwen3_fixed_rerun_predictions.csv"

# 合并输出
FULL_RESULT_JSONL = f"{EVAL_DIR}/infer_result_qwen3_full_merged.jsonl"
FULL_PRED_CSV = f"{EVAL_DIR}/eval_qwen3_full_predictions.csv"
FULL_REPORT = f"{EVAL_DIR}/eval_qwen3_full_report.txt"
FULL_CM_CSV = f"{EVAL_DIR}/eval_qwen3_full_confusion_matrix.csv"
FULL_PRED_DIST_JSON = f"{EVAL_DIR}/eval_qwen3_full_pred_distribution.json"
FULL_PARSE_DIST_JSON = f"{EVAL_DIR}/eval_qwen3_full_parse_status_distribution.json"

TMP_DIR = f"{EVAL_DIR}/tmp_fixed_rerun"
os.makedirs(TMP_DIR, exist_ok=True)

CLIP_SECONDS = 16
MAX_LENGTH = 8192
MAX_NEW_TOKENS = 32


# ==================== 3. 工具函数 ====================
def get_latest_checkpoint(base_path: str):
    checkpoints = glob.glob(os.path.join(base_path, "v*", "checkpoint-*"))
    checkpoints = [p for p in checkpoints if os.path.isdir(p)]
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)


def load_train_prompt(train_jsonl_path: str):
    with open(train_jsonl_path, "r", encoding="utf-8") as f:
        first = json.loads(f.readline())
    system_prompt = first.get("system", "").strip()
    query_prompt = first.get("query", "").strip()
    return system_prompt, query_prompt


def extract_label(response: str):
    if response is None:
        return None, "empty_response"

    text = str(response).strip()
    if not text:
        return None, "empty_response"

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

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

    m = re.search(r'["\']?label["\']?\s*[:：=]\s*["\']?([0-4])["\']?', text, re.IGNORECASE)
    if m:
        return int(m.group(1)), "regex_label"

    nums = re.findall(r'(?<!\d)([0-4])(?!\d)', text)
    if nums:
        return int(nums[-1]), "regex_digit"

    return None, "parse_failed"


def load_test_labels():
    df = pd.read_csv(TEST_CSV)
    df["video_id"] = df["video_id"].astype(str).str.strip()
    label_col = "gold_label" if "gold_label" in df.columns else "final_code"
    df["gold_label"] = df[label_col].astype(int)
    return df[["video_id", "gold_label"]]


def read_result_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def result_rows_to_pred_df(result_rows, vid_to_label):
    records = []
    seen = set()

    for data in result_rows:
        videos = data.get("videos", [])
        if not videos:
            continue

        clip_path = videos[0]
        base = os.path.splitext(os.path.basename(clip_path))[0]
        vid = base.replace(f"_clip{CLIP_SECONDS}s", "")

        if not vid or vid in seen or vid not in vid_to_label:
            continue

        seen.add(vid)

        response = data.get("response", "")
        pred, parse_status = extract_label(response)

        records.append({
            "video_id": vid,
            "gold_label": int(vid_to_label[vid]),
            "pred_label": pred,
            "parse_status": parse_status,
            "raw_response": response,
            "clip_path": clip_path,
        })

    return pd.DataFrame(records)


def build_rerun_dataset():
    df_fix = pd.read_csv(FIXED_RESULT_CSV)
    df_fix["video_id"] = df_fix["video_id"].astype(str).str.strip()

    # 只重测修复成功的视频
    df_fix = df_fix[df_fix["status"] == "fixed_ok"].copy()

    df_test = load_test_labels()
    df_merge = df_fix.merge(df_test, on="video_id", how="left")

    system_prompt, query_prompt = load_train_prompt(TRAIN_DATA)

    records = []
    for _, row in df_merge.iterrows():
        vid = row["video_id"]
        clip_path = os.path.join(MERGED_CLIP_DIR, f"{vid}_clip16s.mp4")

        if not os.path.exists(clip_path):
            print(f"[WARN] clip not found, skip: {vid}")
            continue
        if os.path.getsize(clip_path) <= 0:
            print(f"[WARN] empty clip, skip: {vid}")
            continue

        records.append({
            "system": system_prompt,
            "query": query_prompt,
            "videos": [clip_path],
            "video_id": vid,
            "gold_label": int(row["gold_label"]),
        })

    with open(RERUN_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[INFO] rerun dataset saved: {RERUN_JSONL}")
    print(f"[INFO] fixed_ok rerun records: {len(records)}")
    return records


def rerun_fixed_videos():
    lora_path = get_latest_checkpoint(RUN_BASE_DIR)
    if lora_path is None:
        raise RuntimeError("未找到 checkpoint")

    if os.path.exists(RERUN_RESULT):
        os.remove(RERUN_RESULT)

    cmd = [
        "swift", "infer",
        "--model", MODEL_DIR,
        "--adapters", lora_path,
        "--val_dataset", RERUN_JSONL,
        "--template", "qwen3_omni",
        "--quant_bits", "4",
        "--max_length", str(MAX_LENGTH),
        "--max_new_tokens", str(MAX_NEW_TOKENS),
        "--result_path", RERUN_RESULT,
        "--model_kwargs",
        '{"disable_talker": true, "use_audio_in_video": true, "trust_remote_code": true, "attn_implementation": "flash_attention_2"}'
    ]

    stdout_path = os.path.join(TMP_DIR, "fixed_rerun_stdout.log")
    stderr_path = os.path.join(TMP_DIR, "fixed_rerun_stderr.log")

    with open(stdout_path, "w", encoding="utf-8") as out_f, open(stderr_path, "w", encoding="utf-8") as err_f:
        proc = subprocess.run(cmd, stdout=out_f, stderr=err_f, text=True)

    print(f"[INFO] rerun returncode: {proc.returncode}")
    print(f"[INFO] stdout log: {stdout_path}")
    print(f"[INFO] stderr log: {stderr_path}")

    if proc.returncode != 0:
        raise RuntimeError("修复视频重测失败，请查看 stderr 日志")


def merge_and_eval():
    df_test = load_test_labels()
    vid_to_label = dict(zip(df_test["video_id"], df_test["gold_label"]))

    first_rows = read_result_jsonl(FIRST3000_RESULT)
    remain_rows = read_result_jsonl(REMAINING_RESULT)
    rerun_rows = read_result_jsonl(RERUN_RESULT)

    # 合并原始 jsonl，后写覆盖前写的同 video_id
    merged_map = {}

    for rows in [first_rows, remain_rows, rerun_rows]:
        for data in rows:
            videos = data.get("videos", [])
            if not videos:
                continue
            clip_path = videos[0]
            base = os.path.splitext(os.path.basename(clip_path))[0]
            vid = base.replace(f"_clip{CLIP_SECONDS}s", "")
            if vid:
                merged_map[vid] = data

    merged_rows = list(merged_map.values())

    with open(FULL_RESULT_JSONL, "w", encoding="utf-8") as f:
        for r in merged_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pred_df = result_rows_to_pred_df(merged_rows, vid_to_label)
    pred_df.to_csv(FULL_PRED_CSV, index=False, encoding="utf-8-sig")

    y_true, y_pred = [], []
    pred_counter = Counter()
    parse_status_counter = Counter()

    for _, row in pred_df.iterrows():
        parse_status_counter[row["parse_status"]] += 1
        if pd.notna(row["pred_label"]):
            y_true.append(int(row["gold_label"]))
            y_pred.append(int(row["pred_label"]))
            pred_counter[int(row["pred_label"])] += 1

    if len(y_true) == 0:
        raise RuntimeError("没有可评估样本")

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

    report_text = []
    report_text.append("=" * 80)
    report_text.append("Qwen3 全量合并评估报告")
    report_text.append("=" * 80)
    report_text.append(f"First3000 result rows: {len(first_rows)}")
    report_text.append(f"Remaining success result rows: {len(remain_rows)}")
    report_text.append(f"Fixed rerun result rows: {len(rerun_rows)}")
    report_text.append(f"Merged unique result rows: {len(pred_df)}")
    report_text.append(f"Evaluated rows: {len(y_true)}")
    report_text.append(f"Parse failed rows: {sum(pd.isna(pred_df['pred_label']))}")
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

    with open(FULL_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)

    pd.DataFrame(
        cm,
        index=[f"true_{i}" for i in [0, 1, 2, 3, 4]],
        columns=[f"pred_{i}" for i in [0, 1, 2, 3, 4]],
    ).to_csv(FULL_CM_CSV, encoding="utf-8-sig")

    with open(FULL_PRED_DIST_JSON, "w", encoding="utf-8") as f:
        json.dump(dict(pred_counter), f, ensure_ascii=False, indent=2)

    with open(FULL_PARSE_DIST_JSON, "w", encoding="utf-8") as f:
        json.dump(dict(parse_status_counter), f, ensure_ascii=False, indent=2)

    print(report_text)
    print(f"[INFO] merged result jsonl: {FULL_RESULT_JSONL}")
    print(f"[INFO] merged prediction csv: {FULL_PRED_CSV}")
    print(f"[INFO] full report: {FULL_REPORT}")


def main():
    build_rerun_dataset()
    rerun_fixed_videos()

    # 保存修复重测预测明细
    df_test = load_test_labels()
    vid_to_label = dict(zip(df_test["video_id"], df_test["gold_label"]))
    rerun_rows = read_result_jsonl(RERUN_RESULT)
    rerun_pred_df = result_rows_to_pred_df(rerun_rows, vid_to_label)
    rerun_pred_df.to_csv(RERUN_PRED_CSV, index=False, encoding="utf-8-sig")

    merge_and_eval()


if __name__ == "__main__":
    main()