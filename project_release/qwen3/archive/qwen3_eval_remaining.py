#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen3-Omni-30B 剩余测试集分块推理脚本（带可见日志版）
特点：
1. 按块运行，比逐条启动快
2. 某块失败时自动二分，直到定位到具体坏样本
3. 坏样本自动跳过并记录
4. 终端可看到 chunk 启动/结束日志，不会像“卡住”
"""

import os
import sys
import json
import glob
import time
import re
import shutil
import subprocess
from collections import Counter
from typing import Optional, Tuple, List, Dict

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

# ==================== 2. 路径配置 ====================
PROJECT_ROOT = "/root/autodl-tmp/project_douyin_mm"
MODEL_DIR = f"{PROJECT_ROOT}/models/qwen/Qwen3-Omni-30B-A3B-Instruct"
RUN_BASE_DIR = f"{PROJECT_ROOT}/runs/qwen3_omni_lora_aligned_v2"
TRAIN_DATA = f"{PROJECT_ROOT}/splits/split_v3/swift_train_2200.jsonl"

EVAL_DIR = f"{PROJECT_ROOT}/outputs/qwen3_omni_eval_clip16_wav"
MERGED_CLIP_DIR = f"{EVAL_DIR}/merged_clips_16s"
CSV_PATH = f"{EVAL_DIR}/snapshot_first3000/test_remaining_after3000.csv"

SWIFT_JSONL_PATH = f"{EVAL_DIR}/swift_eval_dataset_qwen3_remaining.jsonl"
INFER_RESULT_PATH = f"{EVAL_DIR}/infer_result_qwen3_remaining.jsonl"
REALTIME_LOG_PATH = f"{EVAL_DIR}/realtime_monitor_remaining.log"

PRED_CSV_PATH = os.path.join(EVAL_DIR, "eval_qwen3_remaining_predictions.csv")
REPORT_PATH = os.path.join(EVAL_DIR, "eval_qwen3_remaining_report.txt")
CM_PATH = os.path.join(EVAL_DIR, "eval_qwen3_remaining_confusion_matrix.csv")
PRED_DIST_PATH = os.path.join(EVAL_DIR, "eval_qwen3_remaining_pred_distribution.json")
PARSE_DIST_PATH = os.path.join(EVAL_DIR, "eval_qwen3_remaining_parse_status_distribution.json")
BAD_CASES_JSON = os.path.join(EVAL_DIR, "qwen3_remaining_bad_cases.json")
BAD_CASES_CSV = os.path.join(EVAL_DIR, "qwen3_remaining_bad_cases.csv")

TMP_DIR = os.path.join(EVAL_DIR, "tmp_chunk_infer")

os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# ==================== 3. 参数 ====================
CLIP_SECONDS = 16
LORA_PATH = None
MAX_LENGTH = 8192
MAX_NEW_TOKENS = 32
CHUNK_SIZE = 32
PRINT_EVERY = 50
CLEAN_TMP_ON_START = False

# 调试开关：先只跑前 N 条可快速验证。正式跑时设为 None
DEBUG_LIMIT = None
# DEBUG_LIMIT = 32


# ==================== 4. 工具函数 ====================
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
):
    valid_records = []
    stats = {"missing_clip": 0, "invalid_clip": 0}

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

        valid_records.append({
            "system": train_system,
            "query": train_query,
            "videos": [clip_path],
            "video_id": vid,
            "gold_label": label,
        })

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in valid_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log_print(f"[INFO] Swift 推理数据集已生成: {jsonl_path}")
    log_print(f"[INFO] 有效样本数: {len(valid_records)}")
    log_print(f"[INFO] missing_clip: {stats['missing_clip']}")
    log_print(f"[INFO] invalid_clip: {stats['invalid_clip']}")

    return valid_records, stats


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


def classify_error(stderr_text: str, stdout_text: str = "") -> str:
    text = (stderr_text or "") + "\n" + (stdout_text or "")
    if "MaxLengthError" in text:
        return "max_length_error"
    if "decord._ffi.base.DECORDError" in text:
        return "decord_error"
    if "KeyError: 'video_fps'" in text:
        return "video_fps_error"
    if "Error while feeding the filter graph" in text:
        return "ffmpeg_filter_graph_error"
    if "CUDA out of memory" in text:
        return "cuda_oom"
    return "other_error"


def append_jsonl_lines(src_path: str, dst_path: str) -> int:
    count = 0
    with open(src_path, "r", encoding="utf-8") as src, open(dst_path, "a", encoding="utf-8") as dst:
        for line in src:
            if line.strip():
                dst.write(line)
                count += 1
    return count


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def chunk_iter(lst: List[dict], size: int):
    for i in range(0, len(lst), size):
        yield i, lst[i:i + size]


def run_chunk(records: List[dict], tag: str, lora_path: str) -> Dict:
    jsonl_path = os.path.join(TMP_DIR, f"{tag}.jsonl")
    result_path = os.path.join(TMP_DIR, f"{tag}_result.jsonl")
    stdout_path = os.path.join(TMP_DIR, f"{tag}_stdout.log")
    stderr_path = os.path.join(TMP_DIR, f"{tag}_stderr.log")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if os.path.exists(result_path):
        os.remove(result_path)

    cmd = [
        "swift", "infer",
        "--model", MODEL_DIR,
        "--adapters", lora_path,
        "--val_dataset", jsonl_path,
        "--template", "qwen3_omni",
        "--quant_bits", "4",
        "--max_length", str(MAX_LENGTH),
        "--max_new_tokens", str(MAX_NEW_TOKENS),
        "--result_path", result_path,
        "--model_kwargs",
        '{"disable_talker": true, "use_audio_in_video": true, "trust_remote_code": true, "attn_implementation": "flash_attention_2"}'
    ]

    log_print(f"[INFO] launching swift infer for {tag} size={len(records)}")

    with open(stdout_path, "w", encoding="utf-8") as out_f, open(stderr_path, "w", encoding="utf-8") as err_f:
        proc = subprocess.run(cmd, stdout=out_f, stderr=err_f, text=True)

    log_print(f"[INFO] finished swift infer for {tag} returncode={proc.returncode}")

    stdout_text = ""
    stderr_text = ""

    if os.path.exists(stdout_path):
        with open(stdout_path, "r", encoding="utf-8", errors="ignore") as f:
            stdout_text = f.read()
    if os.path.exists(stderr_path):
        with open(stderr_path, "r", encoding="utf-8", errors="ignore") as f:
            stderr_text = f.read()

    if proc.returncode == 0 and os.path.exists(result_path):
        return {
            "ok": True,
            "result_path": result_path,
            "stdout_tail": stdout_text[-2000:],
            "stderr_tail": stderr_text[-2000:],
        }

    return {
        "ok": False,
        "error_type": classify_error(stderr_text, stdout_text),
        "result_path": result_path,
        "stdout_tail": stdout_text[-2000:],
        "stderr_tail": stderr_text[-2000:],
    }


# ==================== 5. 主流程 ====================
def run_swift_inference():
    if CLEAN_TMP_ON_START and os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR, ignore_errors=True)
        os.makedirs(TMP_DIR, exist_ok=True)

    if os.path.exists(REALTIME_LOG_PATH):
        os.remove(REALTIME_LOG_PATH)
    if os.path.exists(INFER_RESULT_PATH):
        os.remove(INFER_RESULT_PATH)

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

    log_print("=" * 80)
    log_print("[INFO] Qwen3-Omni 剩余测试集分块推理开始")
    log_print(f"[INFO] 使用 checkpoint: {lora_path}")
    log_print(f"[INFO] CSV_PATH: {CSV_PATH}")
    log_print(f"[INFO] CHUNK_SIZE: {CHUNK_SIZE}")
    log_print("=" * 80)

    merged_clip_paths = build_merged_clip_index(MERGED_CLIP_DIR)
    all_records, prep_stats = prepare_swift_dataset_from_existing_clips(
        df_gold=df_gold,
        merged_clip_paths=merged_clip_paths,
        train_system=train_system,
        train_query=train_query,
        jsonl_path=SWIFT_JSONL_PATH,
    )

    if len(all_records) == 0:
        log_print("[ERROR] 没有有效样本")
        sys.exit(1)

    if DEBUG_LIMIT is not None:
        all_records = all_records[:DEBUG_LIMIT]
        log_print(f"[INFO] DEBUG_LIMIT 已启用，仅处理前 {len(all_records)} 条")

    pred_records = []
    bad_cases = []
    pred_counter = Counter()
    parse_status_counter = Counter()
    y_true, y_pred = [], []

    start_time = time.time()
    processed = 0
    success_rows = 0
    skipped_rows = 0
    split_calls = 0

    def handle_success(result_path: str, input_records: List[dict]):
        nonlocal success_rows, pred_records, y_true, y_pred

        append_jsonl_lines(result_path, INFER_RESULT_PATH)
        out_rows = load_jsonl(result_path)

        if len(out_rows) != len(input_records):
            input_map = {str(r["video_id"]): r for r in input_records}
            for data in out_rows:
                videos = data.get("videos", [])
                clip_path = videos[0] if videos else ""
                base = os.path.splitext(os.path.basename(clip_path))[0]
                vid = base.replace(f"_clip{CLIP_SECONDS}s", "")
                rec = input_map.get(str(vid), None)
                if rec is None:
                    continue
                gold = int(rec["gold_label"])
                response = data.get("response", "")
                pred, parse_status = extract_label(response)
                parse_status_counter[parse_status] += 1
                pred_records.append({
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
                success_rows += 1
            return

        for rec, data in zip(input_records, out_rows):
            vid = str(rec["video_id"])
            gold = int(rec["gold_label"])
            clip_path = rec["videos"][0]
            response = data.get("response", "")

            pred, parse_status = extract_label(response)
            parse_status_counter[parse_status] += 1

            pred_records.append({
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

            success_rows += 1

    def handle_bad_single(rec: dict, err: Dict, local_idx: int):
        nonlocal skipped_rows
        skipped_rows += 1
        bad_cases.append({
            "idx_in_remaining_valid": local_idx,
            "video_id": str(rec["video_id"]),
            "gold_label": int(rec["gold_label"]),
            "clip_path": rec["videos"][0],
            "error_type": err.get("error_type", "other_error"),
            "stdout_tail": err.get("stdout_tail", ""),
            "stderr_tail": err.get("stderr_tail", ""),
        })
        log_print(
            f"[WARN] skip bad sample idx={local_idx} "
            f"video_id={rec['video_id']} error_type={err.get('error_type', 'other_error')}"
        )

    def process_records(records: List[dict], tag: str, base_idx: int):
        nonlocal split_calls, processed

        res = run_chunk(records, tag, lora_path)

        if res["ok"]:
            handle_success(res["result_path"], records)
            processed += len(records)
            return

        if len(records) == 1:
            handle_bad_single(records[0], res, base_idx + 1)
            processed += 1
            return

        split_calls += 1
        mid = len(records) // 2
        left = records[:mid]
        right = records[mid:]
        log_print(f"[INFO] split chunk {tag} -> {len(left)} + {len(right)} because {res['error_type']}")
        process_records(left, f"{tag}_L", base_idx)
        process_records(right, f"{tag}_R", base_idx + mid)

    for offset, chunk in chunk_iter(all_records, CHUNK_SIZE):
        tag = f"chunk_{offset:05d}_{offset + len(chunk):05d}"
        log_print(f"[INFO] start chunk: {tag} size={len(chunk)}")
        process_records(chunk, tag, offset)

        if processed % PRINT_EVERY == 0 or processed == len(all_records):
            acc = accuracy_score(y_true, y_pred) if y_true else None
            macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else None
            elapsed = time.time() - start_time
            if acc is None:
                log_print(
                    f"[REALTIME] processed={processed}/{len(all_records)} | "
                    f"success={success_rows} | skipped={skipped_rows} | "
                    f"splits={split_calls} | elapsed={elapsed/60:.1f}m"
                )
            else:
                log_print(
                    f"[REALTIME] processed={processed}/{len(all_records)} | "
                    f"success={success_rows} | skipped={skipped_rows} | "
                    f"acc={acc:.4f} | macro_f1={macro_f1:.4f} | "
                    f"splits={split_calls} | elapsed={elapsed/60:.1f}m"
                )

    pd.DataFrame(pred_records).to_csv(PRED_CSV_PATH, index=False, encoding="utf-8-sig")

    with open(BAD_CASES_JSON, "w", encoding="utf-8") as f:
        json.dump(bad_cases, f, ensure_ascii=False, indent=2)
    pd.DataFrame(bad_cases).to_csv(BAD_CASES_CSV, index=False, encoding="utf-8-sig")

    if len(y_true) > 0:
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        report = classification_report(y_true, y_pred, digits=4, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

        parse_failed_rows = sum(1 for x in pred_records if x["pred_label"] is None)

        report_text = []
        report_text.append("=" * 80)
        report_text.append("Qwen3-Omni 剩余测试集分块推理报告")
        report_text.append("=" * 80)
        report_text.append(f"Remaining rows in csv: {len(df_gold)}")
        report_text.append(f"Valid clip rows: {len(all_records)}")
        report_text.append(f"Inference success rows: {success_rows}")
        report_text.append(f"Skipped rows: {skipped_rows}")
        report_text.append(f"Evaluated rows: {len(y_true)}")
        report_text.append(f"Parse failed rows: {parse_failed_rows}")
        report_text.append(f"Clip stats: {prep_stats}")
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

        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report_text)

        pd.DataFrame(
            cm,
            index=[f"true_{i}" for i in [0, 1, 2, 3, 4]],
            columns=[f"pred_{i}" for i in [0, 1, 2, 3, 4]],
        ).to_csv(CM_PATH, encoding="utf-8-sig")

        with open(PRED_DIST_PATH, "w", encoding="utf-8") as f:
            json.dump(dict(pred_counter), f, ensure_ascii=False, indent=2)

        with open(PARSE_DIST_PATH, "w", encoding="utf-8") as f:
            json.dump(dict(parse_status_counter), f, ensure_ascii=False, indent=2)

        log_print(report_text)
    else:
        log_print("[WARN] 没有任何成功解析样本，未生成分类报告。")

    log_print(f"[INFO] 逐条预测已保存: {PRED_CSV_PATH}")
    log_print(f"[INFO] 汇总结果已保存: {INFER_RESULT_PATH}")
    log_print(f"[INFO] 坏样本清单已保存: {BAD_CASES_CSV}")
    log_print(f"[INFO] 坏样本明细已保存: {BAD_CASES_JSON}")


if __name__ == "__main__":
    run_swift_inference()