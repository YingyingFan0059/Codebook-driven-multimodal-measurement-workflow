#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
import pandas as pd

PROJECT_ROOT = "/root/autodl-tmp/project_douyin_mm"
EVAL_DIR = f"{PROJECT_ROOT}/outputs/qwen3_omni_eval_clip16_wav"
SNAP_DIR = f"{EVAL_DIR}/snapshot_first3000"

TEST_CSV = f"{PROJECT_ROOT}/splits/split_v3/test_main.csv"

BAD_JSON_MAIN = f"{EVAL_DIR}/qwen3_remaining_bad_cases.json"
BAD_CSV_MAIN = f"{EVAL_DIR}/qwen3_remaining_bad_cases.csv"

OUT_CSV = f"{EVAL_DIR}/bad_videos_for_rebuild.csv"
OUT_JSON = f"{EVAL_DIR}/bad_videos_for_rebuild.json"


def load_bad_from_json(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        for x in data:
            rows.append({
                "video_id": str(x.get("video_id", "")).strip(),
                "error_type": x.get("error_type", ""),
                "clip_path": x.get("clip_path", x.get("video", "")),
                "source_file": path,
                "stderr_tail": x.get("stderr_tail", ""),
                "stdout_tail": x.get("stdout_tail", ""),
            })
    return rows


def load_bad_from_csv(path):
    rows = []
    if not os.path.exists(path):
        return rows
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        rows.append({
            "video_id": str(r.get("video_id", "")).strip(),
            "error_type": r.get("error_type", ""),
            "clip_path": r.get("clip_path", r.get("video", "")),
            "source_file": path,
            "stderr_tail": str(r.get("stderr_tail", "")),
            "stdout_tail": str(r.get("stdout_tail", "")),
        })
    return rows


def load_bad_from_txts(eval_dir):
    rows = []
    txt_files = glob.glob(os.path.join(eval_dir, "debug_len_*", "bad_video_ids.txt")) + \
                glob.glob(os.path.join(eval_dir, "debug_len_*", "*.txt")) + \
                glob.glob(os.path.join(eval_dir, "debug_len_find_all", "bad_video_ids.txt")) + \
                glob.glob(os.path.join(eval_dir, "debug_len_scan_all", "bad_video_ids.txt"))

    for path in txt_files:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) >= 3:
                    # 常见格式：idx video_id error_type video
                    if parts[0].isdigit():
                        video_id = parts[1].strip()
                        error_type = parts[2].strip()
                        clip_path = parts[3].strip() if len(parts) >= 4 else ""
                    else:
                        video_id = parts[0].strip()
                        error_type = parts[1].strip() if len(parts) >= 2 else ""
                        clip_path = parts[2].strip() if len(parts) >= 3 else ""
                    rows.append({
                        "video_id": video_id,
                        "error_type": error_type,
                        "clip_path": clip_path,
                        "source_file": path,
                        "stderr_tail": "",
                        "stdout_tail": "",
                    })
    return rows


def main():
    all_rows = []
    all_rows += load_bad_from_json(BAD_JSON_MAIN)
    all_rows += load_bad_from_csv(BAD_CSV_MAIN)
    all_rows += load_bad_from_txts(EVAL_DIR)

    all_rows = [x for x in all_rows if x["video_id"]]

    if not all_rows:
        print("[WARN] 没找到任何历史坏样本记录")
        return

    df_bad = pd.DataFrame(all_rows)

    # 去重：同一个 video_id 保留第一条，同时合并 error_type
    agg = (
        df_bad.groupby("video_id", as_index=False)
        .agg({
            "error_type": lambda x: "|".join(sorted(set(str(i) for i in x if str(i).strip()))),
            "clip_path": "first",
            "source_file": lambda x: "|".join(sorted(set(str(i) for i in x if str(i).strip()))),
            "stderr_tail": "first",
            "stdout_tail": "first",
        })
    )

    # 关联 test_main.csv，补充 gold_label / 标题 / 链接
    if os.path.exists(TEST_CSV):
        df_test = pd.read_csv(TEST_CSV)
        df_test["video_id"] = df_test["video_id"].astype(str).str.strip()

        label_col = "gold_label" if "gold_label" in df_test.columns else ("final_code" if "final_code" in df_test.columns else None)
        keep_cols = ["video_id"]
        if "标题" in df_test.columns:
            keep_cols.append("标题")
        if "视频链接" in df_test.columns:
            keep_cols.append("视频链接")
        if label_col:
            keep_cols.append(label_col)

        df_test = df_test[keep_cols].copy()
        if label_col:
            df_test = df_test.rename(columns={label_col: "gold_label"})

        agg = agg.merge(df_test, on="video_id", how="left")

    # 给出建议处理策略
    def suggest_fix(err):
        err = str(err)
        if "max_length_error" in err:
            return "shorten_clip"
        if "decord_error" in err or "video_fps_error" in err or "ffmpeg_filter_graph_error" in err:
            return "rebuild_clip"
        return "manual_check"

    agg["suggested_action"] = agg["error_type"].apply(suggest_fix)

    # 排序
    agg = agg.sort_values(by=["suggested_action", "video_id"]).reset_index(drop=True)

    agg.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    agg.to_json(OUT_JSON, orient="records", force_ascii=False, indent=2)

    print(f"[INFO] total_bad_video_ids = {len(agg)}")
    print(f"[INFO] saved csv: {OUT_CSV}")
    print(f"[INFO] saved json: {OUT_JSON}")
    print("\n[INFO] action distribution:")
    print(agg["suggested_action"].value_counts(dropna=False))


if __name__ == "__main__":
    main()