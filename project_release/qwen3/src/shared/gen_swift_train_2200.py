#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 train_2200.csv 生成 swift 格式训练数据
对齐 Qwen2-VL-7B 的 2200 条实验
"""

import json
import os
import pandas as pd

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/root/autodl-tmp/project_douyin_mm")
CSV_PATH = os.environ.get(
    "QWEN3_TRAIN_CSV",
    f"{PROJECT_ROOT}/splits/split_v3/train_2200.csv"
)
OUTPUT_PATH = os.environ.get(
    "QWEN3_TRAIN_DATA",
    f"{PROJECT_ROOT}/splits/split_v3/swift_train_2200.jsonl"
)
VIDEO_BASE_DIR = os.environ.get(
    "VIDEO_BASE_DIR",
    f"{PROJECT_ROOT}/videos/douyin/upload_pack"
)

# 与 fix_training_data.py 完全一致的 prompt
SYSTEM_PROMPT = (
    "你是一位严格的计算社会科学编码员。请同时观看视频画面并聆听音频，将该政务短视频进行唯一分类。\n"
    "类别定义：\n"
    "0 OTHER_OR_UNCLEAR (其他)：主持人播报新闻、纯风景混剪、视听线索矛盾无法归类；\n"
    "1 PERFORMATIVE (绩效展示)：真实社会任务的执行，如站岗巡逻、救灾运输、抓捕抢险；\n"
    "2 MORAL (道德动员)：情感与精神符号，如军民互动、艰苦环境特写、致敬烈士、煽情BGM；\n"
    "3 PROCEDURAL (程序规范)：权力与秩序的仪式，如列队会议、授衔仪式、庄严军乐；\n"
    "4 TECHNICAL (技术与实战)：纯粹武力/专业度展示，如武器特写、实弹演习、战术队形。\n"
    "输出要求：仅返回JSON对象，例如 {\"label\": 4}。不要包含任何多余解释。"
)

USER_QUERY = "<video>请根据视频画面和音频内容对该政务短视频进行分类，仅返回JSON对象。"


def build_video_path_index(base_dir):
    path_map = {}
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(('.mp4', '.avi', '.mov')):
                path_map[os.path.splitext(f)[0]] = os.path.join(root, f)
    return path_map


def main():
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] CSV 文件不存在: {CSV_PATH}")
        return

    print(f"[INFO] 读取 CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"[INFO] CSV 行数: {len(df)}")

    video_paths = build_video_path_index(VIDEO_BASE_DIR)
    print(f"[INFO] 视频索引: {len(video_paths)} 个")

    count = 0
    missing = 0
    label_dist = {}

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            vid = str(row.get("video_id", "")).strip()
            label = int(row["gold_label"] if "gold_label" in df.columns else row["final_code"])

            video_path = video_paths.get(vid)
            if not video_path or not os.path.exists(video_path):
                missing += 1
                continue

            record = {
                "system": SYSTEM_PROMPT,
                "query": USER_QUERY,
                "response": json.dumps({"label": label}, ensure_ascii=False),
                "videos": [video_path]
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            label_dist[label] = label_dist.get(label, 0) + 1

    print(f"\n[DONE] 生成完成!")
    print(f"  输出: {OUTPUT_PATH}")
    print(f"  有效: {count} 条")
    print(f"  缺失: {missing} 条")
    print(f"  标签分布:")
    for label in sorted(label_dist.keys()):
        print(f"    类别 {label}: {label_dist[label]} 条")

    # 验证
    with open(OUTPUT_PATH) as f:
        first = json.loads(f.readline())
    print(f"\n[VERIFY] 第一条:")
    print(f"  response: {first['response']}")
    print(f"  video: {first['videos'][0][:60]}...")


if __name__ == "__main__":
    main()
