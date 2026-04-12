#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正 swift_train.jsonl 的 prompt 和 response 格式
- 加入明确的类别定义 (对齐Qwen2-VL-7B成功的prompt)
- response 从纯数字 "4" 改为 JSON '{"label": 4}'
- 保持旧版 swift 格式 (query/response/videos)
"""

import json
import os
import shutil

PROJECT_ROOT = "/root/autodl-tmp/project_douyin_mm"
INPUT_PATH = f"{PROJECT_ROOT}/splits/split_v3/swift_train.jsonl"
OUTPUT_PATH = f"{PROJECT_ROOT}/splits/split_v3/swift_train_v2.jsonl"
BACKUP_PATH = f"{PROJECT_ROOT}/splits/split_v3/swift_train_v1_backup.jsonl"

# 对齐 Qwen2-VL-7B 成功的 prompt 风格：简洁 + 类别定义明确
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


def main():
    # 备份原文件
    if not os.path.exists(BACKUP_PATH):
        shutil.copy2(INPUT_PATH, BACKUP_PATH)
        print(f"[INFO] 原文件已备份至: {BACKUP_PATH}")

    count = 0
    label_dist = {}

    with open(INPUT_PATH, "r", encoding="utf-8") as fin, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as fout:

        for line in fin:
            if not line.strip():
                continue

            data = json.loads(line)

            # 提取原始 label
            response_raw = data.get("response", "").strip()
            try:
                label = int(response_raw)
            except ValueError:
                # 尝试从JSON解析
                try:
                    obj = json.loads(response_raw)
                    label = int(obj.get("label", -1))
                except:
                    print(f"[WARN] 无法解析 response: {response_raw}, 跳过")
                    continue

            if label < 0 or label > 4:
                print(f"[WARN] label 不在 0-4 范围: {label}, 跳过")
                continue

            # 提取视频路径
            videos = data.get("videos", [])

            # 构建新记录
            new_record = {
                "system": SYSTEM_PROMPT,
                "query": USER_QUERY,
                "response": json.dumps({"label": label}, ensure_ascii=False),
                "videos": videos
            }

            fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")
            count += 1
            label_dist[label] = label_dist.get(label, 0) + 1

    print(f"\n[DONE] 修正完成!")
    print(f"  输出文件: {OUTPUT_PATH}")
    print(f"  总条数: {count}")
    print(f"  标签分布:")
    for label in sorted(label_dist.keys()):
        print(f"    类别 {label}: {label_dist[label]} 条")

    # 验证: 打印第一条
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        first = json.loads(f.readline())
    print(f"\n[VERIFY] 第一条样本:")
    print(f"  system: {first['system'][:60]}...")
    print(f"  query: {first['query']}")
    print(f"  response: {first['response']}")
    print(f"  videos: {first['videos']}")


if __name__ == "__main__":
    main()