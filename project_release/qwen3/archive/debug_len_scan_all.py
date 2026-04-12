#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import subprocess

SRC = "/root/autodl-tmp/project_douyin_mm/outputs/qwen3_omni_eval_clip16_wav/swift_eval_dataset_qwen3_remaining.jsonl"
WORKDIR = "/root/autodl-tmp/project_douyin_mm/outputs/qwen3_omni_eval_clip16_wav/debug_len_find_all"
MODEL = "/root/autodl-tmp/project_douyin_mm/models/qwen/Qwen3-Omni-30B-A3B-Instruct"
ADAPTER = "/root/autodl-tmp/project_douyin_mm/runs/qwen3_omni_lora_aligned_v2/v5-20260310-174026/checkpoint-414"

os.makedirs(WORKDIR, exist_ok=True)

with open(SRC, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

print("total_rows =", len(rows))

def run_slice(l, r):
    part_path = os.path.join(WORKDIR, f"slice_{l}_{r}.jsonl")
    out_path = os.path.join(WORKDIR, f"slice_{l}_{r}_result.jsonl")
    err_path = os.path.join(WORKDIR, f"slice_{l}_{r}_stderr.log")
    outlog_path = os.path.join(WORKDIR, f"slice_{l}_{r}_stdout.log")

    if not os.path.exists(part_path):
        with open(part_path, "w", encoding="utf-8") as w:
            for x in rows[l:r]:
                w.write(json.dumps(x, ensure_ascii=False) + "\n")

    cmd = f"""swift infer \
  --model {MODEL} \
  --adapters {ADAPTER} \
  --val_dataset {part_path} \
  --template qwen3_omni \
  --quant_bits 4 \
  --max_length 8192 \
  --max_new_tokens 32 \
  --result_path {out_path} \
  --model_kwargs '{{"disable_talker": true, "use_audio_in_video": true, "trust_remote_code": true, "attn_implementation": "flash_attention_2"}}' \
  > {outlog_path} 2> {err_path}"""

    code = subprocess.call(cmd, shell=True)

    err_txt = ""
    if os.path.exists(err_path):
        with open(err_path, "r", encoding="utf-8", errors="ignore") as f:
            err_txt = f.read()

    if code == 0:
        return "ok", err_txt
    elif "MaxLengthError" in err_txt:
        return "max_length_error", err_txt
    else:
        return "other_error", err_txt

bad_cases = []
visited = 0

def search(l, r):
    global visited
    visited += 1
    print(f"check [{l}, {r}) size={r-l} visited={visited}", flush=True)

    status, err_txt = run_slice(l, r)

    if status == "ok":
        return

    if r - l == 1:
        row = rows[l]
        bad_cases.append({
            "idx": l,
            "video_id": row.get("video_id"),
            "video": row.get("videos", [""])[0] if row.get("videos") else "",
            "status": status,
            "stderr_tail": err_txt[-1000:]
        })
        print(f"FOUND idx={l} video_id={row.get('video_id')} status={status}", flush=True)
        return

    mid = (l + r) // 2
    search(l, mid)
    search(mid, r)

search(0, len(rows))

bad_cases = sorted(bad_cases, key=lambda x: x["idx"])

bad_json = os.path.join(WORKDIR, "bad_cases_all.json")
bad_txt = os.path.join(WORKDIR, "bad_video_ids.txt")

with open(bad_json, "w", encoding="utf-8") as f:
    json.dump(bad_cases, f, ensure_ascii=False, indent=2)

with open(bad_txt, "w", encoding="utf-8") as f:
    for x in bad_cases:
        f.write(f"{x['idx']}\t{x['video_id']}\t{x['status']}\t{x['video']}\n")

print("\nDONE")
print("bad_count =", len(bad_cases))
print("bad_json =", bad_json)
print("bad_txt =", bad_txt)