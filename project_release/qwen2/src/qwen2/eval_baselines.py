#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import gc
import glob
import os
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import torch
from qwen_vl_utils import process_vision_info
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


RELEASE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(RELEASE_ROOT)))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "cache" / "hf"))
os.environ.setdefault("TMPDIR", str(PROJECT_ROOT / "tmp"))

BASE_VIDEO_DIR = os.environ.get("VIDEO_BASE_DIR", str(PROJECT_ROOT / "data" / "videos"))
SPLITS_DIR = os.environ.get("QWEN2_SPLITS_DIR", str(RELEASE_ROOT / "splits" / "split_v3"))
ARTIFACT_ROOT = Path(os.environ.get("QWEN2_ARTIFACT_ROOT", str(RELEASE_ROOT / "artifacts" / "qwen2")))
MODEL_ID = os.environ.get(
    "QWEN2_MODEL_DIR",
    str(PROJECT_ROOT / "models" / "qwen2" / "Qwen2-VL-7B-Instruct"),
)
CACHE_DIR = os.environ.get("QWEN2_FRAMES_CACHE_DIR", str(ARTIFACT_ROOT / "cache" / "frames_cache"))
EVAL_OUT_DIR = os.environ.get("QWEN2_BASELINE_OUT_DIR", str(ARTIFACT_ROOT / "outputs" / "baselines"))
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", shutil.which("ffmpeg") or "ffmpeg")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(EVAL_OUT_DIR, exist_ok=True)

PROMPT_PATH1 = """You are a computational social science video coder.
Classify the short video into exactly one label from {0,1,2,3,4}.

Label definitions:
0 OTHER_OR_UNCLEAR: unclear, mixed, generic news reading, or non-analytic content.
1 PERFORMATIVE: concrete mission execution or visible institutional performance.
2 MORAL: emotional or symbolic mobilization, sacrifice, solidarity, tribute, patriotic sentiment.
3 PROCEDURAL: ceremony, order, hierarchy, ritual, rule-following, standardized procedure.
4 TECHNICAL: weapons, tactics, drills, combat capability, technical or professional skill display.

Output only a JSON object like {"label": 1}. Do not add any explanation."""

PROMPT_PATH2 = """You are a computational social science coder using a structured visual rulebook.
Classify the short video into exactly one label from {0,1,2,3,4}.

Decision rules:
- Choose 4 TECHNICAL when the clip foregrounds weapons, tactics, combat drills, formations, or professional capability.
- Choose 1 PERFORMATIVE when the clip foregrounds concrete mission execution, patrol, rescue, transport, emergency response, or visible task completion.
- Choose 2 MORAL when the clip foregrounds emotional symbolism, hardship, tribute, bonding, sacrifice, or patriotic affect.
- Choose 3 PROCEDURAL when the clip foregrounds ceremony, formal order, hierarchy, disciplinary ritual, assemblies, or standardized institutional procedure.
- Choose 0 OTHER_OR_UNCLEAR only when none of the four substantive categories can be defended.

Boundary rules:
- If visible action and task completion dominate, prefer 1 over 2.
- If symbolism dominates over action, prefer 2 over 1.
- If ritualized order dominates, prefer 3.
- If technical proficiency or combat capability dominates, prefer 4.

Output only a JSON object like {"label": 4}. Do not add markdown or explanation."""


def build_video_path_index(base_dir: str) -> dict[str, str]:
    print("Scanning video paths...")
    path_map: dict[str, str] = {}
    for root, _, files in os.walk(base_dir):
        for filename in files:
            if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")):
                video_id = os.path.splitext(filename)[0]
                path_map[video_id] = os.path.join(root, filename)
    print(f"Indexed {len(path_map)} videos.")
    return path_map


GLOBAL_VIDEO_PATHS = build_video_path_index(BASE_VIDEO_DIR)


def safe_extract_frames(video_path: str) -> list[str] | None:
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(CACHE_DIR, video_id)
    os.makedirs(out_dir, exist_ok=True)

    existing = sorted(glob.glob(f"{out_dir}/*.jpg"))
    if not existing:
        command = [
            FFMPEG_BIN,
            "-y",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-vf",
            "fps=0.5,scale='min(360,iw)':'min(480,ih)'",
            "-q:v",
            "2",
            f"{out_dir}/frame_%04d.jpg",
        ]
        subprocess.run(command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        existing = sorted(glob.glob(f"{out_dir}/*.jpg"))

    if not existing:
        return None
    if len(existing) > 8:
        sample_ids = [int(i) for i in torch.linspace(0, len(existing) - 1, steps=8).tolist()]
        existing = [existing[index] for index in sample_ids]
    if len(existing) % 2 != 0:
        existing.append(existing[-1])
    return existing


def parse_robust_json(text: str) -> int:
    match = re.search(r'"label"\s*:\s*(\d)', text)
    if match:
        return int(match.group(1))
    digits = "".join(filter(str.isdigit, text))
    return int(digits[0]) if digits else 0


def load_model() -> tuple[AutoProcessor, Qwen2VLForConditionalGeneration]:
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )
    device_map = "auto" if torch.cuda.is_available() else None

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map=device_map,
        attn_implementation="sdpa",
    )
    model.eval()
    return processor, model


def run_baseline_eval(model, processor, path_name: str, prompt_text: str, test_csv_name: str = "test_main.csv") -> None:
    print(f"\n{'=' * 50}\nRunning baseline: [{path_name}]\n{'=' * 50}")

    df = pd.read_csv(os.path.join(SPLITS_DIR, test_csv_name), encoding="utf-8-sig")
    y_true: list[int] = []
    y_pred: list[int] = []
    results: list[dict[str, object]] = []

    output_csv = os.path.join(EVAL_OUT_DIR, f"eval_baseline_{path_name}_predictions.csv")
    report_txt = os.path.join(EVAL_OUT_DIR, f"eval_baseline_{path_name}_report.txt")

    missing_count = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Baseline {path_name}", unit="vid"):
        video_id = str(row["video_id"]).strip()
        gold = int(row["gold_label"] if "gold_label" in df.columns else row["final_code"])

        video_path = GLOBAL_VIDEO_PATHS.get(video_id)
        if not video_path:
            missing_count += 1
            continue

        frames = safe_extract_frames(video_path)
        if not frames:
            continue

        messages = [
            {
                "role": "user",
                "content": [{"type": "video", "video": frames}, {"type": "text", "text": prompt_text}],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=40)
            trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            out_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

        pred = parse_robust_json(out_text)
        if pred not in [0, 1, 2, 3, 4]:
            pred = 0

        y_true.append(gold)
        y_pred.append(pred)
        results.append(
            {
                "video_id": video_id,
                "gold_label": gold,
                "pred_label": pred,
                "raw_output": out_text.strip(),
            }
        )

        if len(results) % 50 == 0:
            pd.DataFrame(results).to_csv(output_csv, index=False, encoding="utf-8-sig")

    pd.DataFrame(results).to_csv(output_csv, index=False, encoding="utf-8-sig")

    valid_labels = [0, 1, 2, 3, 4]
    if y_true:
        macro_f1 = f1_score(y_true, y_pred, labels=valid_labels, average="macro", zero_division=0)
        report = classification_report(y_true, y_pred, labels=valid_labels, zero_division=0)
    else:
        macro_f1 = 0.0
        report = "No predictions were generated."

    with open(report_txt, "w", encoding="utf-8") as handle:
        handle.write(f"Baseline: {path_name}\nMacro-F1: {macro_f1:.4f}\n\n{report}")

    print(f"{path_name} finished. Missing videos: {missing_count}. Report written to {report_txt}")


def run_selected_baselines(methods: list[str] | None = None, test_csv_name: str = "test_main.csv") -> None:
    method_map = {
        "zeroshot": ("path1_zeroshot", PROMPT_PATH1),
        "zero-shot": ("path1_zeroshot", PROMPT_PATH1),
        "rule": ("path2_rulebased", PROMPT_PATH2),
        "rule-based": ("path2_rulebased", PROMPT_PATH2),
        "rulebased": ("path2_rulebased", PROMPT_PATH2),
    }
    selected = methods or ["zeroshot", "rule"]
    pairs = [method_map[method] for method in selected]

    processor, model = load_model()
    try:
        for path_name, prompt_text in pairs:
            run_baseline_eval(model, processor, path_name, prompt_text, test_csv_name=test_csv_name)
    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    run_selected_baselines()
