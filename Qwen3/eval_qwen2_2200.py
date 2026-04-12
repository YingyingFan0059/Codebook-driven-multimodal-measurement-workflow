#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import re
import json
import warnings
import pandas as pd
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from sklearn.metrics import classification_report, f1_score, accuracy_score

# 🌟 屏蔽 Transformers 的生成配置警告，保持日志纯净
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.generation.configuration_utils")

# 引入训练脚本中的关键配置、路径与工具函数
from train_scaling_qwen import (
    extract_frames_ffmpeg_cached,
    load_frames_as_pil,
    GLOBAL_VIDEO_PATHS,
    PROJECT_ROOT,
    MODEL_ID,
    SPLITS_DIR,
    CACHE_DIR,
    INSTR
)

EVAL_OUT_DIR = os.path.join(PROJECT_ROOT, "outputs/scaling_eval")
os.makedirs(EVAL_OUT_DIR, exist_ok=True)

# 只做最小改动：在原始 INSTR 后追加一句，尽量让模型给出 reason
INSTR_WITH_REASON = (
    INSTR.strip()
    + '\n\n'
    + 'Before giving the final label, provide one brief reason. '
      'Return JSON only in this format: {"reason":"...", "label":X}.'
)

def parse_robust_json(text):
    text = text.strip()

    markdown_block_pattern = "`" * 3 + r"(?:json)?(.*?)" + "`" * 3
    match = re.search(markdown_block_pattern, text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    reason = ""

    # 优先按 JSON 解析
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            if "reason" in obj:
                reason = str(obj["reason"]).strip()
            if "label" in obj:
                label = int(obj["label"])
                return label, reason
    except Exception:
        pass

    # 正则提取 reason
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
    if reason_match:
        reason = reason_match.group(1).strip()

    # 正则提取 label
    label_match = re.search(r'"label"\s*:\s*(\d)', text)
    if label_match:
        return int(label_match.group(1)), reason

    digits = ''.join(filter(str.isdigit, text))
    return (int(digits[0]) if digits else 0), reason

def evaluate_model(train_size, test_csv_name="test_main.csv"):
    print(f"\n{'='*50}\n🔍 正在统考: lora_5class_{train_size} | 考卷: {test_csv_name}\n{'='*50}", flush=True)

    lora_path = os.path.join(PROJECT_ROOT, f"runs/scaling_experiments/lora_5class_{train_size}")
    if not os.path.exists(lora_path):
        print(f"⚠️ 模型权重未找到: {lora_path}，跳过此评估。", flush=True)
        return

    test_csv_path = os.path.join(SPLITS_DIR, test_csv_name)
    if not os.path.exists(test_csv_path):
        return
    test_df = pd.read_csv(test_csv_path, encoding='utf-8-sig')

    output_base = os.path.join(EVAL_OUT_DIR, f"eval_lora_5class_{train_size}_reason")
    pred_csv_path = f"{output_base}_predictions.csv"
    if os.path.exists(pred_csv_path):
        os.remove(pred_csv_path)

    print(f"  [Step 1/3] 加载 Processor ({MODEL_ID})...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print("  [Step 2/3] 加载 7B 基座模型...", flush=True)
    # 👇 严格恢复你的核心修复点：将 device_map 强制设为 "cuda"，绕过 accelerate 的 Bug
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa"
    )

    print(f"  [Step 3/3] 融合 LoRA 权重 (N={train_size})...", flush=True)
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()
    print("✅ Model loaded successfully. 准备开启推理管线...", flush=True)

    y_true, y_pred = [], []

    for idx, row in test_df.iterrows():
        video_id = str(row['video_id']).strip()
        gold = int(row['gold_label'] if 'gold_label' in test_df.columns else row['final_code'])

        if video_id not in GLOBAL_VIDEO_PATHS:
            continue
        video_path = GLOBAL_VIDEO_PATHS[video_id]

        try:
            frame_paths = extract_frames_ffmpeg_cached(
                video_path=video_path, cache_dir=CACHE_DIR, fps=0.5, max_frames=8, max_pixels=360*480
            )
            if not frame_paths:
                continue

            frames_pil = load_frames_as_pil(frame_paths)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": frames_pil},
                    {"type": "text", "text": INSTR_WITH_REASON}
                ]
            }]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to("cuda")

            with torch.no_grad():
                # 严格保留你原来的推理逻辑
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    temperature=0.1,
                    do_sample=True
                )
                trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                out_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

            pred, reason = parse_robust_json(out_text)
            if pred not in [0, 1, 2, 3, 4]:
                pred = 0

            y_true.append(gold)
            y_pred.append(pred)

            item_res = {
                "video_id": video_id,
                "gold_label": gold,
                "pred_label": pred,
                "reason": reason,
                "raw_output": out_text.strip().replace('\n', ' ')
            }
            pd.DataFrame([item_res]).to_csv(
                pred_csv_path,
                mode='a',
                index=False,
                header=not os.path.exists(pred_csv_path),
                encoding="utf-8-sig"
            )

            if idx == 0:
                print(f"💓 [Heartbeat] 第 1 个视频({video_id})推理成功！显存通道正常运转中...", flush=True)
            elif (idx + 1) % 100 == 0:
                print(f"📈 [Progress] {idx+1}/{len(test_df)} | Current Acc: {accuracy_score(y_true, y_pred):.4f}", flush=True)

        except Exception as e:
            # 🌟 坚决不静默失败！如果有视频抽帧或显存溢出，直接报红字
            print(f"❌ [Error] 视频 {video_id} 处理异常: {str(e)}", flush=True)

    if len(y_true) > 0:
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        report = classification_report(y_true, y_pred, digits=4, zero_division=0)

        with open(f"{output_base}_report.txt", "w", encoding="utf-8") as f:
            f.write(
                f"Experiment: Scaling Law - Qwen2-VL-7B\n"
                f"Training Size (N): {train_size}\n"
                f"Test Set: {test_csv_name} (Pure N=5000)\n"
                f"Accuracy: {acc:.4f}\n"
                f"Macro-F1: {macro_f1:.4f}\n\n"
                f"Detailed Classification Report:\n{report}"
            )

        print("\n" + "="*40, flush=True)
        print(f"🎯 评估成功 (N={train_size}) | Accuracy : {acc:.4f} | Macro-F1 : {macro_f1:.4f}", flush=True)
        print("="*40 + "\n", flush=True)
    else:
        print(f"⚠️ 警告：模型 lora_5class_{train_size} 未能产生任何有效预测。", flush=True)

    del model, base_model, processor
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # 只跑 N=2200
    print("🔔 启动 N=2200 评估任务...", flush=True)
    evaluate_model(2200, test_csv_name="test_main.csv")
    print("\n🎉 N=2200 评估已完成。", flush=True)