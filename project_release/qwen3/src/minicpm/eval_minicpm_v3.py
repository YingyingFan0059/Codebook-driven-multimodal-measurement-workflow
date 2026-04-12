#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

import re
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report

import math
import tempfile
import librosa
from moviepy.editor import VideoFileClip
from PIL import Image

# ================= 1. 配置路径 =================
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/root/autodl-tmp/project_douyin_mm")
BASE_MODEL_PATH = os.environ.get(
    "MINICPM_MODEL_DIR",
    os.path.join(PROJECT_ROOT, "models/openbmb/MiniCPM-o-2_6")
)
LORA_PATH = os.environ.get(
    "MINICPM_LORA_PATH",
    os.path.join(PROJECT_ROOT, "runs/minicpm_o_lora_a100/v3-20260308-015640/checkpoint-1875")
)

VIDEO_BASE_DIR = os.environ.get(
    "VIDEO_BASE_DIR",
    os.path.join(PROJECT_ROOT, "videos/douyin/upload_pack")
)
OUTPUT_EVAL_DIR = os.environ.get(
    "MINICPM_EVAL_DIR",
    os.path.join(PROJECT_ROOT, "outputs/scaling_eval")
)
os.makedirs(OUTPUT_EVAL_DIR, exist_ok=True)

TEST_FILES = {
    "MainTest": os.environ.get(
        "MINICPM_TEST_CSV",
        os.path.join(PROJECT_ROOT, "splits/split_v3/test_main.csv")
    )
}

SYSTEM_PROMPT = "你是一位资深的计算社会科学研究员。请观看视频画面并聆听声音，依据《短视频声誉编码手册》进行分类（0-4）。注意识别BGM滥用现象，客观判断真实政务意图。"
USER_PROMPT = "请对该视频的政务宣传意图进行分类（输出0-4的数字）。"

def get_video_chunk_content(video_path, max_units=16):
    try:
        video = VideoFileClip(video_path)
        if video.audio is not None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
                temp_audio_file_path = temp_audio_file.name
                video.audio.write_audiofile(temp_audio_file_path, codec="pcm_s16le", fps=16000, logger=None)
                audio_np, sr = librosa.load(temp_audio_file_path, sr=16000, mono=True)
        else:
            sr, audio_np = 16000, np.zeros(int(math.ceil(video.duration) * 16000), dtype=np.float32)
        
        total_seconds = math.ceil(video.duration)
        indices = np.linspace(0, total_seconds - 1, max_units, dtype=int) if total_seconds > max_units else range(total_seconds)
        contents = []
        for i in indices:
            frame = video.get_frame(min(i + 1, video.duration - 0.1))
            image = Image.fromarray((frame).astype(np.uint8))
            audio_chunk = audio_np[sr*i : sr*(i+1)]
            contents.extend(["<unit>", image, audio_chunk])
        video.close()
        return contents
    except Exception as e:
        print(f"\n⚠️ 解码失败 {video_path}: {e}", flush=True)
        return None

def evaluate():
    print(f"⏳ 扫描视频...", flush=True)
    path_map = {}
    for root, _, files in os.walk(VIDEO_BASE_DIR):
        for f in files:
            if f.lower().endswith(('.mp4', '.avi', '.mov')):
                path_map[os.path.splitext(f)[0]] = os.path.join(root, f)

    print(f"🚀 加载模型...", flush=True)
    model = AutoModel.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, init_vision=True, init_audio=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, LORA_PATH)

    for test_name, csv_path in TEST_FILES.items():
        df = pd.read_csv(csv_path)
        csv_save_path = os.path.join(OUTPUT_EVAL_DIR, "eval_minicpm_o_lora_predictions.csv")
        
        # 🌟 核心：断点续评加载逻辑
        processed_vids = set()
        y_true_all, y_pred_all = [], []
        if os.path.exists(csv_save_path):
            existing_df = pd.read_csv(csv_save_path)
            processed_vids = set(existing_df['video_id'].astype(str).str.strip().tolist())
            y_true_all = existing_df['gold_label'].tolist()
            y_pred_all = existing_df['pred_label'].tolist()
            print(f"🔄 检测到断点！已跳过 {len(processed_vids)} 条，当前 Acc: {accuracy_score(y_true_all, y_pred_all):.4f}", flush=True)

        for idx, row in df.iterrows():
            vid = str(row.get('video_id', '')).strip()
            if vid in processed_vids: continue

            label = int(row['gold_label'] if 'gold_label' in df.columns else row['final_code'])
            video_path = path_map.get(vid)
            pred = 0 
            if video_path:
                try:
                    contents = get_video_chunk_content(video_path)
                    if contents:
                        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": contents + [USER_PROMPT]}]
                        with torch.no_grad():
                            response = model.chat(msgs=msgs, tokenizer=tokenizer, sampling=False, max_new_tokens=64)
                        match = re.search(r"['\"]?label['\"]?\s*:\s*['\"]?(\d+)['\"]?", response, re.IGNORECASE)
                        pred = int(match.group(1)) if match else int(re.findall(r'\d', response)[-1]) if re.findall(r'\d', response) else 0
                except: pred = 0
            
            y_true_all.append(label)
            y_pred_all.append(pred)
            pd.DataFrame([{"video_id": vid, "gold_label": label, "pred_label": pred}]).to_csv(csv_save_path, mode='a', index=False, header=not os.path.exists(csv_save_path), encoding="utf-8-sig")
            
            if (len(y_true_all)) % 10 == 0:
                print(f"📈 [Progress] {len(y_true_all)}/{len(df)} | Acc: {accuracy_score(y_true_all, y_pred_all):.4f}", flush=True)
            torch.cuda.empty_cache()

if __name__ == "__main__":
    evaluate()
