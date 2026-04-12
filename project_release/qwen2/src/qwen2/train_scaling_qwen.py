#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import subprocess
import gc
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import LoraConfig, get_peft_model

# ================= 1. 路径与环境配置 =================
RELEASE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/root/autodl-tmp/project_douyin_mm")
os.environ["HF_HOME"] = os.environ.get("HF_HOME", f"{PROJECT_ROOT}/cache/hf")
os.environ["TMPDIR"] = os.environ.get("TMPDIR", f"{PROJECT_ROOT}/tmp")
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "4")

BASE_VIDEO_DIR = os.environ.get(
    "VIDEO_BASE_DIR",
    os.path.join(PROJECT_ROOT, "videos/douyin/upload_pack")
)
SPLITS_DIR = os.environ.get(
    "QWEN2_SPLITS_DIR",
    str(RELEASE_ROOT / "splits" / "split_v3")
)
ARTIFACT_ROOT = Path(
    os.environ.get(
        "QWEN2_ARTIFACT_ROOT",
        str(RELEASE_ROOT / "artifacts" / "qwen2")
    )
)
MODEL_ID = os.environ.get(
    "QWEN2_MODEL_DIR",
    os.path.join(PROJECT_ROOT, "models/qwen/Qwen2-VL-7B-Instruct")
)
RUNS_DIR = os.environ.get(
    "QWEN2_RUNS_DIR",
    str(ARTIFACT_ROOT / "runs" / "scaling_experiments")
)
CACHE_DIR = os.environ.get(
    "QWEN2_FRAMES_CACHE_DIR",
    str(ARTIFACT_ROOT / "cache" / "frames_cache")
)
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

INSTR = (
    "你是一位严格的计算社会科学编码员。请将该政务短视频进行唯一分类。\n"
    "类别定义：0 OTHER_OR_UNCLEAR (其他)；1 PERFORMATIVE (绩效)；2 MORAL (道德)；3 PROCEDURAL (程序)；4 TECHNICAL (技术)。\n"
    "输出要求：仅返回JSON对象，包含键'label' (整数型)。不要包含任何多余解释。"
)

# ================= 2. 物理路径自动索引 =================
def build_video_path_index(base_dir):
    print(f"⏳ 正在扫描视频物理路径...")
    path_map = {}
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(('.mp4', '.avi', '.mov')):
                vid_id = os.path.splitext(f)[0]
                path_map[vid_id] = os.path.join(root, f)
    print(f"✅ 扫描完成，共索引 {len(path_map)} 个视频。")
    return path_map

GLOBAL_VIDEO_PATHS = build_video_path_index(BASE_VIDEO_DIR)

# ================= 3. 强健的底层逻辑：抽帧与组装 =================
def extract_frames_ffmpeg_cached(video_path: str, cache_dir: str, fps: float, max_frames: int, max_pixels: int) -> List[str]:
    os.makedirs(cache_dir, exist_ok=True)
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    vid_cache = os.path.join(cache_dir, video_id)
    os.makedirs(vid_cache, exist_ok=True)

    frames = sorted(glob.glob(os.path.join(vid_cache, "frame_*.jpg")))
    if len(frames) >= 2: 
        if len(frames) > max_frames:
            idx = [int(round(i)) for i in torch.linspace(0, len(frames) - 1, steps=max_frames).tolist()]
            frames = [frames[i] for i in idx]
        return [f"file://{f}" for f in frames]

    scale_expr = (
        f"scale='if(gt(iw*ih,{max_pixels}),trunc(iw*sqrt({max_pixels}/(iw*ih))/2)*2,iw)'"
        f":'if(gt(iw*ih,{max_pixels}),trunc(ih*sqrt({max_pixels}/(iw*ih))/2)*2,ih)'"
    )
    out_pattern = os.path.join(vid_cache, "frame_%06d.jpg")
    cmd = [
        "/usr/bin/ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", video_path, "-vf", f"fps={fps},{scale_expr}", "-q:v", "2", out_pattern,
    ]
    subprocess.check_call(cmd)

    frames = sorted(glob.glob(os.path.join(vid_cache, "frame_*.jpg")))
    if not frames: return []

    if len(frames) > max_frames:
        idx = [int(round(i)) for i in torch.linspace(0, len(frames) - 1, steps=max_frames).tolist()]
        frames = [frames[i] for i in idx]

    return [f"file://{f}" for f in frames]

def load_frames_as_pil(frame_paths: List[str]):
    clean_paths = [p.replace("file://", "") for p in frame_paths]
    imgs = [Image.open(p).convert("RGB") for p in clean_paths]
    if len(imgs) % 2 != 0:
        imgs.append(imgs[-1].copy())
    return imgs

def build_multimodal_inputs(processor, frames_pil, prompt_text, answer_text=None, max_length=16384):
    messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt_text}]}]
    if answer_text is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer_text}]})
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return processor(text=[chat_text], videos=[frames_pil], return_tensors="pt", truncation=True, max_length=max_length)

# ================= 4. Dataset & 降维拼接 Collator =================
class CsvVideoDataset(Dataset):
    def __init__(self, csv_path: str):
        self.items = []
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        for _, row in df.iterrows():
            vid = str(row['video_id']).strip()
            if vid in GLOBAL_VIDEO_PATHS:
                self.items.append({
                    "video_path": GLOBAL_VIDEO_PATHS[vid],
                    "label": int(row['gold_label'] if 'gold_label' in df.columns else row['final_code'])
                })

    def __len__(self): return len(self.items)
    def __getitem__(self, idx: int): return self.items[idx]

@dataclass
class MMDataCollator:
    processor: Qwen2VLProcessor
    frames_cache_dir: str
    fps: float = 0.5
    max_frames: int = 8
    max_pixels: int = 360 * 480
    max_length: int = 16384

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list, attention_mask_list, labels_list = [], [], []
        pixel_values_videos_list, video_grid_thw_list = [], []

        for ex in batch:
            frame_paths = extract_frames_ffmpeg_cached(
                ex["video_path"], self.frames_cache_dir, self.fps, self.max_frames, self.max_pixels
            )
            if not frame_paths: continue 
            
            frames_pil = load_frames_as_pil(frame_paths)
            answer = f"{{\"label\": {ex['label']}}}"

            full = build_multimodal_inputs(self.processor, frames_pil, INSTR, answer, self.max_length)
            prompt_only = build_multimodal_inputs(self.processor, frames_pil, INSTR, None, self.max_length)

            full_input_ids = full["input_ids"][0]
            labels = full_input_ids.clone()
            labels[: prompt_only["input_ids"][0].shape[0]] = -100 

            input_ids_list.append(full_input_ids)
            attention_mask_list.append(full["attention_mask"][0])
            labels_list.append(labels)

            if "pixel_values_videos" in full:
                val = full["pixel_values_videos"]
                pixel_values_videos_list.append(val.view(-1, val.shape[-1]) if val.ndim > 2 else val)
            if "video_grid_thw" in full:
                val = full["video_grid_thw"]
                video_grid_thw_list.append(val.view(-1, val.shape[-1]) if val.ndim > 2 else val)

        pad_id = self.processor.tokenizer.pad_token_id or 0
        out = {
            "input_ids": torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0),
            "labels": torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
        }
        if pixel_values_videos_list:
            out["pixel_values_videos"] = torch.cat(pixel_values_videos_list, dim=0)
            out["video_grid_thw"] = torch.cat(video_grid_thw_list, dim=0)
        return out

# ================= 5. 循环微调管线 (N=400 到 N=3000) =================
def train_scaling(train_size: int):
    print(f"\n{'='*50}\n🚀 启动微调实验: N = {train_size}\n{'='*50}")
    
    train_csv = os.path.join(SPLITS_DIR, f"train_{train_size}.csv")
    out_dir = os.path.join(RUNS_DIR, f"lora_5class_{train_size}")
    os.makedirs(out_dir, exist_ok=True)
    
    set_seed(42)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    processor = Qwen2VLProcessor.from_pretrained(MODEL_ID)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=dtype, attn_implementation="sdpa", device_map="auto"
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)

    # 🌟 强制恢复的手动 Hook：专治大模型多模态梯度断裂
    embed_tokens = model.get_input_embeddings()
    if hasattr(embed_tokens, "_forward_hooks"): 
        embed_tokens._forward_hooks.clear()
        
    def make_inputs_require_grad_and_clone(module, input, output):
        output.requires_grad_(True)
        return output.clone()
        
    embed_tokens.register_forward_hook(make_inputs_require_grad_and_clone)

    train_ds = CsvVideoDataset(train_csv)
    collator = MMDataCollator(processor=processor, frames_cache_dir=CACHE_DIR)

    args = TrainingArguments(
        output_dir=os.path.join(RUNS_DIR, "_tmp_trainer"),
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=1e-4, num_train_epochs=3, save_strategy="no",
        bf16=(dtype == torch.bfloat16), fp16=(dtype == torch.float16),
        remove_unused_columns=False, logging_steps=10
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collator)
    trainer.train()

    trainer.model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    print(f"✅ N={train_size} 完成，保存至: {out_dir}")
    
    del trainer, model, processor, train_ds, collator
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    sizes = list(range(400, 3200, 200))
    for s in sizes:
        train_scaling(s)
    print("\n🎉 Scaling Law 5分类实验全部圆满完成！")
