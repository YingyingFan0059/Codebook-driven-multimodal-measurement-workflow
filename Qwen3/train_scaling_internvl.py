#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
InternVL2-8B LoRA fine-tuning for short-video multimodal classification.
Updated for: 5-class system & Scaling Law (N=400 to 3000).
Core model logics strictly preserved.
"""

import os
import glob
import subprocess
import gc
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from transformers import (
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import LoraConfig, get_peft_model

# ================== 1. Hard env & Paths ==================
PROJECT_ROOT = "/root/autodl-tmp/project_douyin_mm"
os.environ["HF_HOME"] = f"{PROJECT_ROOT}/cache/hf"
os.environ["TMPDIR"] = f"{PROJECT_ROOT}/tmp"

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ.pop("TRANSFORMERS_CACHE", None)

BASE_VIDEO_DIR = os.path.join(PROJECT_ROOT, "videos/douyin/upload_pack")
SPLITS_DIR = os.path.join(PROJECT_ROOT, "splits/split_v3") # 对齐最新 split_v3
MODEL_ID = os.path.join(PROJECT_ROOT, "models/InternVL2-8B")
CACHE_DIR = os.path.join(PROJECT_ROOT, "runs/scaling_experiments/frames_cache_internvl")
os.makedirs(CACHE_DIR, exist_ok=True)

# 🌟 核心修改 1：对齐 5分类 Prompt
INSTR = (
    "你是一位严格的计算社会科学编码员。请将该政务短视频进行唯一分类。\n"
    "类别定义：0 OTHER_OR_UNCLEAR (其他)；1 PERFORMATIVE (绩效)；2 MORAL (道德)；3 PROCEDURAL (程序)；4 TECHNICAL (技术)。\n"
    "输出要求：仅返回JSON对象，包含键'label' (整数型)。不要包含任何多余解释。"
)

# 🌟 核心修改 2：强健的物理路径自动索引 (兼容最新 CSV)
def build_video_path_index(base_dir):
    print(f"⏳ 正在扫描视频物理路径 (InternVL)...")
    path_map = {}
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(('.mp4', '.avi', '.mov')):
                vid_id = os.path.splitext(f)[0]
                path_map[vid_id] = os.path.join(root, f)
    print(f"✅ 扫描完成，共索引 {len(path_map)} 个视频。")
    return path_map

GLOBAL_VIDEO_PATHS = build_video_path_index(BASE_VIDEO_DIR)

# ================== 2. 原汁原味的 InternVL 抽帧与预处理 (未改动) ==================
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
        return frames

    scale_expr = (
        f"scale='if(gt(iw*ih,{max_pixels}),trunc(iw*sqrt({max_pixels}/(iw*ih))/2)*2,iw)'"
        f":'if(gt(iw*ih,{max_pixels}),trunc(ih*sqrt({max_pixels}/(iw*ih))/2)*2,ih)'"
    )
    out_pattern = os.path.join(vid_cache, "frame_%06d.jpg")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", video_path, "-vf", f"fps={fps},{scale_expr}", "-q:v", "2", out_pattern,
    ]

    try:
        p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if p.returncode != 0: return []
    except Exception:
        return []

    frames = sorted(glob.glob(os.path.join(vid_cache, "frame_*.jpg")))
    if not frames: return []

    if len(frames) > max_frames:
        idx = [int(round(i)) for i in torch.linspace(0, len(frames) - 1, steps=max_frames).tolist()]
        frames = [frames[i] for i in idx]

    return frames

def load_frames_as_pil(frame_paths: List[str]):
    from PIL import Image
    imgs = []
    for p in frame_paths:
        try:
            with Image.open(p) as im:
                imgs.append(im.convert("RGB"))
        except Exception:
            continue
    if not imgs: return []
    if len(imgs) % 2 != 0: imgs.append(imgs[-1].copy())
    return imgs

def build_transform(input_size: int = 448):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if getattr(img, "mode", "RGB") != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

# ================== 3. Dataset & Collator ==================
class CsvVideoDataset(Dataset):
    def __init__(self, csv_path: str):
        self.items: List[Dict[str, Any]] = []
        df = pd.read_csv(csv_path, encoding="utf-8-sig")

        for _, row in df.iterrows():
            # 兼容老版本 rel_path 和新版本 video_id
            vid = str(row.get('video_id', '')).strip()
            if not vid and 'rel_path' in df.columns:
                vid = os.path.splitext(os.path.basename(str(row["rel_path"])))[0]
                
            if vid in GLOBAL_VIDEO_PATHS:
                self.items.append({
                    "video_id": vid,
                    "label": int(row['gold_label'] if 'gold_label' in df.columns else row['final_code']),
                    "video_path": GLOBAL_VIDEO_PATHS[vid],
                })

    def __len__(self): return len(self.items)
    def __getitem__(self, idx: int): return self.items[idx]

@dataclass
class InternVLDataCollator:
    tokenizer: AutoTokenizer
    frames_cache_dir: str
    num_image_token: int = 256
    fps: float = 0.5
    max_frames: int = 8
    max_pixels: int = 360 * 480
    input_size: int = 448

    def __post_init__(self):
        self.transform = build_transform(self.input_size)

    def build_text_inputs_internvl_contract(self, num_images: int, prompt: str, answer: Optional[str] = None):
        query = "<image>\n" + prompt
        image_tokens = "<img>" + ("<IMG_CONTEXT>" * (self.num_image_token * num_images)) + "</img>"
        query = query.replace("<image>", image_tokens, 1)

        base = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        prompt_ids = self.tokenizer.encode(base, add_special_tokens=True)

        text = base
        if answer is not None:
            text += f"{answer}<|im_end|>\n"

        full_ids = self.tokenizer.encode(text, add_special_tokens=True)
        return full_ids, prompt_ids

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list, attention_mask_list, labels_list = [], [], []
        pixel_values_list, image_flags_list = [], []

        for ex in batch:
            frame_paths = extract_frames_ffmpeg_cached(
                ex["video_path"], self.frames_cache_dir, self.fps, self.max_frames, self.max_pixels
            )
            if not frame_paths: continue

            imgs = load_frames_as_pil(frame_paths)
            if not imgs: continue

            pixel_values = torch.stack([self.transform(img) for img in imgs])
            num_images = pixel_values.shape[0]
            image_flags = torch.ones((num_images, 1), dtype=torch.long)

            answer = f"{{\"label\": {int(ex['label'])}}}"
            full_ids, prompt_ids = self.build_text_inputs_internvl_contract(num_images, INSTR, answer)

            full_input_ids = torch.tensor(full_ids, dtype=torch.long)
            labels = full_input_ids.clone()
            labels[: len(prompt_ids)] = -100

            input_ids_list.append(full_input_ids)
            attention_mask_list.append(torch.ones_like(full_input_ids))
            labels_list.append(labels)

            pixel_values_list.append(pixel_values)
            image_flags_list.append(image_flags)

        if self.tokenizer.pad_token_id is None:
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
        else:
            pad_id = self.tokenizer.pad_token_id

        if len(input_ids_list) == 0:
            dummy_ids = torch.tensor([pad_id, pad_id], dtype=torch.long)
            return {
                "input_ids": dummy_ids.unsqueeze(0),
                "attention_mask": torch.ones_like(dummy_ids).unsqueeze(0),
                "labels": (torch.ones_like(dummy_ids) * -100).unsqueeze(0),
                "pixel_values": torch.zeros((1, 3, self.input_size, self.input_size), dtype=torch.float32),
                "image_flags": torch.zeros((1, 1), dtype=torch.long),
            }

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)

        pixel_values = torch.cat(pixel_values_list, dim=0)
        image_flags = torch.cat(image_flags_list, dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_flags": image_flags,
        }

class InternVLLoraWrapper(nn.Module):
    def __init__(self, peft_model: nn.Module):
        super().__init__()
        self.model = peft_model
        self.config = getattr(peft_model, "config", None)
        self.warnings_issued = getattr(peft_model, "warnings_issued", {})

    def forward(self, *args, **kwargs):
        for k in ["inputs_embeds", "num_items_in_batch", "return_loss"]:
            if k in kwargs: kwargs.pop(k)
        return self.model(*args, **kwargs)

    def save_pretrained(self, save_directory, **kwargs):
        self.model.save_pretrained(save_directory, **kwargs)

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(**kwargs)

def sanity_check_alignment(peft_model, collator: InternVLDataCollator, train_ds: Dataset):
    base = peft_model.get_base_model()
    assert base.img_context_token_id is not None, "img_context_token_id is None. MUST set before training."
    ex = train_ds[0]
    batch = collator([ex])
    input_ids, pixel_values = batch["input_ids"], batch["pixel_values"]
    selected = (input_ids.reshape(-1) == base.img_context_token_id)
    n_selected = int(selected.sum().item())
    expected = int(pixel_values.shape[0] * base.num_image_token)
    assert n_selected == expected, f"Mismatch: IMG_CONTEXT tokens={n_selected} vs expected={expected}"
    print("✅ SANITY PASS: IMG_CONTEXT alignment OK")

# ================== 4. Scaling Law 循环管线 ==================
def train_scaling_internvl(train_size: int):
    print(f"\n{'='*60}\n🚀 Start InternVL2-8B Scaling Law: N={train_size}\n{'='*60}")

    train_csv = os.path.join(SPLITS_DIR, f"train_{train_size}.csv")
    out_dir = os.path.join(PROJECT_ROOT, f"runs/scaling_experiments/internvl_lora_5class_{train_size}")
    log_dir = os.path.join(PROJECT_ROOT, "runs/internvl_logs")
    tmp_run_dir = os.path.join(PROJECT_ROOT, "runs/internvl_tmp")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tmp_run_dir, exist_ok=True)

    set_seed(42)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=dtype, trust_remote_code=True, device_map="auto")

    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
    if hasattr(base_model, "config") and hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["attention.wqkv", "attention.wo", "feed_forward.w1", "feed_forward.w2", "feed_forward.w3"],
        lora_dropout=0.05, bias="none",
    )
    peft_model = get_peft_model(base_model, lora_cfg)

    # 关键机制保留
    img_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    peft_model.get_base_model().img_context_token_id = img_id
    
    base = peft_model.get_base_model()
    if hasattr(base, "enable_input_require_grads"): base.enable_input_require_grads()
    if hasattr(base, "language_model") and hasattr(base.language_model, "enable_input_require_grads"):
        base.language_model.enable_input_require_grads()
    if hasattr(peft_model, "enable_input_require_grads"): peft_model.enable_input_require_grads()

    def make_inputs_require_grad(module, input, output):
        if isinstance(output, torch.Tensor): output.requires_grad_(True)
        elif isinstance(output, tuple): output[0].requires_grad_(True)
            
    try:
        embed_tokens = base.language_model.get_input_embeddings()
        embed_tokens.register_forward_hook(make_inputs_require_grad)
    except Exception:
        pass

    safe_model = InternVLLoraWrapper(peft_model)
    train_ds = CsvVideoDataset(train_csv)
    model_num_image_token = getattr(peft_model.get_base_model(), "num_image_token", 256)

    collator = InternVLDataCollator(
        tokenizer=tokenizer, frames_cache_dir=CACHE_DIR,
        num_image_token=int(model_num_image_token),
        fps=0.5, max_frames=8, max_pixels=360 * 480, input_size=448,
    )

    sanity_check_alignment(peft_model, collator, train_ds)

    args = TrainingArguments(
        output_dir=tmp_run_dir, per_device_train_batch_size=1, gradient_accumulation_steps=16,
        learning_rate=1e-4, num_train_epochs=3, save_strategy="no",
        bf16=(dtype == torch.bfloat16), fp16=(dtype == torch.float16),
        remove_unused_columns=False, logging_dir=log_dir, logging_steps=5,
        dataloader_num_workers=2, report_to="none",
    )

    trainer = Trainer(model=safe_model, args=args, train_dataset=train_ds, data_collator=collator)
    trainer.train()

    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"✅ N={train_size} Done! Saved to: {out_dir}")

    # 🌟 核心修改 3：必须彻底清空显存，应对 14 轮缩放循环
    del trainer, safe_model, peft_model, base_model, tokenizer, train_ds, collator
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    import sys
    
    # 🌟 灵活设置：直接将默认最佳规模焊死为 2200
    OPTIMAL_N = 2200 
    
    # 允许通过命令行参数覆盖默认值 (例如: python train_scaling_internvl.py 2000)
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        OPTIMAL_N = int(sys.argv[1])
        
    print(f"🔔 启动 InternVL2 单点最佳规模微调任务 (N={OPTIMAL_N})...")
    train_scaling_internvl(OPTIMAL_N)
        
    print(f"\n🎉 InternVL2 最佳规模 (N={OPTIMAL_N}) 5分类微调圆满完成！")