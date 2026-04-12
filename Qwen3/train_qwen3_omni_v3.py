#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
 Qwen3-Omni-30B  微调训练脚本 (参数对齐 Qwen2-VL-7B 成功配置)
 Swift 4.0.1 | Template: qwen3_omni
==========================================================================

对齐策略 (基于 Qwen2-VL-7B 69.46% 的成功经验):
  - lora_rank:    16 (与Qwen2一致)
  - lora_alpha:   32 (与Qwen2一致)
  - lora_targets: q/k/v/o_proj (与Qwen2一致)
  - epochs:       3
  - lr:           1e-4
  - FPS:          8帧 (与Qwen2一致)
  - Prompt:       包含明确类别定义
  - Response:     JSON格式 {"label": X}

使用前:
  1. 先运行 fix_training_data.py 修正训练数据格式
  2. 再运行本脚本
"""

import os
import sys
import json

# ==================== 1. 环境变量 ====================
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 对齐 Qwen2-VL-7B: 8帧
os.environ["FPS_MAX_FRAMES"] = "8"
os.environ["MAX_PIXELS"] = "150000"
os.environ["MAX_RATIO"] = "4"

# ==================== 2. Swift 4.0.1 导入 ====================
try:
    from swift.pipelines.train.sft import sft_main
    from swift.arguments import SftArguments
    print("[INFO] Swift 4.0.1 模块加载成功")
except ImportError as e:
    print(f"[ERROR] Swift 导入失败: {e}")
    sys.exit(1)

# ==================== 3. 路径定义 ====================
PROJECT_ROOT = "/root/autodl-tmp/project_douyin_mm"
MODEL_DIR    = f"{PROJECT_ROOT}/models/qwen/Qwen3-Omni-30B-A3B-Instruct"
TRAIN_DATA   = f"{PROJECT_ROOT}/splits/split_v3/swift_train_2200.jsonl"
EVAL_DATA    = f"{PROJECT_ROOT}/splits/split_v3/swift_eval.jsonl"
OUTPUT_DIR   = f"{PROJECT_ROOT}/runs/qwen3_omni_lora_aligned_v2"
os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)

# ==================== 4. 数据验证 ====================
def validate_data(data_path):
    if not os.path.exists(data_path):
        print(f"[ERROR] 数据文件不存在: {data_path}")
        print(f"[ERROR] 请先运行: python fix_training_data.py")
        sys.exit(1)

    with open(data_path, "r", encoding="utf-8") as f:
        first = json.loads(f.readline())

    print(f"\n[DATA] 第一条样本:")
    print(f"  keys: {list(first.keys())}")
    print(f"  system: {first.get('system','')[:50]}...")
    print(f"  query: {first.get('query','')[:50]}...")
    print(f"  response: {first.get('response','')}")
    print(f"  videos: {first.get('videos',[])[0][:50] if first.get('videos') else 'N/A'}...")

    with open(data_path, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f)
    print(f"  总条数: {total}\n")
    return total


# ==================== 5. 主函数 ====================
def main():
    print("=" * 60)
    print("  Qwen3-Omni-30B 微调 (对齐Qwen2-VL-7B成功参数)")
    print("=" * 60)

    total_samples = validate_data(TRAIN_DATA)

    # 验证集
    val_dataset_arg = None
    eval_kwargs = {}
    if os.path.exists(EVAL_DATA):
        val_dataset_arg = [EVAL_DATA]
        eval_kwargs["eval_strategy"] = "steps"
        eval_kwargs["eval_steps"] = 50
        print(f"[INFO] 验证集: {EVAL_DATA}")
    else:
        print(f"[WARN] 未找到验证集\n")

    # 构建参数 — 完全对齐 Qwen2-VL-7B 成功配置
    sft_kwargs = dict(
        # ===== 模型 & 数据 =====
        model=MODEL_DIR,
        dataset=[TRAIN_DATA],
        output_dir=OUTPUT_DIR,

        # ===== 模板 (已验证) =====
        template="qwen3_omni",

        # ===== LoRA — 对齐 Qwen2 成功配置 =====
        train_type="lora",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 对齐Qwen2
        lora_rank=16,          # 对齐Qwen2
        lora_alpha=32,         # 对齐Qwen2
        lora_dropout=0.05,     # 对齐Qwen2
        quant_bits=4,          # A800必须, Qwen2是bf16全精度但7B小很多
        bf16=True,

        # ===== 序列长度 =====
        max_length=4096,       # 8帧视频token不会太长

        # ===== 模型加载 =====
        model_kwargs={
            "disable_talker": True,
            "use_audio_in_video": True,
            "trust_remote_code": True,
            "attn_implementation": "flash_attention_2",
        },

        # ===== 优化器 — 对齐Qwen2 =====
        num_train_epochs=3,
        learning_rate=1e-4,    # 对齐Qwen2
        warmup_ratio=0.05,
        weight_decay=0.01,

        # ===== 显存管理 =====
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,  # 对齐Qwen2
        gradient_checkpointing=True,

        # ===== 日志 & 保存 =====
        logging_steps=10,      # 对齐Qwen2
        save_steps=100,
        save_total_limit=3,
        dataloader_num_workers=0,
    )

    if val_dataset_arg:
        sft_kwargs["val_dataset"] = val_dataset_arg
        sft_kwargs.update(eval_kwargs)

    # 构建 args
    try:
        args = SftArguments(**sft_kwargs)
    except TypeError as e:
        error_msg = str(e)
        print(f"[WARN] 参数兼容: {error_msg}")
        if "val_dataset" in error_msg:
            sft_kwargs.pop("val_dataset", None)
            sft_kwargs.pop("eval_strategy", None)
            sft_kwargs.pop("eval_steps", None)
        if "lora_rank" in error_msg:
            sft_kwargs["lora_r"] = sft_kwargs.pop("lora_rank", 16)
        if "lora_alpha" in error_msg:
            sft_kwargs["lora_target_alpha"] = sft_kwargs.pop("lora_alpha", 32)
        args = SftArguments(**sft_kwargs)

    # 参数确认
    print("\n" + "=" * 60)
    print("  参数确认 (对齐 Qwen2-VL-7B)")
    print("=" * 60)
    print(f"  template:        qwen3_omni")
    print(f"  lora_targets:    q/k/v/o_proj")
    print(f"  lora_rank:       16")
    print(f"  lora_alpha:      32")
    print(f"  learning_rate:   1e-4")
    print(f"  max_length:      6144")
    print(f"  FPS_MAX_FRAMES:  8")
    print(f"  quant_bits:      4")
    print(f"  epochs:          3")
    print(f"  effective_batch: 8")
    print(f"  数据:            {TRAIN_DATA}")
    print(f"  输出:            {OUTPUT_DIR}")
    print("=" * 60)

    # 启动训练
    print("\n[START] 启动训练...\n")
    try:
        sft_main(args)
        print("\n" + "=" * 60)
        print("  训练完成!")
        print("=" * 60)
        print(f"  模型: {OUTPUT_DIR}")
        print(f"\n  推理时参数:")
        print(f"    --template qwen3_omni")
        print(f"    --max_length 6144")
        print(f"    --quant_bits 4")
        print(f"    FPS_MAX_FRAMES=8")
        print("=" * 60)

    except Exception as e:
        import traceback
        traceback.print_exc()
        error_str = str(e).lower()
        print(f"\n[ERROR] {e}")
        if "out of memory" in error_str or "oom" in error_str:
            print("\n[OOM] 尝试:")
            print("  1. max_length: 6144 → 4096")
            print("  2. lora_rank: 16 → 8")
            print("  3. 启用 DeepSpeed ZeRO-2")


if __name__ == "__main__":
    main()