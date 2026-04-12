#!/bin/bash

# 1. 显存优化环境变量
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_VISIBLE_DEVICES=0

# 2. Qwen3 特有的视频解析优化
export FPS_MAX_FRAMES=64
export MAX_PIXELS=230400
export MAX_RATIO=4

# 3. 启动 Swift SFT (这是 v3 版本的标准入口)
# 我们直接在这里配置所有参数，效果和 Python 脚本完全一样，但不会报 Import 错误
swift sft \
    --model "models/qwen/Qwen3-Omni-30B-A3B-Instruct" \
    --dataset "splits/split_v3/swift_train.jsonl" \
    --output_dir "runs/qwen3_omni_lora_a800" \
    --train_type "lora" \
    --target_modules "all-linear" \
    --quant_bits 4 \
    --bf16 True \
    --max_length 8192 \
    --model_kwargs '{"disable_talker": true, "use_audio_in_video": true, "trust_remote_code": true, "attn_implementation": "flash_attention_2"}' \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --logging_steps 5 \
    --save_steps 100 \
    --save_total_limit 3 \
    --template "qwen" \
    --system "你是一位资深的计算社会科学研究员。请观看视频画面并聆听声音，依据《短视频声誉编码手册》进行分类（0-4）。注意识别BGM滥用现象，客观判断真实政务意图。"