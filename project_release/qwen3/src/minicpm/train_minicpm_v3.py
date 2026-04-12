#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

# ================= 0. 核心框架补丁 (终极暴力解锁版) =================
import torch
import transformers
from transformers import TrainingArguments, Seq2SeqTrainingArguments
from transformers import Trainer, Seq2SeqTrainer
from transformers.processing_utils import ProcessorMixin

# 🌟 0. 修复 PyTorch 2.6+ 默认限制导致无法读取优化器进度 (断点续传克星) 的报错
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load
print("🔓 已成功强制放行 PyTorch torch.load 的安全反序列化限制！")

# 1. 修复 group_by_length 缺失报错
for cls in [TrainingArguments, Seq2SeqTrainingArguments]:
    if not hasattr(cls, 'group_by_length'):
        setattr(cls, 'group_by_length', False)
    if not hasattr(cls, 'length_column_name'):
        setattr(cls, 'length_column_name', "length")

# 2. 彻底破解 ms-swift 和 transformers 所有的 tokenizer 只读锁
try:
    from swift.trainers.mixin import SwiftMixin
    
    def _dummy_get(self):
        return self.__dict__.get('processing_class', None)
    def _dummy_set(self, val):
        self.__dict__['processing_class'] = val
        self.__dict__['tokenizer'] = val
    
    for target_class in [SwiftMixin, Trainer, Seq2SeqTrainer]:
        if hasattr(target_class, 'tokenizer'):
            setattr(target_class, 'tokenizer', property(_dummy_get, _dummy_set))
except Exception as e:
    pass

# 3. 修复 Processor 的 tokenizer 锁
if hasattr(ProcessorMixin, 'tokenizer') and isinstance(getattr(ProcessorMixin, 'tokenizer'), property):
    orig_fget = ProcessorMixin.tokenizer.fget
    def _set_processor_tokenizer(self, value):
        if hasattr(self, 'kwargs'):
            self.kwargs['tokenizer'] = value
        self.__dict__['tokenizer'] = value
    ProcessorMixin.tokenizer = property(orig_fget, _set_processor_tokenizer)

# ================= 1. 物理环境与 A800 满血配置 =================
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

if "NPROC_PER_NODE" in os.environ:
    del os.environ["NPROC_PER_NODE"]

os.environ.setdefault("FPS_MAX_FRAMES", "64")
os.environ.setdefault("MAX_RATIO", "4")
os.environ.setdefault("USE_AUDIO_IN_VIDEO", "True")

from swift.llm.argument.train_args import TrainArguments as SwiftTrainArguments
from swift.llm.train.sft import SwiftSft

# ================= 2. 路径定义 =================
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/root/autodl-tmp/project_douyin_mm")
MODEL_DIR = os.environ.get(
    "MINICPM_MODEL_DIR",
    f"{PROJECT_ROOT}/models/openbmb/MiniCPM-o-2_6"
)
DATA_PATH = os.environ.get(
    "MINICPM_TRAIN_DATA",
    f"{PROJECT_ROOT}/splits/split_v3/swift_train.jsonl"
)
OUTPUT_DIR = os.environ.get(
    "MINICPM_OUTPUT_DIR",
    f"{PROJECT_ROOT}/runs/minicpm_o_lora_a100"
)
RESUME_FROM_CHECKPOINT = os.environ.get("MINICPM_RESUME_FROM_CHECKPOINT", "").strip()

os.makedirs(f"{PROJECT_ROOT}/runs", exist_ok=True)

def main():
    no_split_classes = ["Resampler", "MiniCPMOModel", "MiniCPMOLayer"]

    args_kwargs = dict(
        model=MODEL_DIR,
        dataset=[DATA_PATH],
        output_dir=OUTPUT_DIR,
        train_type="lora",
        target_modules=["all-linear"],
        quant_bits=4,
        bf16=True,
        
        max_length=24000,  
        
        model_kwargs={
            "init_vision": True,  
            "init_audio": True,   
            "init_tts": False,    
            "no_split_module_classes": no_split_classes,
            "attn_implementation": "flash_attention_2"  
        },
        
        num_train_epochs=3,
        learning_rate=2e-5,
        
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4, 
        
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        
        logging_steps=5,
        save_steps=100,
        save_total_limit=3
    )

    if RESUME_FROM_CHECKPOINT:
        args_kwargs["resume_from_checkpoint"] = RESUME_FROM_CHECKPOINT

    args = SwiftTrainArguments(**args_kwargs)

    if RESUME_FROM_CHECKPOINT:
        print(f"🚀 [CSS-Project] 启动 MiniCPM-o 2.6 多模态微调（续训: {RESUME_FROM_CHECKPOINT}）...")
    else:
        print("🚀 [CSS-Project] 启动 MiniCPM-o 2.6 多模态微调（从头开始）...")
    
    try:
        trainer = SwiftSft(args)
        trainer.main()
        print("🎉 恭喜！A100/A800 微调任务已圆满完成！")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ 训练发生内部错误: {str(e)}")

if __name__ == "__main__":
    main()
