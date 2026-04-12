#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
原生全模态端到端推理脚本 (Local End-to-End Omni-modal Inference)
模型: Qwen3-Omni-30B-A3B-Instruct (8-bit 量化版)
环境: AutoDL 双 RTX 4090 (48GB VRAM)
特点: 模型权重定向下载至数据盘 models 目录，彻底规避系统盘爆满及 BGM 滥用误判。
"""

import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

# ================= 1. 下载并加载 Qwen3-Omni =================
print("⏳ 正在从魔搭社区下载 Qwen3-Omni-30B 模型权重...")
print("💡 模型将被安全存放在数据盘: /root/autodl-tmp/models/ (约需15-20分钟)")
MODEL_ID = "qwen/Qwen3-Omni-30B-A3B-Instruct"
# 🌟 这里已经修改为你指定的数据盘 models 专属文件夹
model_dir = snapshot_download(MODEL_ID, cache_dir='/root/autodl-tmp/models/')

print("⏳ 正在开启 8-bit 量化，并将模型加载到两张 4090 显卡中...")
# 
# 核心魔法：将 30B 模型的显存占用从 60GB 压缩至 32GB 左右
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)

processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

# device_map="auto" 会自动将 32GB 的负载均匀分配给你的两张显卡
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    quantization_config=quantization_config,
    trust_remote_code=True
)
model.eval()
print("✅ Qwen3-Omni 全模态大脑加载完毕！开始原生音视频联合推理。")

# ================= 2. 核心提示词与路径配置 =================
# 视频存放目录与结果输出路径
VIDEO_DIR = "/root/autodl-tmp/project_douyin_mm/videos/" 
OUTPUT_CSV = "/root/autodl-tmp/project_douyin_mm/qwen3_omni_local_results.csv"

# 针对政务短视频生态深化的全模态 Prompt
SYSTEM_PROMPT = """你是一位资深的计算社会科学研究员。
请观看视频画面并聆听声音，严格依据《短视频声誉编码手册》进行意图分类（0-4）。

【核心警告：BGM滥用素养】
中国政务短视频及引流视频(0类)常滥用“激昂/悲壮”音乐。请你综合评估“客观画面”、“字幕”和“声音”。只有当视觉画面本身具备严肃政务属性时，音频的情感动员功能才有效；绝不可仅因配乐激昂就盲目判定为1或2类。

【编码手册】：
0 (引流)：无政务目的、搞笑、引流、无意义混剪。
1 (绩效)：交付核心职能、取得成就、保家卫国。
2 (道德)：奉献牺牲、军民情、坚韧不拔。
3 (程序)：军队纪律、仪式、正义之师。
4 (技术)：武器展示、演习实战、专业科普。

请以 JSON 格式返回：
{
    "final_label": 整数 (0-4),
    "reasoning": "说明你如何综合画面与声音做出判断（50字以内）"
}"""

# ================= 3. 推理与解析逻辑 =================
def analyze_video(video_path):
    # 构建 Qwen3-Omni 专属的多模态消息体
    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": SYSTEM_PROMPT}
        ]}
    ]
    
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # 处理器会自动从 mp4 中提取视觉帧与音频波形
        inputs = processor(text=[text], videos=[video_path], padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        
        # 鲁棒的 JSON 清理与解析
        res_content = output_text.replace('```json', '').replace('```', '').strip()
        res_json = json.loads(res_content)
        return int(res_json.get("final_label", 0)), res_json.get("reasoning", "Success")
        
    except Exception as e:
        return -1, f"Error: {str(e)}"

# ================= 4. 主执行流 =================
def main():
    if not os.path.exists(VIDEO_DIR):
        print(f"❌ 找不到视频目录：{VIDEO_DIR}")
        return

    all_videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
    print(f"📂 共发现 {len(all_videos)} 个待处理视频。")
    
    # 智能断点续传机制
    processed_ids = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            df_old = pd.read_csv(OUTPUT_CSV)
            processed_ids = set(df_old['video_id'].astype(str).tolist())
            print(f"📦 发现历史进度，已自动跳过 {len(processed_ids)} 个已裁决样本。")
        except:
            pass

    videos_todo = [v for v in all_videos if v.split('.')[0] not in processed_ids]

    for video_name in tqdm(videos_todo, desc="Qwen3-Omni Inferencing"):
        video_id = video_name.split('.')[0]
        video_path = os.path.join(VIDEO_DIR, video_name)
        
        final_l, reason = analyze_video(video_path)
        
        record = {
            "video_id": video_id,
            "qwen3_omni_label": final_l,
            "reasoning": reason
        }
        
        # 单条实时落盘，防止中断丢失
        pd.DataFrame([record]).to_csv(
            OUTPUT_CSV, mode='a', index=False, 
            header=not os.path.exists(OUTPUT_CSV), encoding='utf-8-sig'
        )

    print(f"\n🎉 裁决全部完成！结果已保存至: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()