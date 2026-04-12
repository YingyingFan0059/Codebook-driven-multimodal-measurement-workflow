import pandas as pd
import json
import os

# 1. 读取包含 2200 个样本和正确标签(gold_label)的表格
df = pd.read_csv('/root/autodl-tmp/project_douyin_mm/outputs/scaling_eval/eval_lora_5class_2200_predictions.csv')

# 2. 视频总目录
base_video_dir = "/root/autodl-tmp/project_douyin_mm/videos/douyin/upload_pack/"

# ================= 核心修改区 =================
print("🔍 正在扫描所有子文件夹，建立视频路径索引库...")
video_path_map = {}
# 使用 os.walk 递归遍历 base_video_dir 下的所有子文件夹
for root, dirs, files in os.walk(base_video_dir):
    for file in files:
        if file.endswith('.mp4'):
            # 将文件名（如 123.mp4）作为 key，完整绝对路径作为 value
            video_path_map[file] = os.path.join(root, file)

print(f"✅ 扫描完毕！在各子文件夹中共发现 {len(video_path_map)} 个 mp4 文件。")
# ==============================================

# 3. 确保输出目录存在
output_dir = "/root/autodl-tmp/project_douyin_mm/splits/split_v3/"
os.makedirs(output_dir, exist_ok=True)
output_jsonl_path = os.path.join(output_dir, "swift_train.jsonl")

system_prompt = """你是一位资深的计算社会科学研究员。请观看视频画面并聆听声音，依据《短视频声誉编码手册》进行分类（0-4）。注意识别BGM滥用现象，客观判断真实政务意图。"""

jsonl_data = []
missing_videos = [] # 用来记录找不到的视频

for index, row in df.iterrows():
    # 假设 CSV 中的 video_id 列不带后缀，我们需要拼上 .mp4 去字典里查
    target_filename = f"{row['video_id']}.mp4"
    
    # 从刚刚建立的索引库中寻找真实路径
    if target_filename in video_path_map:
        actual_video_path = video_path_map[target_filename]
    else:
        missing_videos.append(target_filename)
        continue
        
    gold_label = str(row['gold_label'])
    
    # 4. 构造 ms-swift v3 官方推荐的标准多模态微调格式
    item = {
        "system": system_prompt,
        "query": "<video>请对该视频的政务宣传意图进行分类（输出0-4的数字）。",
        "response": gold_label,
        "videos": [actual_video_path]  # 填入查找到的真实绝对路径
    }
    jsonl_data.append(item)

# 5. 保存为 jsonl 文件
with open(output_jsonl_path, 'w', encoding='utf-8') as f:
    for item in jsonl_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"\n✅ 成功生成 {len(jsonl_data)} 条多模态微调数据！")
print(f"📁 数据集已保存至: {output_jsonl_path}")

if missing_videos:
    print(f"\n⚠️ 警告：发现 {len(missing_videos)} 个视频在任何子文件夹中都找不到！")
    print(f"可能是 CSV 中的 video_id 与文件名不一致，缺失示例: {missing_videos[:3]}")
else:
    print("\n🎉 完美！2200 个视频文件全部核对无误，真实路径已完美绑定！")