#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import re

# ================= 1. 配置路径 =================
PROJECT_ROOT = "/root/autodl-tmp/project_douyin_mm"
# 确保这里是你保存的那个 nohup 日志文件的名字
LOG_FILE = os.path.join(PROJECT_ROOT, "eval_final.log") 
CSV_FILE = os.path.join(PROJECT_ROOT, "outputs/scaling_eval/eval_minicpm_o_lora_predictions.csv")

def rescue_data():
    broken_vids = set()
    
    print(f"🔍 [1/3] 正在扫描日志文件: {LOG_FILE}")
    if not os.path.exists(LOG_FILE):
        print(f"❌ 找不到日志文件，请确认路径或文件名是否正确！")
        return

    # 1. 像法医一样扫描日志，提取所有报错的视频 ID
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            # 精准捕获：⚠️ 推理异常 4170: Expecting property name...
            # 使用 [\w\-]+ 是为了兼容纯数字ID(4170)或带字母横杠的ID
            match = re.search(r"⚠️ 推理异常 ([\w\-]+):", line)
            if match:
                broken_vids.add(str(match.group(1)).strip())

    print(f"🎯 找到了 {len(broken_vids)} 个因 JSON 解析报错而冤死的视频 ID。")

    if not broken_vids:
        print("🎉 没有发现需要修复的 ID，或者报错提取失败。")
        return

    print(f"\n🧹 [2/3] 正在清理 CSV 文件: {CSV_FILE}")
    if not os.path.exists(CSV_FILE):
        print(f"❌ 找不到 CSV 文件！")
        return

    # 2. 读取现有的 CSV
    df = pd.read_csv(CSV_FILE)
    initial_len = len(df)
    
    # 确保类型一致，方便精准匹配 (防范整数和字符串的匹配bug)
    df['video_id_str'] = df['video_id'].astype(str).str.strip()
    
    # 只保留【不在】报错列表里的健康数据
    df_cleaned = df[~df['video_id_str'].isin(broken_vids)].copy()
    df_cleaned.drop(columns=['video_id_str'], inplace=True) # 删掉辅助列

    # 3. 覆盖保存
    print(f"\n💾 [3/3] 正在保存洗净后的数据...")
    df_cleaned.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")
    
    deleted_count = initial_len - len(df_cleaned)
    print(f"=========================================")
    print(f"✅ 清理大功告成！数据统计：")
    print(f"   - 清理前原有记录 : {initial_len} 条")
    print(f"   - 成功剔除错案   : {deleted_count} 条")
    print(f"   - 保留健康记录   : {len(df_cleaned)} 条")
    print(f"=========================================")
    print("🚀 下一步：请修改主评估脚本中的正则提取代码，然后重新运行主脚本！")

if __name__ == "__main__":
    rescue_data()