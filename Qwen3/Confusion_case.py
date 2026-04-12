#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
混淆矩阵错误案例提取器 (Confusion Matrix Case Extractor)
目标：提取 N=2200 时，Qwen2-VL 和 InternVL2 的预测对比结果，找出它们各自的“错判重灾区”。
输出：包含真实标签、双模型预测标签及正误状态的合并 CSV 文件。
"""

import os
import pandas as pd

# ================= 1. 配置与路径 =================
PROJECT_ROOT = "/root/autodl-tmp/project_douyin_mm"
# 如果你在本地运行，请将下面的路径修改为你本地存放 CSV 的实际路径，例如 "./eval_lora_5class_2200_predictions.csv"
QWEN_CSV = os.path.join(PROJECT_ROOT, "outputs/scaling_eval/eval_lora_5class_2200_predictions.csv")
INTERNVL_CSV = os.path.join(PROJECT_ROOT, "outputs/scaling_eval/eval_internvl_lora_5class_2200_predictions.csv")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "outputs/confusion_analysis_N2200.csv")

# 类别字典映射，方便人类直接阅读，不用再对着数字猜
LABEL_MAP = {
    0: "0_Other (其他/引流)",
    1: "1_Performative (绩效)",
    2: "2_Moral (道德)",
    3: "3_Procedural (程序)",
    4: "4_Technical (技术)"
}

def map_label(val):
    try:
        return LABEL_MAP[int(val)]
    except:
        return "Unknown"

def main():
    print("🔍 启动混淆矩阵错误案例分析...\n")
    
    if not os.path.exists(QWEN_CSV) or not os.path.exists(INTERNVL_CSV):
        print("❌ 找不到输入文件，请检查 QWEN_CSV 和 INTERNVL_CSV 路径是否正确！")
        return

    # 读取数据
    df_qwen = pd.read_csv(QWEN_CSV)
    df_int = pd.read_csv(INTERNVL_CSV)
    
    # 统一 ID 格式
    df_qwen['video_id'] = df_qwen['video_id'].astype(str)
    df_int['video_id'] = df_int['video_id'].astype(str)
    
    # 提取需要比对的核心列
    # 考虑到不同的表可能列名有差异，统一处理
    qwen_core = df_qwen[['video_id', 'gold_label', 'pred_label']].rename(columns={'pred_label': 'qwen_pred'})
    int_core = df_int[['video_id', 'pred_label']].rename(columns={'pred_label': 'internvl_pred'})
    
    # 根据 video_id 合并两张表
    df_merged = pd.merge(qwen_core, int_core, on='video_id', how='inner')
    
    if df_merged.empty:
        print("❌ 合并失败，两张表没有相同的 video_id！请检查数据。")
        return
        
    print(f"✅ 成功合并 {len(df_merged)} 条评估数据！\n")
    
    # ================= 2. 数据增强与状态判定 =================
    
    # 映射为可读的中文字符串
    df_merged['gold_name'] = df_merged['gold_label'].apply(map_label)
    df_merged['qwen_pred_name'] = df_merged['qwen_pred'].apply(map_label)
    df_merged['internvl_pred_name'] = df_merged['internvl_pred'].apply(map_label)
    
    # 判定 Qwen 状态
    def get_qwen_status(row):
        if row['gold_label'] == row['qwen_pred']: return "✅ Qwen 正确"
        return f"❌ 错判为 {row['qwen_pred']}"
        
    # 判定 InternVL 状态
    def get_internvl_status(row):
        if row['gold_label'] == row['internvl_pred']: return "✅ InternVL 正确"
        return f"❌ 错判为 {row['internvl_pred']}"
        
    # 判定双模型联合状态 (非常有意思的论文切入点)
    def get_joint_status(row):
        q_right = (row['gold_label'] == row['qwen_pred'])
        i_right = (row['gold_label'] == row['internvl_pred'])
        if q_right and i_right: return "🌟 双模型皆准"
        if not q_right and not i_right: return "☠️ 联合盲区 (双双错判)"
        if q_right and not i_right: return "🔹 仅 Qwen 对"
        if not q_right and i_right: return "🔸 仅 InternVL 对"
        
    df_merged['qwen_status'] = df_merged.apply(get_qwen_status, axis=1)
    df_merged['internvl_status'] = df_merged.apply(get_internvl_status, axis=1)
    df_merged['joint_status'] = df_merged.apply(get_joint_status, axis=1)
    
    # ================= 3. 整理列顺序并保存 =================
    final_cols = [
        'video_id', 
        'gold_label', 'gold_name', 
        'qwen_pred', 'qwen_pred_name', 'qwen_status',
        'internvl_pred', 'internvl_pred_name', 'internvl_status',
        'joint_status'
    ]
    df_final = df_merged[final_cols]
    
    # 为了方便查阅，我们按照“真实标签”进行排序，把错误的顶在前面
    df_final = df_final.sort_values(by=['gold_label', 'joint_status'])
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_final.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"🎉 混淆分析表已成功导出至: {OUTPUT_CSV}")
    
    # ================= 4. 打印核心诊断信息 =================
    print("\n" + "="*40)
    print("📊 典型错判统计 (可直接用于论文撰写)")
    print("="*40)
    
    # 统计“原本是 1，却被误判为其他”的情况
    for target_label in [1, 2, 3, 4]:
        df_target = df_final[df_final['gold_label'] == target_label]
        qwen_wrong = df_target[df_target['qwen_pred'] != target_label]
        int_wrong = df_target[df_target['internvl_pred'] != target_label]
        
        print(f"\n🎯 真实标签为 【{LABEL_MAP[target_label]}】 时的误判情况:")
        print(f"  - Qwen2-VL 错判了 {len(qwen_wrong)} 个。最常错判成了: " + 
              (str(qwen_wrong['qwen_pred_name'].mode().iloc[0]) if not qwen_wrong.empty else "无"))
        print(f"  - InternVL2 错判了 {len(int_wrong)} 个。最常错判成了: " + 
              (str(int_wrong['internvl_pred_name'].mode().iloc[0]) if not int_wrong.empty else "无"))
        
    print("\n💡 提示：打开生成的 CSV 文件，筛选 `joint_status` 为 '☠️ 联合盲区 (双双错判)' 的数据。")
    print("这些视频就是纯视觉模型（无论哪种架构）根本无法处理的终极难题，也就是必须引入 Agent 3 (听觉) 才能解决的案例！")

if __name__ == "__main__":
    main()