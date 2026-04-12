#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
计算社会科学实证论文指标对账脚本 (后处理版)
目标：读取 Agent 3 产生的终极 CSV 成绩单，计算纯视觉与 Agent 融合后的指标对比。
支持：准确率 (Accuracy) 和 Macro-F1。
用法：在主推理脚本跑完并产生 CSV 后运行此脚本。
"""

import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# ================= 1. 配置路径 =================
# 🔑 请替换为你跑出来的那个终极 CSV 文件的绝对路径
# 例如你在本地跑，就是 "./final_adjudicated_results_glm_5.csv"
FINAL_RESULTS_CSV = "/root/autodl-tmp/project_douyin_mm/final_adjudicated_results_glm_5.csv" 

def main():
    print("📋 开始进行 CSS 多智能体融合指标对账与对比...")
    print("-" * 50)
    
    if not os.path.exists(FINAL_RESULTS_CSV):
        print(f"❌ 错误：找不到文件 {FINAL_RESULTS_CSV}")
        print("💡 提示：请确保你的 Agent 3 推理脚本已经跑完，或者至少产生了一个 CSV 文件。")
        return

    # 2. 读取数据
    df = pd.read_csv(FINAL_RESULTS_CSV)
    print(f"✅ 成功读取到 {len(df)} 条有效裁决记录。")
    
    # 清理数据（确保标签全为整数类型，防止计算出错）
    try:
        y_true = df['gold_label'].astype(int)
        y_pred_vis = df['visual_pred'].astype(int) # 纯视觉单模态初判
        y_pred_final = df['final_label'].astype(int) # 加入听觉后的主编终判
    except KeyError:
        print("❌ 错误：CSV 文件中缺失必需的 'gold_label', 'visual_pred' 或 'final_label' 列。")
        return

    # ================= 3. 核心计算逻辑 =================
    
    # 🎯 计算【准确率 (Accuracy)】
    acc_vis = accuracy_score(y_true, y_pred_vis)
    acc_adj = accuracy_score(y_true, y_pred_final)
    acc_lift = acc_adj - acc_vis
    
    # 🎯 计算【Macro-F1 ( CSS 核心指标 )】
    f1_vis = f1_score(y_true, y_pred_vis, average='macro')
    f1_adj = f1_score(y_true, y_pred_final, average='macro')
    f1_lift = f1_adj - f1_vis

    # ================= 4. 打印最终学术成绩单 =================
    print("\n" + "="*50)
    print("🏆 跨模态融合(Agentic Integration) 最终对比成绩单")
    print("="*50)
    
    # 打印 F1 (宏观 F1 是社会科学多分类任务的最核心指标)
    print(f"🔹 【Macro-F1】成绩:")
    print(f"   Agent 1 (纯视觉单模态 N=2200): {f1_vis:.4f}")
    print(f"   Agent 3 (主编终判 - 加入听觉): {f1_adj:.4f}")
    print(f"   📈 F1 提升幅度: +{f1_lift:.4f} (论文核心数据)")
    
    print("-" * 30)
    
    # 打印准确率
    print(f"🔹 【Accuracy】成绩:")
    print(f"   Agent 1 (纯视觉单模态 N=2200): {acc_vis:.4f}")
    print(f"   Agent 3 (主编终判 - 加入听觉): {acc_adj:.4f}")
    print(f"   📈 Accuracy 提升幅度: +{acc_lift:.4f}")
    
    print("="*50)
    print(f"🎉 提升幅度如果 +0.1 以上 (即10个百分点)，这在 Agent 辅助社科编码论文中是极其震撼的提升！")
    print("-" * 50)

if __name__ == "__main__":
    main()