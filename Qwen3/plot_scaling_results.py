#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
计算社会科学实证论文绘图脚本：Scaling Law 曲线与混淆矩阵 (含基线)
输入：Qwen2-VL 统考生成的各个 N 的 predictions.csv 以及 基线 predictions.csv
输出：符合学术发表标准的高清图表 (300 DPI)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from matplotlib.lines import Line2D

# ================= 1. 路径与配置 =================
PROJECT_ROOT = "/root/autodl-tmp/project_douyin_mm"
EVAL_OUT_DIR = os.path.join(PROJECT_ROOT, "outputs/scaling_eval")
PLOT_OUT_DIR = os.path.join(PROJECT_ROOT, "outputs/plots")
os.makedirs(PLOT_OUT_DIR, exist_ok=True)

# 纯英文标签，并且按照要求的展示顺序（将 Other 放在最后）
CLASS_NAMES = ['Performative', 'Moral', 'Procedural', 'Technical', 'Other']
ORDERED_LABELS = [1, 2, 3, 4, 0]

# 图表美化设置
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def get_metrics_from_csv(csv_path):
    if not os.path.exists(csv_path): return None
    df = pd.read_csv(csv_path)
    y_true = df['gold_label'].values
    y_pred = df['pred_label'].values
    class_f1s = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3, 4])
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Macro_F1': f1_score(y_true, y_pred, average='macro'),
        'F1_Other': class_f1s[0] if len(class_f1s) > 0 else 0,
        'F1_Performative': class_f1s[1] if len(class_f1s) > 1 else 0,
        'F1_Moral': class_f1s[2] if len(class_f1s) > 2 else 0,
        'F1_Procedural': class_f1s[3] if len(class_f1s) > 3 else 0,
        'F1_Technical': class_f1s[4] if len(class_f1s) > 4 else 0,
        'y_true': y_true, 'y_pred': y_pred
    }

def gather_baseline_data():
    print("⏳ 正在读取 Zero-shot 与 Rule-based 基线数据...")
    baselines = {}
    zs_path = os.path.join(EVAL_OUT_DIR, "eval_baseline_path1_zeroshot_predictions.csv")
    rb_path = os.path.join(EVAL_OUT_DIR, "eval_baseline_path2_rulebased_predictions.csv")
    baselines['zeroshot'] = get_metrics_from_csv(zs_path)
    baselines['rulebased'] = get_metrics_from_csv(rb_path)
    return baselines

def gather_scaling_data(scaling_sizes):
    print("⏳ 正在收集所有 LoRA 梯度的评估数据...")
    results = []
    for size in scaling_sizes:
        csv_path = os.path.join(EVAL_OUT_DIR, f"eval_lora_5class_{size}_predictions.csv")
        metrics = get_metrics_from_csv(csv_path)
        if metrics:
            metrics['N'] = size
            results.append(metrics)
    print(f"✅ 成功收集 {len(results)} 个有效微调数据点。")
    return pd.DataFrame(results)

def plot_overall_learning_curve(df_results, baselines):
    if df_results.empty: return
    plt.figure(figsize=(10, 6), dpi=300)
    
    if baselines['rulebased']: 
        plt.axhline(y=baselines['rulebased']['Macro_F1'], color='black', linestyle='--', linewidth=1.5, alpha=0.8, label='Baseline: Rule-based (Prompting)')
    if baselines['zeroshot']: 
        plt.axhline(y=baselines['zeroshot']['Macro_F1'], color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='Baseline: Zero-shot')
    
    plt.plot(df_results['N'], df_results['Macro_F1'], marker='o', linewidth=2.5, markersize=8, label='Fine-tuned Macro-F1', color='#1f77b4')
    plt.plot(df_results['N'], df_results['Accuracy'], marker='s', linewidth=2, markersize=7, linestyle='-.', label='Fine-tuned Accuracy', color='#7f7f7f', alpha=0.6)
    
    plt.title('Overall Scaling Law vs. Prompting Baselines', pad=15)
    plt.xlabel('Training Data Size (N)')
    plt.ylabel('Score')
    plt.xticks(df_results['N'], rotation=45)
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right', frameon=True, shadow=True)
    out_path = os.path.join(PLOT_OUT_DIR, "1a_scaling_law_overall_curve.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

# 🌟 核心修改处：去除了 baselines 参数，彻底移除水平基线与干扰图例
def plot_class_learning_curves(df_results):
    if df_results.empty: return
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    colors = {'Performative': '#ff7f0e', 'Moral': '#d62728', 'Procedural': '#9467bd', 'Technical': '#2ca02c', 'Other': '#7f7f7f'}
    
    ax.plot(df_results['N'], df_results['F1_Performative'], marker='v', linewidth=2.5, markersize=7, label='Performative', color=colors['Performative'])
    ax.plot(df_results['N'], df_results['F1_Moral'], marker='x', linewidth=2.5, markersize=8, label='Moral', color=colors['Moral'])
    ax.plot(df_results['N'], df_results['F1_Procedural'], marker='s', linewidth=2.5, markersize=7, label='Procedural', color=colors['Procedural'])
    ax.plot(df_results['N'], df_results['F1_Technical'], marker='^', linewidth=2.5, markersize=7, label='Technical', color=colors['Technical'])
    ax.plot(df_results['N'], df_results['F1_Other'], marker='o', linewidth=2.5, markersize=7, linestyle='-', label='Other', color=colors['Other'])
    
    ax.set_title('Class-specific Scaling Trajectories (F1 Score)', pad=15)
    ax.set_xlabel('Training Data Size (N)')
    ax.set_ylabel('F1 Score')
    ax.set_xticks(df_results['N'])
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, 1.0)
    
    ax.legend(loc='lower right', frameon=True, shadow=True, ncol=2)
    out_path = os.path.join(PLOT_OUT_DIR, "1b_scaling_law_class_curves.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

# ================= 5. 绘制特定或最佳 N 的混淆矩阵 =================
def plot_best_confusion_matrix(df_results, target_N=None):
    if df_results.empty: return
    
    if target_N is not None:
        matching_rows = df_results[df_results['N'] == target_N]
        if matching_rows.empty:
            print(f"⚠️ 找不到 N={target_N} 的数据，请检查该规模的模型是否已评估完。")
            return
        best_row = matching_rows.iloc[0]
        best_N = int(best_row['N'])
        print(f"📊 手动指定绘制 N={best_N} 的混淆矩阵 (Macro-F1: {best_row['Macro_F1']:.4f})")
    else:
        best_row = df_results.loc[df_results['Macro_F1'].idxmax()]
        best_N = int(best_row['N'])
        print(f"📊 自动定位到最佳性能点 N={best_N} (Macro-F1: {best_row['Macro_F1']:.4f})")
        
    y_true = best_row['y_true']
    y_pred = best_row['y_pred']
    
    cm_absolute = confusion_matrix(y_true, y_pred, labels=ORDERED_LABELS)
    cm_normalized = cm_absolute.astype('float') / cm_absolute.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    # --- 版本 A：归一化（带小数点，适合论文正文横向比较） ---
    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                annot_kws={"size": 12}, vmin=0, vmax=1)
    plt.title(f'Normalized Confusion Matrix (Proportions) at N={best_N}', pad=15)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    out_norm = os.path.join(PLOT_OUT_DIR, f"2a_confusion_matrix_normalized_N{best_N}.png")
    plt.savefig(out_norm, bbox_inches='tight')
    plt.close()

    # --- 版本 B：绝对数量（整数形式，适合附录和数据统计） ---
    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(cm_absolute, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                annot_kws={"size": 12})
    plt.title(f'Absolute Confusion Matrix (Counts) at N={best_N}', pad=15)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    out_abs = os.path.join(PLOT_OUT_DIR, f"2b_confusion_matrix_absolute_N{best_N}.png")
    plt.savefig(out_abs, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 混淆矩阵已保存至: {PLOT_OUT_DIR} (包含百分比与整数双版本)")

if __name__ == "__main__":
    scaling_sizes = list(range(400, 3200, 200))
    baselines = gather_baseline_data()
    df_res = gather_scaling_data(scaling_sizes)
    
    if not df_res.empty:
        plot_overall_learning_curve(df_res, baselines)
        # 🌟 这里的调用也已经更新，去掉了 baselines 参数
        plot_class_learning_curves(df_res)
        
        plot_best_confusion_matrix(df_res, target_N=2200) 
        
        csv_out = os.path.join(PLOT_OUT_DIR, "0_aggregated_scaling_metrics.csv")
        df_res.drop(columns=['y_true', 'y_pred']).to_csv(csv_out, index=False)
        print(f"\n🎉 绘图任务全部完成！图表和指标汇总 CSV 已存放在: {PLOT_OUT_DIR}")
    else:
        print("\n❌ 没有找到微调结果数据，请等待管线产出 predictions.csv。")