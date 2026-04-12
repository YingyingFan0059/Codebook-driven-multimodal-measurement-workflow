#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
实证研究绘图脚本：多模型底座全局对比
包含：
1. MiniCPM-o-2.6 vs Qwen3-Omni 双模型对比 (柱状图、雷达图、并排混淆矩阵)
2. Qwen2-VL vs InternVL2 vs MiniCPM vs Qwen3 四模型全局对比 (柱状图、雷达图、2x2混淆矩阵)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ================= 1. 路径与配置 =================
PROJECT_ROOT = "/root/autodl-tmp/project_douyin_mm"
EVAL_OUT_DIR = os.path.join(PROJECT_ROOT, "outputs/scaling_eval")
PLOT_OUT_DIR = os.path.join(PROJECT_ROOT, "outputs/plots")
os.makedirs(PLOT_OUT_DIR, exist_ok=True)

CLASS_NAMES = ['Performative', 'Moral', 'Procedural', 'Technical', 'Other']
ORDERED_LABELS = [1, 2, 3, 4, 0] # 对应真实标签 ID

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# 模型全局配色方案
COLOR_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # 蓝(Qwen2), 橙(Intern), 绿(MiniCPM), 红(Qwen3)
CMAP_PALETTE = ['Blues', 'Oranges', 'Greens', 'Reds']

# ================= 2. 数据读取工具 =================
def get_metrics_from_csv(csv_path):
    if not os.path.exists(csv_path): return None
    df = pd.read_csv(csv_path)
    y_true = df['gold_label'].values
    y_pred = df['pred_label'].values
    class_f1s = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3, 4])
    return {
        'F1_Other': class_f1s[0] if len(class_f1s) > 0 else 0,
        'F1_Performative': class_f1s[1] if len(class_f1s) > 1 else 0,
        'F1_Moral': class_f1s[2] if len(class_f1s) > 2 else 0,
        'F1_Procedural': class_f1s[3] if len(class_f1s) > 3 else 0,
        'F1_Technical': class_f1s[4] if len(class_f1s) > 4 else 0,
        'Macro_F1': f1_score(y_true, y_pred, average='macro'),
        'Accuracy': accuracy_score(y_true, y_pred),
        'y_true': y_true,
        'y_pred': y_pred
    }

# ================= 3. 通用绘图：分组柱状图 =================
def plot_bar_chart(models_metrics, title, out_filename):
    if not models_metrics: return
    categories = CLASS_NAMES + ['Overall\n(Macro-F1)']
    x = np.arange(len(categories))
    
    n_models = len(models_metrics)
    width = 0.8 / n_models  # 动态调整柱子宽度
    
    fig, ax = plt.subplots(figsize=(max(10, 2.5 * n_models), 6), dpi=300)
    
    for i, (model_name, metrics) in enumerate(models_metrics.items()):
        scores = [metrics[f'F1_{c}'] for c in CLASS_NAMES] + [metrics['Macro_F1']]
        # 动态计算偏移量，让柱子紧凑并排
        offset = (i - n_models/2 + 0.5) * width
        rects = ax.bar(x + offset, scores, width, label=model_name, 
                       color=COLOR_PALETTE[i % len(COLOR_PALETTE)], edgecolor='black', alpha=0.8)
        ax.bar_label(rects, fmt='%.2f', padding=3, fontsize=9)
        
    ax.set_title(title, pad=20)
    ax.set_ylabel('Score (F1)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=0)
    ax.set_ylim(0, 1.1)
    
    # 图例放于正上方，自适应列数
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=n_models, frameon=True)
    
    out_path = os.path.join(PLOT_OUT_DIR, out_filename)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

# ================= 4. 通用绘图：多边形雷达图 =================
def plot_radar_chart(models_metrics, title, out_filename):
    if not models_metrics: return
    categories = CLASS_NAMES
    N_vars = len(categories)
    
    angles = [n / float(N_vars) * 2 * np.pi for n in range(N_vars)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300, subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    for i, (model_name, metrics) in enumerate(models_metrics.items()):
        vals = [metrics[f'F1_{c}'] for c in categories]
        vals += vals[:1]
        c = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        
        ax.plot(angles, vals, linewidth=2, linestyle='solid', color=c, label=model_name)
        ax.fill(angles, vals, color=c, alpha=0.15)
        
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1.0)
    
    ax.set_title(title, pad=30, fontsize=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    out_path = os.path.join(PLOT_OUT_DIR, out_filename)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

# ================= 5. 通用绘图：混淆矩阵网格 =================
def plot_confusion_matrices(models_metrics, title_prefix, out_filename):
    n_models = len(models_metrics)
    if n_models == 0: return
    
    cols = min(2, n_models)
    rows = (n_models + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 6*rows), dpi=300)
    if n_models == 1: axes = [axes]
    elif rows > 1: axes = axes.flatten()
    
    for i, (model_name, metrics) in enumerate(models_metrics.items()):
        ax = axes[i]
        cm = confusion_matrix(metrics['y_true'], metrics['y_pred'], labels=ORDERED_LABELS)
        cm_norm = np.nan_to_num(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
        
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap=CMAP_PALETTE[i % len(CMAP_PALETTE)], ax=ax, 
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, vmin=0, vmax=1, annot_kws={"size": 13})
        ax.set_title(f"{model_name} Confusion Matrix", pad=15)
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        
    # 隐藏多余的空白子图 (如果模型数量为 3)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    out_path = os.path.join(PLOT_OUT_DIR, out_filename)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("⏳ 正在读取评估结果数据...")
    
    # 历史两大数据
    f_qwen2 = os.path.join(EVAL_OUT_DIR, "eval_lora_5class_2200_predictions.csv")
    f_internvl = os.path.join(EVAL_OUT_DIR, "eval_internvl_lora_5class_2200_predictions.csv")
    
    # 新晋两大数据
    f_minicpm = os.path.join(EVAL_OUT_DIR, "eval_minicpm_o_lora_predictions.csv")
    f_qwen3 = os.path.join(EVAL_OUT_DIR, "eval_qwen3_omni_predictions.csv")
    
    m_qwen2 = get_metrics_from_csv(f_qwen2)
    m_internvl = get_metrics_from_csv(f_internvl)
    m_minicpm = get_metrics_from_csv(f_minicpm)
    m_qwen3 = get_metrics_from_csv(f_qwen3)
    
    # ================= 阶段 1：绘制 2个新模型的对比 (MiniCPM vs Qwen3) =================
    models_2 = {}
    if m_minicpm: models_2['MiniCPM-o-2.6'] = m_minicpm
    if m_qwen3: models_2['Qwen3-Omni-30B'] = m_qwen3
    
    if len(models_2) > 0:
        print(f"🚀 生成 2个新模型 (MiniCPM vs Qwen3) 专属图表...")
        plot_bar_chart(models_2, 'Architectural Bias Analysis: MiniCPM-o vs Qwen3-Omni', '4a_bar_chart_minicpm_vs_qwen3.png')
        plot_radar_chart(models_2, 'Cognitive Shape Radar: MiniCPM-o vs Qwen3-Omni', '4b_radar_chart_minicpm_vs_qwen3.png')
        plot_confusion_matrices(models_2, '', '4c_cm_minicpm_vs_qwen3.png')
    
    # ================= 阶段 2：绘制 4个模型的史诗级全局对比 =================
    models_4 = {}
    if m_qwen2: models_4['Qwen2-VL-7B'] = m_qwen2
    if m_internvl: models_4['InternVL2-8B'] = m_internvl
    if m_minicpm: models_4['MiniCPM-o-2.6'] = m_minicpm
    if m_qwen3: models_4['Qwen3-Omni-30B'] = m_qwen3
    
    if len(models_4) > 1:
        print(f"🚀 生成 {len(models_4)}大模型全家福对比图表...")
        plot_bar_chart(models_4, 'Global Architectural Bias Analysis (4 Models)', '5a_bar_chart_4models.png')
        plot_radar_chart(models_4, 'Global Cognitive Shape Radar (4 Models)', '5b_radar_chart_4models.png')
        plot_confusion_matrices(models_4, '', '5c_cm_4models.png')
        
    print("\n🎉 大满贯！所有模型深度对比图表 (分组柱状图、雷达图、矩阵网格) 均已生成至 outputs/plots 目录。")