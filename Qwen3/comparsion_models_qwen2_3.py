#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen2-VL-7B vs Qwen3-Omni-30B 绘图脚本
生成：
1. 双模型分组柱状图
2. 双模型雷达图
3. 双模型并排混淆矩阵
4. Qwen3 单独雷达图
5. Qwen3 单独混淆矩阵
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ================= 1. 路径与配置 =================
PROJECT_ROOT = "/root/autodl-tmp/project_douyin_mm"
QWEN2_CSV = os.path.join(
    PROJECT_ROOT,
    "outputs/scaling_eval/eval_lora_5class_2200_predictions.csv"
)
QWEN3_CSV = os.path.join(
    PROJECT_ROOT,
    "outputs/scaling_eval/eval_qwen3_omni_predictions.csv"
)

PLOT_OUT_DIR = os.path.join(PROJECT_ROOT, "outputs/plots")
os.makedirs(PLOT_OUT_DIR, exist_ok=True)

CLASS_NAMES = ["Performative", "Moral", "Procedural", "Technical", "Other"]
LABEL_ORDER = [1, 2, 3, 4, 0]

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

# Qwen2 保持最初蓝色；Qwen3 改为紫色
MODEL_COLORS = {
    "Qwen2-VL-7B": "#1f77b4",
    "Qwen3-Omni-30B": "#6a3d9a",
}

MODEL_CMAPS = {
    "Qwen2-VL-7B": "Blues",
    "Qwen3-Omni-30B": "Purples",
}


# ================= 2. 数据读取 =================
def get_metrics_from_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"gold_label", "pred_label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{csv_path} 缺少必要列: {required_cols}")

    y_true = df["gold_label"].astype(int).values
    y_pred = df["pred_label"].astype(int).values

    class_f1 = f1_score(
        y_true, y_pred, average=None, labels=[0, 1, 2, 3, 4], zero_division=0
    )

    return {
        "F1_Other": class_f1[0],
        "F1_Performative": class_f1[1],
        "F1_Moral": class_f1[2],
        "F1_Procedural": class_f1[3],
        "F1_Technical": class_f1[4],
        "Macro_F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ================= 3. 双模型分组柱状图 =================
def plot_bar_chart(models_metrics, out_filename):
    categories = CLASS_NAMES + ["Overall\n(Macro-F1)"]
    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

    model_names = list(models_metrics.keys())
    for i, model_name in enumerate(model_names):
        metrics = models_metrics[model_name]
        scores = [
            metrics["F1_Performative"],
            metrics["F1_Moral"],
            metrics["F1_Procedural"],
            metrics["F1_Technical"],
            metrics["F1_Other"],
            metrics["Macro_F1"],
        ]

        offset = -width / 2 if i == 0 else width / 2
        bars = ax.bar(
            x + offset,
            scores,
            width=width,
            label=model_name,
            color=MODEL_COLORS[model_name],
            edgecolor="black",
            alpha=0.85,
        )

        # 数值标签放在柱顶
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)

    ax.set_title("Performance Comparison: Qwen2-VL-7B vs Qwen3-Omni-30B", pad=16)
    ax.set_ylabel("F1 Score")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.1)

    # 图例放右上角，纵向排列
    ax.legend(loc="upper right", frameon=True, ncol=1)

    out_path = os.path.join(PLOT_OUT_DIR, out_filename)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ================= 4. 双模型雷达图 =================
def plot_radar_chart(models_metrics, out_filename):
    categories = CLASS_NAMES
    n_vars = len(categories)

    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300, subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for model_name, metrics in models_metrics.items():
        values = [
            metrics["F1_Performative"],
            metrics["F1_Moral"],
            metrics["F1_Procedural"],
            metrics["F1_Technical"],
            metrics["F1_Other"],
        ]
        values += values[:1]

        ax.plot(
            angles,
            values,
            linewidth=2,
            color=MODEL_COLORS[model_name],
            label=model_name,
        )
        ax.fill(angles, values, color=MODEL_COLORS[model_name], alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_title("Class-wise F1 Radar: Qwen2-VL-7B vs Qwen3-Omni-30B", pad=24, fontsize=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.10))

    out_path = os.path.join(PLOT_OUT_DIR, out_filename)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ================= 5. 双模型并排混淆矩阵 =================
def plot_confusion_matrices(models_metrics, out_filename):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    for ax, (model_name, metrics) in zip(axes, models_metrics.items()):
        cm = confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=LABEL_ORDER)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, where=row_sums != 0)
        cm_norm = np.nan_to_num(cm_norm)

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap=MODEL_CMAPS[model_name],
            ax=ax,
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            vmin=0,
            vmax=1,
            annot_kws={"size": 11},
        )
        ax.set_title(model_name, pad=10)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    fig.suptitle("Normalized Confusion Matrices", fontsize=16, y=1.02)

    out_path = os.path.join(PLOT_OUT_DIR, out_filename)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ================= 6. Qwen3 单独雷达图 =================
def plot_single_radar(model_name, metrics, out_filename):
    categories = CLASS_NAMES
    n_vars = len(categories)

    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]

    values = [
        metrics["F1_Performative"],
        metrics["F1_Moral"],
        metrics["F1_Procedural"],
        metrics["F1_Technical"],
        metrics["F1_Other"],
    ]
    values += values[:1]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300, subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(
        angles,
        values,
        linewidth=2.5,
        color=MODEL_COLORS[model_name],
        label=model_name,
    )
    ax.fill(angles, values, color=MODEL_COLORS[model_name], alpha=0.20)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Class-wise F1 Radar: {model_name}", pad=24, fontsize=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.10))

    out_path = os.path.join(PLOT_OUT_DIR, out_filename)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ================= 7. Qwen3 单独混淆矩阵 =================
def plot_single_confusion_matrix(model_name, metrics, out_filename):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    cm = confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=LABEL_ORDER)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums != 0)
    cm_norm = np.nan_to_num(cm_norm)

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap=MODEL_CMAPS[model_name],
        ax=ax,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        vmin=0,
        vmax=1,
        annot_kws={"size": 12},
    )
    ax.set_title(f"Normalized Confusion Matrix: {model_name}", pad=12)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    out_path = os.path.join(PLOT_OUT_DIR, out_filename)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ================= 8. 主程序 =================
if __name__ == "__main__":
    print("Loading evaluation results...")

    qwen2_metrics = get_metrics_from_csv(QWEN2_CSV)
    qwen3_metrics = get_metrics_from_csv(QWEN3_CSV)

    models_metrics = {
        "Qwen2-VL-7B": qwen2_metrics,
        "Qwen3-Omni-30B": qwen3_metrics,
    }

    plot_bar_chart(models_metrics, "qwen2_vs_qwen3_bar.png")
    plot_radar_chart(models_metrics, "qwen2_vs_qwen3_radar.png")
    plot_confusion_matrices(models_metrics, "qwen2_vs_qwen3_confusion_matrix.png")

    plot_single_radar("Qwen3-Omni-30B", qwen3_metrics, "qwen3_only_radar.png")
    plot_single_confusion_matrix("Qwen3-Omni-30B", qwen3_metrics, "qwen3_only_confusion_matrix.png")

    print("Done. Plots saved to:")
    print(PLOT_OUT_DIR)