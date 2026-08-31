# -*- coding: utf-8 -*-
"""
generate_report_figures.py -- CableFormer V1-V4 全版本对比图生成
生成: Loss 曲线对比、real_inference 逐样本对比、pred_examples 逐样本对比
输出: [V2]CableFormer/ 目录
"""
from __future__ import annotations

import sys
import csv
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ── matplotlib 全局样式 ──
rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "SimSun"],
    "mathtext.fontset":  "stix",
    "axes.unicode_minus": False,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.top":         True,
    "ytick.right":       True,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth":   1.0,
    "legend.fontsize":   8,
    "figure.dpi":        300,
})

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "Src"))

from config import load_config, Config
from models.unet1d import UNet1D
from models.cableformer import CableFormer
from core.dataset import CableDefectDataset
from core.tdr_signal import read_s11_csv, s11_to_responses, to_fixed_distance_grid

D_MAX = 1200.0
DD = 0.5
N_GRID = int(round(D_MAX / DD))
GRID = np.linspace(0, D_MAX, N_GRID, endpoint=False) + DD / 2

# ── 实验配置 ──
EXPERIMENTS = {
    "V1 (UNet1D)": {
        "config": "AgentsStorage/experiments/exp_20260617_150349_unet1d_medium_baseline/config.yaml",
        "checkpoint": "AgentsStorage/experiments/exp_20260617_150349_unet1d_medium_baseline/best.pt",
        "train_log": "AgentsStorage/experiments/exp_20260617_150349_unet1d_medium_baseline/train_log.csv",
        "color": "#2196F3",
        "ls": "-",
    },
    "V2 (Dice Loss)": {
        "config": "AgentsStorage/experiments/exp_20260617_175743_cableformer_medium_v2/config.yaml",
        "checkpoint": "AgentsStorage/experiments/exp_20260617_175743_cableformer_medium_v2/best.pt",
        "train_log": "AgentsStorage/experiments/exp_20260617_175743_cableformer_medium_v2/train_log.csv",
        "color": "#FF9800",
        "ls": "--",
    },
    "V3 (Final)": {
        "config": "AgentsStorage/experiments/exp_20260618_004435_cableformer_medium_v3/config.yaml",
        "checkpoint": "AgentsStorage/experiments/exp_20260618_004435_cableformer_medium_v3/best.pt",
        "train_log": "AgentsStorage/experiments/exp_20260618_004435_cableformer_medium_v3/train_log.csv",
        "color": "#4CAF50",
        "ls": "-",
    },
    "V4 (Relaxed)": {
        "config": "AgentsStorage/experiments/exp_20260618_035421_cableformer_medium_v4/config.yaml",
        "checkpoint": "AgentsStorage/experiments/exp_20260618_035421_cableformer_medium_v4/best.pt",
        "train_log": "AgentsStorage/experiments/exp_20260618_035421_cableformer_medium_v4/train_log.csv",
        "color": "#E91E63",
        "ls": ":",
    },
}

OUTPUT_DIR = Path("[V2]CableFormer")
REAL_DATA_DIR = Path(r"D:\FDR案例-csv\无校准S11")


def load_model(cfg: Config, checkpoint_path: str, device: torch.device):
    if cfg.model.name == "cableformer":
        model = CableFormer(
            base_ch=cfg.model.base_ch,
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            transformer_dim=cfg.model.transformer_dim,
            n_transformer_blocks=cfg.model.n_transformer_blocks,
            n_heads=cfg.model.n_heads,
            use_cbam=cfg.model.use_cbam,
        )
    else:
        model = UNet1D(
            base_ch=cfg.model.base_ch,
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            kernel_size=cfg.model.kernel_size,
        )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_all_models(device):
    """Load all 4 models."""
    models = {}
    configs = {}
    for name, info in EXPERIMENTS.items():
        print(f"  Loading {name}...")
        cfg = load_config(info["config"])
        model = load_model(cfg, info["checkpoint"], device)
        models[name] = model
        configs[name] = cfg
    return models, configs


def run_inference(models, inp_tensor, device):
    """Run all models on same input, return dict of predictions."""
    preds = {}
    with torch.no_grad():
        for name, model in models.items():
            _, probs = model(inp_tensor.to(device))
            preds[name] = probs[0].cpu().numpy()
    return preds


def read_train_log(log_path):
    """Read train_log.csv, return dict of lists."""
    data = {"epoch": [], "train_loss": [], "val_loss": [],
            "val_recall": [], "val_precision": [], "val_far": []}
    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["epoch"].append(int(row["epoch"]))
            data["train_loss"].append(float(row["train_loss"]))
            data["val_loss"].append(float(row["val_loss"]))
            data["val_recall"].append(float(row["val_recall"]))
            data["val_precision"].append(float(row["val_precision"]))
            data["val_far"].append(float(row["val_far"]))
    return data


# ═══════════════════════════════════════════
# 1. Loss 曲线对比
# ═══════════════════════════════════════════

def plot_loss_comparison():
    """Plot loss curves for all versions side by side."""
    print("Generating loss comparison...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1: Train Loss
    ax = axes[0]
    for name, info in EXPERIMENTS.items():
        log = read_train_log(info["train_log"])
        ax.plot(log["epoch"], log["train_loss"],
                color=info["color"], ls=info["ls"], label=name, linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss")
    ax.set_title("(a) Train Loss")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Panel 2: Val Loss
    ax = axes[1]
    for name, info in EXPERIMENTS.items():
        log = read_train_log(info["train_log"])
        ax.plot(log["epoch"], log["val_loss"],
                color=info["color"], ls=info["ls"], label=name, linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("(b) Validation Loss")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Panel 3: Val Recall
    ax = axes[2]
    for name, info in EXPERIMENTS.items():
        log = read_train_log(info["train_log"])
        ax.plot(log["epoch"], log["val_recall"],
                color=info["color"], ls=info["ls"], label=name, linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Recall")
    ax.set_title("(c) Validation Recall")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig_loss_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════
# 2. 单样本对比图 (通用)
# ═══════════════════════════════════════════

def plot_single_sample_comparison(
    imp, step, preds, label, title, filename,
    label_peaks=None, pred_peaks_dict=None,
):
    """
    生成单样本对比图:
    上图: 脉冲/阶跃响应
    下图: V1~V4 诊断结果 + Label (纵向偏移分开)
    """
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 7),
                                          gridspec_kw={"height_ratios": [1, 1.8]})

    # ── 上图: 输入信号 ──
    ax_top.plot(GRID, imp, color="#1565C0", alpha=0.8, linewidth=0.7, label="Impulse Response")
    ax_top.plot(GRID, step, color="#C62828", alpha=0.8, linewidth=0.7, label="Step Response")
    ax_top.set_xlabel("Distance (m)")
    ax_top.set_ylabel("Amplitude (norm.)")
    ax_top.set_title(title)
    ax_top.legend(loc="upper right")
    ax_top.grid(True, alpha=0.3, linewidth=0.5)
    ax_top.set_xlim(0, D_MAX)

    # ── 下图: 各版本预测 + Label (偏移) ──
    offsets = {
        "V1 (UNet1D)": 0.0,
        "V2 (Dice Loss)": 0.35,
        "V3 (Final)": 0.70,
        "V4 (Relaxed)": 1.05,
    }
    label_offset = 1.40

    # Label (ground truth)
    if label is not None:
        ax_bot.plot(GRID, label + label_offset, color="#333333",
                    linestyle="--", linewidth=1.0, alpha=0.8, label="Ground Truth")
        if label_peaks is not None and len(label_peaks) > 0:
            ax_bot.scatter(GRID[label_peaks], label[label_peaks] + label_offset,
                           c="#333333", marker="v", s=40, zorder=5)

    # 各版本预测
    for name, info in EXPERIMENTS.items():
        if name not in preds:
            continue
        off = offsets.get(name, 0)
        pred = preds[name]
        ax_bot.plot(GRID, pred + off, color=info["color"],
                    linestyle=info["ls"], linewidth=1.0, alpha=0.9, label=name)

        # 标注峰
        if pred_peaks_dict and name in pred_peaks_dict:
            pk = pred_peaks_dict[name]
            if len(pk) > 0:
                ax_bot.scatter(GRID[pk], pred[pk] + off,
                               c=info["color"], marker="^", s=30, zorder=5)

        # 版本名标注在右侧
        ax_bot.text(D_MAX + 5, off + 0.45, name, fontsize=7.5,
                    color=info["color"], va="center", fontweight="bold",
                    clip_on=False)

    # Label 标注
    if label is not None:
        ax_bot.text(D_MAX + 5, label_offset + 0.45, "Ground Truth", fontsize=7.5,
                    color="#333333", va="center", fontweight="bold",
                    clip_on=False)

    ax_bot.set_xlabel("Distance (m)")
    ax_bot.set_ylabel("Confidence (offset)")
    ax_bot.set_xlim(0, D_MAX + 80)
    ax_bot.set_ylim(-0.1, label_offset + 1.2)
    ax_bot.grid(True, alpha=0.3, linewidth=0.5)
    ax_bot.legend(loc="upper left", fontsize=7, ncol=2)

    plt.tight_layout()
    out_path = OUTPUT_DIR / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


# ═══════════════════════════════════════════
# 3. Prediction Examples (合成测试集)
# ═══════════════════════════════════════════

def plot_pred_examples_comparison(models, configs, device):
    """Select diverse test samples, generate per-sample comparison figures."""
    from scipy.signal import find_peaks

    print("Generating prediction examples comparison...")

    # 用 V3 config 加载测试集
    cfg = configs["V3 (Final)"]
    ds = CableDefectDataset(
        manifest_path=cfg.data.manifest,
        split="test",
        channels=cfg.data.channels,
        augment=False,
    )

    # 选样本: 找有中段缺陷的样本 + 末端样本
    np.random.seed(42)
    grid = GRID
    n_total = len(ds)

    # 先扫描找有中段缺陷的样本
    mid_defect_indices = []
    long_cable_indices = []
    for i in range(min(200, n_total)):
        _, labels = ds[i]
        label_np = labels.numpy()
        peaks, _ = find_peaks(label_np, height=0.3, distance=4)
        if len(peaks) > 0:
            peak_positions_m = grid[peaks]
            # 中段缺陷: 距起点 > 50m 且距末端 > 50m
            if any(50 < p < 450 for p in peak_positions_m):
                mid_defect_indices.append(i)
        # 长电缆
        sample = ds.samples[i]
        total_len = sample.get("total_length_m", 0)
        if total_len > 400:
            long_cable_indices.append(i)

    # 选取: 2个中段缺陷 + 2个长电缆 + 2个随机 = 6
    selected = []
    if len(mid_defect_indices) >= 2:
        selected.extend(mid_defect_indices[:2])
    if len(long_cable_indices) >= 2:
        for idx in long_cable_indices[:3]:
            if idx not in selected:
                selected.append(idx)
                if len(selected) >= 4:
                    break
    # 补随机
    random_pool = [i for i in range(n_total) if i not in selected]
    np.random.shuffle(random_pool)
    while len(selected) < 6:
        selected.append(random_pool.pop())

    print(f"  Selected samples: {selected}")

    for i, idx in enumerate(selected):
        inputs, labels = ds[idx]
        inp_tensor = inputs.unsqueeze(0).to(device)
        label_np = labels.numpy()

        # 跑所有模型
        preds = run_inference(models, inp_tensor, device)

        # 找峰
        label_peaks, _ = find_peaks(label_np, height=0.2, distance=4, prominence=0.1)
        pred_peaks_dict = {}
        for name, pred in preds.items():
            pk, _ = find_peaks(pred, height=0.15, distance=4, prominence=0.08)
            pred_peaks_dict[name] = pk

        imp = inputs[0].numpy()
        step = inputs[1].numpy()
        sample_info = ds.samples[idx]
        total_len = sample_info.get("total_length_m", 0)

        title = f"Test Sample #{idx} (Cable Length: {total_len:.0f}m)"

        fname = f"fig_pred_sample_{i+1}_s{idx}.png"
        out = plot_single_sample_comparison(
            imp, step, preds, label_np, title, fname,
            label_peaks=label_peaks, pred_peaks_dict=pred_peaks_dict,
        )
        print(f"  Saved: {out}")

    return selected


# ═══════════════════════════════════════════
# 4. Real Inference 对比
# ═══════════════════════════════════════════

def plot_real_inference_comparison(models, device):
    """Run all models on real S11 data, generate per-sample comparison figures."""
    print("Generating real inference comparison...")

    if not REAL_DATA_DIR.exists():
        print(f"  Warning: real data dir not found: {REAL_DATA_DIR}")
        return []

    csv_files = sorted(REAL_DATA_DIR.glob("**/*.csv"))
    if not csv_files:
        print("  Warning: no CSV files found")
        return []

    # 选多样化长度的样本 (每个子文件夹取第一个)
    seen_folders = set()
    selected_files = []
    for f in csv_files:
        folder = f.parent.name
        if folder not in seen_folders:
            seen_folders.add(folder)
            selected_files.append(f)

    # 取前6个不同场景
    selected_files = selected_files[:6]
    print(f"  Selected {len(selected_files)} real data files:")
    for f in selected_files:
        print(f"    {f.parent.name}/{f.name}")

    for i, csv_path in enumerate(selected_files):
        try:
            freqs, S = read_s11_csv(str(csv_path))
            epsr = 2.23
            distance, impulse, step, Z = s11_to_responses(freqs, S, epsr=epsr, window="hann")
            _, imp_grid, step_grid = to_fixed_distance_grid(
                distance, impulse, step, d_max=D_MAX, dd=DD
            )

            # 归一化
            max_abs = max(np.abs(imp_grid).max(), 1e-10)
            imp_norm = imp_grid / max_abs
            max_abs_s = max(np.abs(step_grid).max(), 1e-10)
            step_norm = step_grid / max_abs_s

            inp = torch.tensor(
                np.stack([imp_norm, step_norm], axis=0),
                dtype=torch.float32
            ).unsqueeze(0)

            preds = run_inference(models, inp, device)

            cable_info = csv_path.parent.name
            title = f"{csv_path.parent.name} / {csv_path.name}"

            fname = f"fig_real_sample_{i+1}.png"

            out = plot_single_sample_comparison(
                imp_norm, step_norm, preds, None, title, fname,
            )
            print(f"  Saved: {out}")

        except Exception as e:
            print(f"  Warning: failed on {csv_path.name}: {e}")

    return selected_files


# ═══════════════════════════════════════════
# 5. Metrics 汇总表图
# ═══════════════════════════════════════════

def plot_metrics_summary():
    """Generate a visual bar chart comparing test metrics across versions."""
    print("Generating metrics summary chart...")

    metrics = {
        "V1 (UNet1D)":    {"Recall": 0.9052, "Precision": 0.7544, "FAR": 0.774, "End Recall": 0.9160, "AUC": 0.9984},
        "V2 (Dice Loss)": {"Recall": 0.8782, "Precision": 0.8771, "FAR": 0.267, "End Recall": 0.9080, "AUC": 0.9987},
        "V3 (Final)":     {"Recall": 0.8967, "Precision": 0.7704, "FAR": 0.713, "End Recall": 0.9147, "AUC": 0.9984},
        "V4 (Relaxed)":   {"Recall": 0.9037, "Precision": 0.6417, "FAR": 1.553, "End Recall": 0.9293, "AUC": 0.9981},
    }

    metric_keys = ["Recall", "Precision", "End Recall"]
    versions = list(metrics.keys())
    n_versions = len(versions)
    n_metrics = len(metric_keys)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Recall / Precision / End Recall (grouped bar)
    ax = axes[0]
    x = np.arange(n_metrics)
    width = 0.18
    colors = [EXPERIMENTS[v]["color"] for v in versions]

    for i, v in enumerate(versions):
        vals = [metrics[v][k] for k in metric_keys]
        bars = ax.bar(x + i * width, vals, width, label=v, color=colors[i],
                       edgecolor="black", linewidth=0.5, alpha=0.85)
        # 数值标签
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=45)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metric_keys)
    ax.set_ylabel("Score")
    ax.set_title("(a) Detection Performance")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, linewidth=0.5, axis="y")

    # Panel 2: FAR (lower is better)
    ax = axes[1]
    far_vals = [metrics[v]["FAR"] for v in versions]
    bars = ax.bar(versions, far_vals, color=colors, edgecolor="black",
                   linewidth=0.5, alpha=0.85)
    for bar, val in zip(bars, far_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("False Alarm Rate (per sample)")
    ax.set_title("(b) False Alarm Rate")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3, linewidth=0.5, axis="y")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig_metrics_summary.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output: {OUTPUT_DIR.resolve()}")

    # 1. Loss comparison (no GPU needed)
    plot_loss_comparison()

    # 2. Metrics summary (no GPU needed)
    plot_metrics_summary()

    # 3. Load all models
    print("\nLoading models...")
    models, configs = load_all_models(device)

    # 4. Prediction examples
    print()
    plot_pred_examples_comparison(models, configs, device)

    # 5. Real inference
    print()
    plot_real_inference_comparison(models, device)

    print("\nAll figures generated!")


if __name__ == "__main__":
    main()
