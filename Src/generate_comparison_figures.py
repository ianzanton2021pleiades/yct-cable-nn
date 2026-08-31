"""
generate_comparison_figures.py
Generates comparison figures for V1 UNet1D, UNet1D V2, and CableFormer V2 (Dice).

Output directory: [V1]1D-UNet/figures/
"""
import os
import sys
import csv
import json
import random
import glob as glob_mod

# ── working directory ──
PROJECT_ROOT = r"D:\GitRepository\cable-defect-location-method-based-on-BIS\AI_TEST"
os.chdir(PROJECT_ROOT)
sys.path.insert(0, "Src")

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from pathlib import Path
from scipy.signal import find_peaks

from config import load_config, Config
from models.unet1d import UNet1D
from models.unet1d_v2 import UNet1DV2
from models.cableformer import CableFormer
from core.dataset import CableDefectDataset
from core.tdr_signal import read_s11_csv, s11_to_responses, to_fixed_distance_grid

# ════════════════════════════════════════════════════════
#  Constants
# ════════════════════════════════════════════════════════
D_MAX = 1200.0
DD = 0.5
N_GRID = int(round(D_MAX / DD))
GRID = np.linspace(0, D_MAX, N_GRID, endpoint=False) + DD / 2

DPI = 300
OUTPUT_DIR = Path(PROJECT_ROOT) / "[V1]1D-UNet" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Experiment directories
EXPERIMENTS = {
    "V1 UNet1D": {
        "dir": "AgentsStorage/experiments/exp_20260617_150349_unet1d_medium_baseline",
        "config": "AgentsStorage/experiments/exp_20260617_150349_unet1d_medium_baseline/config.yaml",
        "checkpoint": "AgentsStorage/experiments/exp_20260617_150349_unet1d_medium_baseline/best.pt",
        "color": "#1f77b4",
        "short": "V1",
    },
    "UNet1D V2": {
        "dir": "AgentsStorage/experiments/exp_20260618_170217_unet1d_v2_medium",
        "config": "AgentsStorage/experiments/exp_20260618_170217_unet1d_v2_medium/config.yaml",
        "checkpoint": "AgentsStorage/experiments/exp_20260618_170217_unet1d_v2_medium/best.pt",
        "color": "#d62728",
        "short": "V2",
    },
    "CableFormer V2": {
        "dir": "AgentsStorage/experiments/exp_20260617_175743_cableformer_medium_v2",
        "config": "AgentsStorage/experiments/exp_20260617_175743_cableformer_medium_v2/config.yaml",
        "checkpoint": "AgentsStorage/experiments/exp_20260617_175743_cableformer_medium_v2/best.pt",
        "color": "#2ca02c",
        "short": "CF",
    },
}

# Real data directories to sample from
REAL_DATA_BASE = Path(r"D:\FDR案例-csv\无校准S11")
REAL_DATA_DIRS = [
    "大芬-3100m",
    "威海-543m",
    "常州-5100m",
    "深圳-830m",
    "莱芜-1500m",
    "某地-780m",
]

# ════════════════════════════════════════════════════════
#  Font setup — robust SimSun + Times New Roman
# ════════════════════════════════════════════════════════
def setup_fonts():
    """Configure matplotlib fonts: SimSun for Chinese, Times New Roman for English."""
    # Remove font cache to force rediscovery
    cache_dir = matplotlib.get_cachedir()
    for p in Path(cache_dir).glob("fontlist-*.json"):
        try:
            p.unlink()
            print(f"  Removed font cache: {p.name}")
        except Exception:
            pass

    fm._load_fontmanager(try_read_cache=False)

    # Try to find SimSun
    simsun_path = None
    for name in ["simsun.ttc", "SIMSUN.TTC", "simsun.ttf"]:
        fp = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / name
        if fp.exists():
            simsun_path = str(fp)
            break

    if simsun_path is None:
        # Try font manager
        try:
            found = fm.findfont(fm.FontProperties(family="SimSun"), fallback_to_default=False)
            if found and "simsun" in found.lower():
                simsun_path = found
        except Exception:
            pass

    if simsun_path:
        print(f"  Using font: {simsun_path}")
        fp = fm.FontProperties(fname=simsun_path)
        font_name = fp.get_name()
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = [font_name, "Times New Roman"]
    else:
        print("  WARNING: SimSun not found, using Times New Roman only")
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman", "SimSun"]

    # Common settings
    plt.rcParams.update({
        "mathtext.fontset":  "stix",
        "axes.unicode_minus": False,
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "xtick.top":         True,
        "ytick.right":       True,
        "axes.linewidth":    0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    # Quick test
    fig, ax = plt.subplots(figsize=(2, 1))
    ax.set_title("测试 Test")
    plt.close(fig)
    print("  Font test passed")


# ════════════════════════════════════════════════════════
#  Model loading
# ════════════════════════════════════════════════════════
def build_model(cfg: Config) -> torch.nn.Module:
    if cfg.model.name == "cableformer":
        return CableFormer(
            base_ch=cfg.model.base_ch,
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            transformer_dim=cfg.model.transformer_dim,
            n_transformer_blocks=cfg.model.n_transformer_blocks,
            n_heads=cfg.model.n_heads,
            use_cbam=cfg.model.use_cbam,
        )
    elif cfg.model.name == "unet1d_v2":
        return UNet1DV2(
            base_ch=cfg.model.base_ch,
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            use_cbam=cfg.model.use_cbam,
        )
    else:
        return UNet1D(
            base_ch=cfg.model.base_ch,
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            kernel_size=cfg.model.kernel_size,
        )


def load_all_models(device):
    """Load all three models and return dict of {name: (model, cfg)}."""
    models = {}
    for name, info in EXPERIMENTS.items():
        cfg = load_config(info["config"])
        model = build_model(cfg)
        ckpt = torch.load(info["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        models[name] = (model, cfg)
        print(f"  Loaded {name} ({cfg.model.name}, {cfg.model.base_ch}ch)")
    return models


def predict_all(models, input_tensor):
    """Run prediction with all models, return dict of {name: probs_np}."""
    results = {}
    with torch.no_grad():
        for name, (model, _) in models.items():
            _, probs = model(input_tensor)
            results[name] = probs[0].cpu().numpy()
    return results


# ════════════════════════════════════════════════════════
#  Figure: Loss comparison
# ════════════════════════════════════════════════════════
def gen_loss_comparison():
    """Generate loss curve comparison from train_log.csv files."""
    print("\n[1/3] Generating loss comparison...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for name, info in EXPERIMENTS.items():
        log_path = Path(info["dir"]) / "train_log.csv"
        if not log_path.exists():
            print(f"  Warning: {log_path} not found")
            continue

        epochs, train_loss, val_loss, val_recall, val_precision = [], [], [], [], []
        with open(log_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                train_loss.append(float(row["train_loss"]))
                val_loss.append(float(row["val_loss"]))
                val_recall.append(float(row["val_recall"]))
                val_precision.append(float(row["val_precision"]))

        # Plot train_loss
        axes[0].plot(epochs, train_loss, label=name, color=info["color"], linewidth=1.5)
        # Plot val_loss
        axes[1].plot(epochs, val_loss, label=name, color=info["color"], linewidth=1.5)
        # Plot val_recall & val_precision
        axes[2].plot(epochs, val_recall, color=info["color"], linewidth=1.5,
                     label=f"{name} Recall")
        axes[2].plot(epochs, val_precision, color=info["color"], linewidth=1.5,
                     linestyle="--", label=f"{name} Precision")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("(a) Training Loss")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].grid(True, alpha=0.3, linewidth=0.5)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Loss")
    axes[1].set_title("(b) Validation Loss")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(True, alpha=0.3, linewidth=0.5)

    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Metric Value")
    axes[2].set_title("(c) Validation Recall & Precision")
    axes[2].legend(fontsize=7, loc="lower right", ncol=2)
    axes[2].grid(True, alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig_loss_comparison.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ════════════════════════════════════════════════════════
#  Figure: Metrics summary bar chart
# ════════════════════════════════════════════════════════
def gen_metrics_summary():
    """Generate metrics summary bar chart from metrics.json files."""
    print("\n[Metrics] Generating metrics summary...")

    metrics_names = ["Recall", "Precision", "FAR", "Loc Error\nMedian (m)",
                     "AUC", "End Recall"]
    metrics_keys = ["recall", "precision", "far", "loc_error_median",
                    "auc", "end_recall"]

    model_names = list(EXPERIMENTS.keys())
    model_colors = [EXPERIMENTS[n]["color"] for n in model_names]
    all_metrics = {}

    for name, info in EXPERIMENTS.items():
        with open(Path(info["dir"]) / "metrics.json", "r") as f:
            all_metrics[name] = json.load(f)

    n_metrics = len(metrics_keys)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    x = np.arange(len(model_names))
    bar_width = 0.6

    for i, (mname, mkey) in enumerate(zip(metrics_names, metrics_keys)):
        ax = axes[i]
        values = [all_metrics[n][mkey] for n in model_names]
        bars = ax.bar(x, values, bar_width, color=model_colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=8, rotation=15, ha="right")
        ax.set_ylabel(mname)
        ax.set_title(mname)
        ax.grid(True, alpha=0.3, linewidth=0.5, axis="y")

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}" if val < 10 else f"{val:.2f}",
                    ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig_metrics_summary.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ════════════════════════════════════════════════════════
#  Figure: Prediction examples (synthetic test data)
# ════════════════════════════════════════════════════════
def gen_pred_examples(models, sample_indices, ds, sample_labels, device):
    """
    Generate prediction example figures.
    Each figure: 4 subplots (1 input + 3 model predictions).
    """
    print(f"\n[2/3] Generating {len(sample_indices)} prediction examples...")

    for fig_idx, (sample_idx, label_text) in enumerate(zip(sample_indices, sample_labels), 1):
        inputs, labels = ds[sample_idx]
        inp = inputs.unsqueeze(0).to(device)

        preds = predict_all(models, inp)
        imp = inputs[0].numpy()
        step = inputs[1].numpy()
        label = labels.numpy()

        fig, axes = plt.subplots(4, 1, figsize=(14, 12),
                                 gridspec_kw={"height_ratios": [1.2, 1, 1, 1]})

        # Subplot 0: Input responses
        ax = axes[0]
        ax.plot(GRID, imp, label="脉冲响应 (Impulse)", color="#333333", alpha=0.8, linewidth=0.8)
        ax2 = ax.twinx()
        ax2.plot(GRID, step, label="阶跃响应 (Step)", color="#888888", alpha=0.8, linewidth=0.8)
        ax.set_title(f"{label_text} — 输入信号", fontsize=11)
        ax.set_ylabel("脉冲响应幅值")
        ax2.set_ylabel("阶跃响应幅值")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
        ax2.legend(loc="upper right", fontsize=8, framealpha=0.8)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_xlim(0, D_MAX)

        # Subplots 1-3: Each model's prediction vs label
        for row, (name, info) in enumerate(EXPERIMENTS.items(), 1):
            ax = axes[row]
            pred = preds[name]

            # Label as reference
            ax.plot(GRID, label, color="#AAAAAA", linewidth=2.0, linestyle="--",
                    label="真实标签 (Ground Truth)", alpha=0.8)
            # Model prediction
            ax.plot(GRID, pred, color=info["color"], linewidth=1.5,
                    label=f"{name} 预测", alpha=0.9)

            # Mark peaks
            label_peaks, _ = find_peaks(label, height=0.2, distance=4, prominence=0.1)
            pred_peaks, _ = find_peaks(pred, height=0.2, distance=4, prominence=0.1)

            if len(label_peaks) > 0:
                ax.scatter(GRID[label_peaks], label[label_peaks], c="#666666",
                           marker="v", s=60, zorder=5, label="真实峰值")
            if len(pred_peaks) > 0:
                peak_positions = GRID[pred_peaks]
                ax.scatter(peak_positions, pred[pred_peaks], c=info["color"],
                           marker="^", s=60, zorder=5, label="预测峰值")
                # Annotate peak positions
                for pp in peak_positions:
                    ax.annotate(f"{pp:.1f}m", xy=(pp, 0.95), fontsize=7,
                                ha="center", va="top", color=info["color"])

            ax.set_ylabel("置信度")
            ax.set_title(f"{name} 诊断结果", fontsize=10)
            ax.legend(loc="upper right", fontsize=7, framealpha=0.8, ncol=2)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.set_xlim(0, D_MAX)
            ax.set_ylim(-0.05, 1.15)

        axes[-1].set_xlabel("距离 / m")

        plt.tight_layout()
        out_path = OUTPUT_DIR / f"fig_pred_sample_{fig_idx}.png"
        plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")


# ════════════════════════════════════════════════════════
#  Figure: Real data inference
# ════════════════════════════════════════════════════════
def gen_real_inference(models, real_dirs, device):
    """Generate real data inference figures."""
    print(f"\n[3/3] Generating {len(real_dirs)} real data inference figures...")

    for fig_idx, subdir_name in enumerate(real_dirs, 1):
        real_dir = REAL_DATA_BASE / subdir_name
        if not real_dir.exists():
            print(f"  Warning: {real_dir} not found, skipping")
            continue

        csv_files = sorted(real_dir.glob("*.csv"))
        if not csv_files:
            print(f"  Warning: no CSV files in {real_dir}")
            continue

        csv_path = csv_files[0]  # Take first CSV from each directory

        try:
            freqs, S = read_s11_csv(str(csv_path))
            epsr = 2.23
            distance, impulse, step, Z = s11_to_responses(freqs, S, epsr=epsr, window="hann")
            _, imp_grid, step_grid = to_fixed_distance_grid(
                distance, impulse, step, d_max=D_MAX, dd=DD
            )

            # Normalize
            max_abs = max(np.abs(imp_grid).max(), 1e-10)
            imp_norm = imp_grid / max_abs
            max_abs_s = max(np.abs(step_grid).max(), 1e-10)
            step_norm = step_grid / max_abs_s

            inp = torch.tensor(
                np.stack([imp_norm, step_norm], axis=0), dtype=torch.float32
            ).unsqueeze(0).to(device)

            preds = predict_all(models, inp)

            fig, axes = plt.subplots(4, 1, figsize=(14, 12),
                                     gridspec_kw={"height_ratios": [1.2, 1, 1, 1]})

            # Subplot 0: Input responses
            ax = axes[0]
            ax.plot(GRID, imp_norm, label="脉冲响应 (Impulse)", color="#333333",
                    alpha=0.8, linewidth=0.8)
            ax2 = ax.twinx()
            ax2.plot(GRID, step_norm, label="阶跃响应 (Step)", color="#888888",
                     alpha=0.8, linewidth=0.8)
            ax.set_title(f"{subdir_name} / {csv_path.name} — 输入信号", fontsize=11)
            ax.set_ylabel("脉冲响应幅值")
            ax2.set_ylabel("阶跃响应幅值")
            ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
            ax2.legend(loc="upper right", fontsize=8, framealpha=0.8)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.set_xlim(0, D_MAX)

            # Subplots 1-3: Each model's prediction
            for row, (name, info) in enumerate(EXPERIMENTS.items(), 1):
                ax = axes[row]
                pred = preds[name]

                ax.plot(GRID, pred, color=info["color"], linewidth=1.5,
                        label=f"{name} 预测", alpha=0.9)

                # Mark peaks
                pred_peaks, _ = find_peaks(pred, height=0.2, distance=4, prominence=0.1)
                if len(pred_peaks) > 0:
                    peak_positions = GRID[pred_peaks]
                    ax.scatter(peak_positions, pred[pred_peaks], c=info["color"],
                               marker="^", s=60, zorder=5, label="预测峰值")
                    for pp in peak_positions:
                        ax.annotate(f"{pp:.1f}m", xy=(pp, 0.95), fontsize=7,
                                    ha="center", va="top", color=info["color"])

                ax.set_ylabel("置信度")
                ax.set_title(f"{name} 诊断结果", fontsize=10)
                ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
                ax.grid(True, alpha=0.3, linewidth=0.5)
                ax.set_xlim(0, D_MAX)
                ax.set_ylim(-0.05, 1.15)

            axes[-1].set_xlabel("距离 / m")

            plt.tight_layout()
            out_path = OUTPUT_DIR / f"fig_real_sample_{fig_idx}.png"
            plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {out_path}")

        except Exception as e:
            print(f"  Error processing {csv_path.name}: {e}")
            continue


# ════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Comparison Figure Generator")
    print("=" * 60)

    # Setup fonts
    print("\nSetting up fonts...")
    setup_fonts()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load models
    print("\nLoading models...")
    models = load_all_models(device)

    # 1. Loss comparison
    gen_loss_comparison()

    # 1b. Metrics summary
    gen_metrics_summary()

    # 2. Prediction examples (synthetic test data)
    # Load manifest to find RG58-74M matching samples
    manifest_path = Path(PROJECT_ROOT) / "DataSet" / "manifest.yaml"
    with open(manifest_path, "r") as f:
        manifest = yaml.safe_load(f)

    # Build mapping: manifest_index -> test_dataset_index
    all_samples = manifest["samples"]
    test_only = [s for s in all_samples if s["split"] == "test"]
    manifest_to_test = {}
    test_idx = 0
    for man_idx, s in enumerate(all_samples):
        if s["split"] == "test":
            manifest_to_test[man_idx] = test_idx
            test_idx += 1

    test_samples_with_idx = [(manifest_to_test[i], s)
                             for i, s in enumerate(all_samples) if s["split"] == "test"]

    # Find ~74m samples with 1 defect (RG58-74M simulation)
    rg58_candidates = [(ti, s) for ti, s in test_samples_with_idx
                       if 68 <= s["total_length_m"] <= 80 and s["n_defects"] == 1]
    rg58_candidates.sort(key=lambda x: abs(x[1]["total_length_m"] - 74))

    # Also get some multi-defect and long cable samples for diversity
    multi_defect = [(ti, s) for ti, s in test_samples_with_idx if s["n_defects"] >= 2]
    long_cable = [(ti, s) for ti, s in test_samples_with_idx if s["total_length_m"] > 400]

    # Select 6 samples: 3 RG58-74M + 2 multi-defect + 1 long cable
    selected_indices = []
    selected_labels = []

    # RG58-74M samples (3)
    for ti, s in rg58_candidates[:3]:
        selected_indices.append(ti)
        selected_labels.append(
            f"RG58-74M 模拟样本: {s['sample_id']} "
            f"(总长={s['total_length_m']:.1f}m, {s['n_defects']}个缺陷, "
            f"εr={s['epsr']:.3f})"
        )

    # Multi-defect samples (2)
    random.seed(42)
    multi_sampled = random.sample(multi_defect, min(2, len(multi_defect)))
    for ti, s in multi_sampled:
        selected_indices.append(ti)
        selected_labels.append(
            f"多缺陷样本: {s['sample_id']} "
            f"(总长={s['total_length_m']:.1f}m, {s['n_defects']}个缺陷, "
            f"εr={s['epsr']:.3f})"
        )

    # Long cable sample (1)
    long_sampled = random.sample(long_cable, min(1, len(long_cable)))
    for ti, s in long_sampled:
        selected_indices.append(ti)
        selected_labels.append(
            f"长距离样本: {s['sample_id']} "
            f"(总长={s['total_length_m']:.1f}m, {s['n_defects']}个缺陷, "
            f"εr={s['epsr']:.3f})"
        )

    # Load test dataset
    print("\nLoading test dataset...")
    first_cfg = next(iter(models.values()))[1]
    ds = CableDefectDataset(
        manifest_path=str(manifest_path),
        split="test",
        channels=first_cfg.data.channels,
        augment=False,
    )

    gen_pred_examples(models, selected_indices, ds, selected_labels, device)

    # 3. Real data inference
    gen_real_inference(models, REAL_DATA_DIRS, device)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
