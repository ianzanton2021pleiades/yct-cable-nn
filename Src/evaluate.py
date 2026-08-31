"""
evaluate.py - model evaluation and visualization

Usage:
    python Src/evaluate.py --config Src/configs/default.yaml \
                           --checkpoint AgentsStorage/experiments/exp_xxx/best.pt \
                           --split test
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── matplotlib 全局样式 ──
# 宋体(SimSun)处理中文，Times New Roman 处理英文/数字，同属 serif 族
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["SimSun", "Times New Roman"],
    "mathtext.fontset":  "stix",          # 数学公式也用 serif 风格
    "axes.unicode_minus": False,          # 负号不被替换为全角
    "xtick.direction":   "in",            # 刻度朝内
    "ytick.direction":   "in",
    "xtick.top":         True,            # 顶轴也显示刻度
    "ytick.right":       True,            # 右轴也显示刻度
})

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import load_config, Config
from models.unet1d import UNet1D
from models.unet1d_v2 import UNet1DV2
from models.cableformer import CableFormer
from core.dataset import CableDefectDataset
from core.tdr_signal import s11_to_responses, to_fixed_distance_grid
from metrics import compute_sample_metrics, aggregate_metrics, SampleMetrics


D_MAX = 1200.0
DD = 0.5
N_GRID = int(round(D_MAX / DD))


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
    elif cfg.model.name == "unet1d_v2":
        model = UNet1DV2(
            base_ch=cfg.model.base_ch,
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            use_cbam=cfg.model.use_cbam,
        )
    else:
        model = UNet1D(
            base_ch=cfg.model.base_ch,
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            kernel_size=cfg.model.kernel_size,
        )
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate_split(
    model: UNet1D,
    cfg: Config,
    split: str,
    device: torch.device,
) -> tuple[list[SampleMetrics], list[np.ndarray], list[np.ndarray], list[float]]:
    """Run evaluation on a split, return per-sample metrics, preds, labels, end_positions."""
    ds = CableDefectDataset(
        manifest_path=cfg.data.manifest,
        split=split,
        channels=cfg.data.channels,
        augment=False,
    )
    loader = DataLoader(ds, batch_size=cfg.train.batch_size,
                        shuffle=False, num_workers=cfg.data.num_workers)

    grid = np.linspace(0, D_MAX, N_GRID, endpoint=False) + DD / 2
    all_metrics = []
    all_preds = []
    all_labels = []
    end_positions = []

    idx = 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=f"Evaluating {split}"):
            inputs = inputs.to(device)
            _, probs = model(inputs)

            for i in range(probs.shape[0]):
                pred_np = probs[i].cpu().numpy()
                label_np = labels[i].numpy()
                all_preds.append(pred_np)
                all_labels.append(label_np)

                # Get end position from manifest
                sample = ds.samples[idx]
                end_pos = sample.get("total_length_m", None)
                if end_pos is not None:
                    end_positions.append(end_pos)

                m = compute_sample_metrics(
                    pred_np, label_np, grid,
                    peak_height=cfg.eval.find_peaks_height,
                    peak_distance=cfg.eval.find_peaks_distance,
                    peak_prominence=cfg.eval.find_peaks_prominence,
                    tolerance_m=cfg.eval.peak_tolerance_m,
                    end_pos=end_pos,
                )
                all_metrics.append(m)
                idx += 1

    return all_metrics, all_preds, all_labels, end_positions


def plot_training_curves(exp_dir: Path, image_dir: Path):
    """Plot loss and metric curves from train_log.csv."""
    import csv
    log_path = exp_dir / "train_log.csv"
    if not log_path.exists():
        print(f"Warning: {log_path} not found, skipping training curves")
        return

    epochs, train_loss, val_loss, val_recall, val_precision = [], [], [], [], []
    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            val_recall.append(float(row["val_recall"]))
            val_precision.append(float(row["val_precision"]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_loss, label="train_loss")
    axes[0].plot(epochs, val_loss, label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_recall, label="val_recall")
    axes[1].plot(epochs, val_precision, label="val_precision")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric")
    axes[1].set_title("Validation Metrics")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(image_dir / "phase2_train_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: phase2_train_curve.png")


def plot_prediction_examples(
    model: UNet1D,
    cfg: Config,
    device: torch.device,
    image_dir: Path,
    n_examples: int = 6,
):
    """Plot prediction examples: input responses + pred vs label."""
    from scipy.signal import find_peaks

    ds = CableDefectDataset(
        manifest_path=cfg.data.manifest,
        split="test",
        channels=cfg.data.channels,
        augment=False,
    )
    grid = np.linspace(0, D_MAX, N_GRID, endpoint=False) + DD / 2

    n = min(n_examples, len(ds))
    indices = np.random.choice(len(ds), n, replace=False)

    fig, axes = plt.subplots(n, 2, figsize=(14, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    model.eval()
    with torch.no_grad():
        for row, idx in enumerate(indices):
            inputs, labels = ds[idx]
            _, probs = model(inputs.unsqueeze(0).to(device))
            pred = probs[0].cpu().numpy()
            label = labels.numpy()
            imp = inputs[0].numpy()    # impulse global
            step = inputs[2].numpy()   # step global

            # Left: input responses
            ax = axes[row, 0]
            ax.plot(grid, imp, label="Impulse", alpha=0.7, linewidth=0.8)
            ax.plot(grid, step, label="Step", alpha=0.7, linewidth=0.8)
            ax.set_title(f"Sample {idx} - Input")
            ax.set_xlabel("Distance (m)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Right: pred vs label
            ax = axes[row, 1]
            ax.plot(grid, label, "g--", label="Label", alpha=0.7, linewidth=1.0)
            ax.plot(grid, pred, "r-", label="Prediction", alpha=0.8, linewidth=1.0)

            # Mark peaks
            label_peaks, _ = find_peaks(label, height=0.2, distance=4, prominence=0.1)
            pred_peaks, _ = find_peaks(pred, height=0.2, distance=4, prominence=0.1)
            if len(label_peaks) > 0:
                ax.scatter(grid[label_peaks], label[label_peaks], c="green",
                           marker="v", s=50, zorder=5, label="Label peaks")
            if len(pred_peaks) > 0:
                ax.scatter(grid[pred_peaks], pred[pred_peaks], c="red",
                           marker="^", s=50, zorder=5, label="Pred peaks")

            ax.set_title(f"Sample {idx} - Prediction vs Label")
            ax.set_xlabel("Distance (m)")
            ax.set_ylabel("Confidence")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(image_dir / "phase2_pred_examples.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: phase2_pred_examples.png")


def plot_error_histogram(all_metrics: list[SampleMetrics], image_dir: Path):
    """Plot peak localization error distribution."""
    errors = [m.loc_error_mean for m in all_metrics if m.loc_error_mean > 0]
    if not errors:
        print("  No localization errors to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Localization Error (m)")
    ax.set_ylabel("Count")
    ax.set_title(f"Peak Localization Error Distribution (mean={np.mean(errors):.2f}m)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(image_dir / "phase2_error_hist.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: phase2_error_hist.png")


def plot_end_recall(all_metrics: list[SampleMetrics], end_positions: list[float],
                    image_dir: Path):
    """Plot end recall vs cable length."""
    if not end_positions or len(end_positions) != len(all_metrics):
        print("  Skipping end recall plot (missing end positions)")
        return

    # Bin by cable length
    lengths = np.array(end_positions)
    recalled = np.array([m.end_recalled for m in all_metrics], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    n_bins = 10
    bin_edges = np.linspace(lengths.min(), lengths.max(), n_bins + 1)
    bin_recall = []
    bin_centers = []
    for i in range(n_bins):
        mask = (lengths >= bin_edges[i]) & (lengths < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_recall.append(recalled[mask].mean())
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)

    ax.bar(bin_centers, bin_recall, width=(bin_edges[1] - bin_edges[0]) * 0.8,
           edgecolor="black", alpha=0.7)
    ax.set_xlabel("Cable Length (m)")
    ax.set_ylabel("End Recall Rate")
    ax.set_title("End Detection Recall vs Cable Length")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(image_dir / "phase2_end_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: phase2_end_accuracy.png")


def real_data_inference(
    model: UNet1D,
    cfg: Config,
    device: torch.device,
    image_dir: Path,
    real_data_dir: str = r"D:\FDR\u6848\u4f8b-csv\\\u65e0\u6821\u51c6S11",
    n_samples: int = 6,
):
    """Run inference on real (unlabeled) S11 data for qualitative evaluation."""
    from core.tdr_signal import read_s11_csv

    real_dir = Path(real_data_dir)
    if not real_dir.exists():
        print(f"  Warning: real data dir not found: {real_dir}")
        return

    csv_files = sorted(real_dir.glob("*.csv"))
    if not csv_files:
        csv_files = sorted(real_dir.glob("**/*.csv"))
    if not csv_files:
        print(f"  Warning: no CSV files found in {real_dir}")
        return

    n = min(n_samples, len(csv_files))
    selected = csv_files[:n]

    grid = np.linspace(0, D_MAX, N_GRID, endpoint=False) + DD / 2
    fig, axes = plt.subplots(n, 2, figsize=(14, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    model.eval()
    with torch.no_grad():
        for row, csv_path in enumerate(selected):
            try:
                freqs, S = read_s11_csv(str(csv_path))
                epsr = 2.23  # default
                distance, impulse, step, Z = s11_to_responses(freqs, S, epsr=epsr, window="hann")
                _, imp_grid, step_grid = to_fixed_distance_grid(
                    distance, impulse, step, d_max=D_MAX, dd=DD
                )

                # Normalize — 4 channels: impulse_global, impulse_local, step_global, step_local
                skip_idx = int(15.0 / DD)  # skip first 15m

                imp_global_max = max(np.abs(imp_grid).max(), 1e-10)
                imp_global = imp_grid / imp_global_max
                imp_local_max = max(np.abs(imp_grid[skip_idx:]).max(), 1e-10)
                imp_local = imp_grid / imp_local_max
                imp_local[:skip_idx] = np.clip(imp_local[:skip_idx], -1.0, 1.0)

                step_global_max = max(np.abs(step_grid).max(), 1e-10)
                step_global = step_grid / step_global_max
                step_local_max = max(np.abs(step_grid[skip_idx:]).max(), 1e-10)
                step_local = step_grid / step_local_max
                step_local[:skip_idx] = np.clip(step_local[:skip_idx], -1.0, 1.0)

                inp = torch.tensor(
                    np.stack([imp_global, imp_local, step_global, step_local], axis=0),
                    dtype=torch.float32
                ).unsqueeze(0).to(device)

                _, probs = model(inp)
                pred = probs[0].cpu().numpy()

                # Left: input responses
                ax = axes[row, 0]
                ax.plot(grid, imp_global, label="Impulse", alpha=0.7, linewidth=0.8)
                ax.plot(grid, step_global, label="Step", alpha=0.7, linewidth=0.8)
                ax.set_title(f"{csv_path.name}")
                ax.set_xlabel("Distance (m)")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

                # Right: prediction
                ax = axes[row, 1]
                ax.plot(grid, pred, "r-", linewidth=1.0)
                ax.set_title("Predicted Confidence")
                ax.set_xlabel("Distance (m)")
                ax.set_ylabel("Confidence")
                ax.grid(True, alpha=0.3)

            except Exception as e:
                print(f"  Warning: failed on {csv_path.name}: {e}")
                axes[row, 0].set_title(f"{csv_path.name} - ERROR")
                axes[row, 1].set_title("Skipped")

    plt.tight_layout()
    plt.savefig(image_dir / "phase2_real_inference.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: phase2_real_inference.png")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="Src/configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--image_dir", type=str, default="Image")
    parser.add_argument("--real_data_dir", type=str,
                        default=r"D:\FDR\u6848\u4f8b-csv\\\u65e0\u6821\u51c6S11")
    parser.add_argument("--exp_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(cfg, args.checkpoint, device)
    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    # 1. Quantitative evaluation
    print(f"\nEvaluating on {args.split} split...")
    all_metrics, all_preds, all_labels, end_positions = evaluate_split(
        model, cfg, args.split, device
    )
    grid = np.linspace(0, D_MAX, N_GRID, endpoint=False) + DD / 2
    agg = aggregate_metrics(all_metrics, all_preds, all_labels, end_positions, grid)

    print(f"\nResults on {args.split}:")
    print(f"  Recall:          {agg.recall:.4f}")
    print(f"  Precision:       {agg.precision:.4f}")
    print(f"  FAR:             {agg.far:.2f}")
    print(f"  Loc error mean:  {agg.loc_error_mean:.2f} m")
    print(f"  Loc error median:{agg.loc_error_median:.2f} m")
    print(f"  Loc error max:   {agg.loc_error_max:.2f} m")
    print(f"  AUC:             {agg.auc:.4f}")
    print(f"  End recall:      {agg.end_recall:.4f}")

    # Save metrics
    metrics_dict = {
        "split": args.split,
        "recall": agg.recall,
        "precision": agg.precision,
        "far": agg.far,
        "loc_error_mean": agg.loc_error_mean,
        "loc_error_median": agg.loc_error_median,
        "loc_error_max": agg.loc_error_max,
        "auc": agg.auc,
        "end_recall": agg.end_recall,
    }

    # If exp_dir provided, save metrics.json
    if args.exp_dir:
        exp_dir = Path(args.exp_dir)
        with open(exp_dir / "metrics.json", "w") as f:
            json.dump(metrics_dict, f, indent=2)
        plot_training_curves(exp_dir, image_dir)

    # 2. Visualizations
    print("\nGenerating visualizations...")
    plot_prediction_examples(model, cfg, device, image_dir, n_examples=6)
    plot_error_histogram(all_metrics, image_dir)
    plot_end_recall(all_metrics, end_positions, image_dir)

    # 3. Real data inference
    print("\nRunning real data inference...")
    real_data_path = r"D:\FDR案例-csv\无校准S11"
    real_data_inference(model, cfg, device, image_dir, real_data_dir=real_data_path)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
