"""
generate_core_figures.py
Run 3-model inference on Core- CSV files from RG58-74M(40+4+30) and generate comparison figures.
"""
import os
import sys
import json

PROJECT_ROOT = r"D:\GitRepository\cable-defect-location-method-based-on-BIS\AI_TEST"
os.chdir(PROJECT_ROOT)
sys.path.insert(0, "Src")

import numpy as np
import torch
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
from core.tdr_signal import read_s11_csv, s11_to_responses, to_fixed_distance_grid

# ── Constants ──
D_MAX = 1200.0
DD = 0.5
N_GRID = int(round(D_MAX / DD))
GRID = np.linspace(0, D_MAX, N_GRID, endpoint=False) + DD / 2
DPI = 300
OUTPUT_DIR = Path(PROJECT_ROOT) / "[V1]1D-UNet" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RG58_DIR = Path(PROJECT_ROOT) / "REF" / "RG58-74M(40+4+30)"

# Select 6 representative Core files
CORE_FILES = [
    "Core-LineA+CUT1+LineB(20degree)-1.csv",
    "Core-LineA+CUT1+LineB(40degree)-1.csv",
    "Core-LineA+CUT1+LineB(60degree)-1.csv",
    "Core-LineA+CUT2+LineB(0.5circle)-1.csv",
    "Core-LineA+CUT2+LineB(5.5circle)-1.csv",
    "Core-LineA+CUT2+LineB(10.5circle)-1.csv",
]

EXPERIMENTS = {
    "V1 UNet1D": {
        "config": "AgentsStorage/experiments/exp_20260617_150349_unet1d_medium_baseline/config.yaml",
        "checkpoint": "AgentsStorage/experiments/exp_20260617_150349_unet1d_medium_baseline/best.pt",
        "color": "#1f77b4",
    },
    "UNet1D V2": {
        "config": "AgentsStorage/experiments/exp_20260618_170217_unet1d_v2_medium/config.yaml",
        "checkpoint": "AgentsStorage/experiments/exp_20260618_170217_unet1d_v2_medium/best.pt",
        "color": "#d62728",
    },
    "CableFormer V2": {
        "config": "AgentsStorage/experiments/exp_20260617_175743_cableformer_medium_v2/config.yaml",
        "checkpoint": "AgentsStorage/experiments/exp_20260617_175743_cableformer_medium_v2/best.pt",
        "color": "#2ca02c",
    },
}


# ── Font setup ──
def setup_fonts():
    cache_dir = matplotlib.get_cachedir()
    for p in Path(cache_dir).glob("fontlist-*.json"):
        try:
            p.unlink()
        except Exception:
            pass
    fm._load_fontmanager(try_read_cache=False)

    simsun_path = None
    for name in ["simsun.ttc", "SIMSUN.TTC", "simsun.ttf"]:
        fp = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / name
        if fp.exists():
            simsun_path = str(fp)
            break

    if simsun_path is None:
        try:
            found = fm.findfont(fm.FontProperties(family="SimSun"), fallback_to_default=False)
            if found and "simsun" in found.lower():
                simsun_path = found
        except Exception:
            pass

    if simsun_path:
        print(f"  Font: {simsun_path}")
        fp = fm.FontProperties(fname=simsun_path)
        font_name = fp.get_name()
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = [font_name, "Times New Roman"]
    else:
        print("  WARNING: SimSun not found")
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman", "SimSun"]

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


# ── Model loading ──
def build_model(cfg):
    if cfg.model.name == "cableformer":
        return CableFormer(
            base_ch=cfg.model.base_ch, in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            transformer_dim=cfg.model.transformer_dim,
            n_transformer_blocks=cfg.model.n_transformer_blocks,
            n_heads=cfg.model.n_heads, use_cbam=cfg.model.use_cbam,
        )
    elif cfg.model.name == "unet1d_v2":
        return UNet1DV2(
            base_ch=cfg.model.base_ch, in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels, use_cbam=cfg.model.use_cbam,
        )
    else:
        return UNet1D(
            base_ch=cfg.model.base_ch, in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels, kernel_size=cfg.model.kernel_size,
        )


def load_all_models(device):
    models = {}
    for name, info in EXPERIMENTS.items():
        cfg = load_config(info["config"])
        model = build_model(cfg)
        ckpt = torch.load(info["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        models[name] = model
        print(f"  Loaded {name}")
    return models


def predict_all(models, input_tensor):
    results = {}
    with torch.no_grad():
        for name, model in models.items():
            _, probs = model(input_tensor)
            results[name] = probs[0].cpu().numpy()
    return results


def make_nice_title(filename):
    """Convert filename to a readable title."""
    # e.g. Core-LineA+CUT1+LineB(20degree)-1.csv -> Core-LineA+CUT1+LineB (20°)
    name = filename.replace(".csv", "")
    name = name.replace("degree)", "°)")
    name = name.replace("circle)", "圈)")
    name = name.replace("CUT1", "缺陷1(角度)")
    name = name.replace("CUT2", "缺陷2(环形)")
    return name


def gen_core_figure(models, csv_path, fig_idx, device):
    """Generate one figure for a Core CSV file."""
    try:
        freqs, S = read_s11_csv(str(csv_path))
        epsr = 2.23
        distance, impulse, step, Z = s11_to_responses(freqs, S, epsr=epsr, window="hann")
        _, imp_grid, step_grid = to_fixed_distance_grid(
            distance, impulse, step, d_max=D_MAX, dd=DD
        )

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
        title = make_nice_title(csv_path.name)
        ax.set_title(f"RG58-74M(40+4+30) {title} — 输入信号", fontsize=11)
        ax.set_ylabel("脉冲响应幅值")
        ax2.set_ylabel("阶跃响应幅值")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
        ax2.legend(loc="upper right", fontsize=8, framealpha=0.8)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_xlim(0, D_MAX)

        # Subplots 1-3: Each model
        for row, (name, info) in enumerate(EXPERIMENTS.items(), 1):
            ax = axes[row]
            pred = preds[name]
            color = info["color"]

            ax.plot(GRID, pred, color=color, linewidth=1.5,
                    label=f"{name} 预测", alpha=0.9)

            pred_peaks, _ = find_peaks(pred, height=0.2, distance=4, prominence=0.1)
            if len(pred_peaks) > 0:
                peak_positions = GRID[pred_peaks]
                ax.scatter(peak_positions, pred[pred_peaks], c=color,
                           marker="^", s=60, zorder=5, label="预测峰值")
                for pp in peak_positions:
                    ax.annotate(f"{pp:.1f}m", xy=(pp, 0.95), fontsize=7,
                                ha="center", va="top", color=color)

            ax.set_ylabel("置信度")
            ax.set_title(f"{name} 诊断结果", fontsize=10)
            ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.set_xlim(0, D_MAX)
            ax.set_ylim(-0.05, 1.15)

        axes[-1].set_xlabel("距离 / m")

        plt.tight_layout()
        out_path = OUTPUT_DIR / f"fig_core_sample_{fig_idx}.png"
        plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")
        return True

    except Exception as e:
        print(f"  Error: {csv_path.name}: {e}")
        return False


def main():
    print("=" * 60)
    print("Core Cable Figure Generator")
    print("=" * 60)

    print("\nSetting up fonts...")
    setup_fonts()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    print("\nLoading models...")
    models = load_all_models(device)

    print(f"\nGenerating {len(CORE_FILES)} Core cable figures...")
    for fig_idx, fname in enumerate(CORE_FILES, 1):
        csv_path = RG58_DIR / fname
        if not csv_path.exists():
            print(f"  NOT FOUND: {csv_path}")
            continue
        gen_core_figure(models, csv_path, fig_idx, device)

    print(f"\nAll figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
