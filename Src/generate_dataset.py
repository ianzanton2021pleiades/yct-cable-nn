"""
generate_dataset.py — 合成数据集生成 CLI

按照 Client 导出格式生成带标签的 S11 数据集。
每个样本包含:
  - <id>.csv: Frequency,S11_Real,S11_Imaginary,Distance,ImpulseResponse,StepResponse
  - <id>.yaml: 元数据（缺陷位置/严重度/末端/电缆长/epsr/种子等）
  - <id>.npy: 固定距离网格上的置信度标签向量

划分 train/val/test 并生成 manifest.yaml。

用法:
    python generate_dataset.py --output_dir ../DataSet --n_total 3000 --seed 2024
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# 添加 Src 到路径
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR
sys.path.insert(0, str(SRC_DIR))

from core.s11_generator import (
    generate_s11, generate_random_cable, generate_sample, SweepConfig, CableSample
)
from core.tdr_signal import s11_to_responses, to_fixed_distance_grid
from core.label import build_label_vector

# ═══════════════════════════════════════════════
# 固定网格参数
# ═══════════════════════════════════════════════
D_MAX = 1200.0  # 最大距离 (m)
DD = 0.5        # 距离步长 (m)
N_GRID = int(round(D_MAX / DD))  # 2400 点


def save_sample_csv(
    csv_path: Path,
    freq_hz: np.ndarray,
    s11: np.ndarray,
    distance: np.ndarray,
    impulse: np.ndarray,
    step: np.ndarray,
    cable_length: float,
) -> None:
    """
    保存单样本 CSV（Client 导出格式）。
    Distance/ImpulseResponse/StepResponse 截断到电缆长度×1.2。
    """
    cutoff = cable_length * 1.2
    mask = distance <= cutoff
    # 取最后一个 <= cutoff 的点，至少保留10个点
    indices = np.where(mask)[0]
    if len(indices) < 10:
        n_keep = min(len(distance), 100)
    else:
        n_keep = indices[-1] + 1

    d_out = distance[:n_keep]
    imp_out = np.real(impulse[:n_keep])
    step_out = step[:n_keep]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Frequency", "S11_Real", "S11_Imaginary",
                         "Distance", "ImpulseResponse", "StepResponse"])
        for i in range(len(freq_hz)):
            writer.writerow([
                f"{freq_hz[i]:.1f}",
                f"{s11[i].real:.10f}",
                f"{s11[i].imag:.10f}",
                "", ""  # 占位，距离/响应按最长列填充
            ])
        # 在频率行后，填充距离/响应列
        # 重新读取并补写 — 改用一次性写入
    # 重写为正确格式
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Frequency", "S11_Real", "S11_Imaginary",
                         "Distance", "ImpulseResponse", "StepResponse"])
        max_rows = max(len(freq_hz), len(d_out))
        for i in range(max_rows):
            row = []
            if i < len(freq_hz):
                row.extend([f"{freq_hz[i]:.1f}", f"{s11[i].real:.10f}", f"{s11[i].imag:.10f}"])
            else:
                row.extend(["", "", ""])
            if i < len(d_out):
                row.extend([f"{d_out[i]:.4f}", f"{imp_out[i]:.10f}", f"{step_out[i]:.10f}"])
            else:
                row.extend(["", "", ""])
            writer.writerow(row)


def save_metadata_yaml(
    yaml_path: Path,
    sample_id: str,
    cable: CableSample,
    defect_info: list,
    total_length: float,
    epsr: float,
    seed: int,
    split: str,
    joint_positions: list = None,
) -> None:
    """保存样本元数据 YAML"""
    meta = {
        "sample_id": sample_id,
        "total_length_m": float(round(total_length, 4)),
        "epsr": float(round(epsr, 4)),
        "seed": int(seed),
        "split": split,
        "n_segments": int(len(cable.segments)),
        "defects": [
            {
                "position_m": float(round(d["position"], 4)),
                "length_m": float(round(d["length"], 4)),
                "z0_ohm": float(round(d["z0"], 4)),
                "epsr": float(round(d["epsr"], 4)),
                "severity": float(round(d["severity"], 4)),
            }
            for d in defect_info
        ],
        "joint_positions_m": [float(round(p, 4)) for p in (joint_positions or [])],
        "end_position_m": float(round(total_length, 4)),
        "has_joint_reflections": cable.has_joint_reflections,
    }
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(meta, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def generate_one_sample(
    seed: int,
    sample_id: str,
    split: str,
    output_dir: Path,
    sweep: SweepConfig,
    fixed_grid: np.ndarray,
) -> dict:
    """
    生成单个样本并保存文件。

    Returns:
        manifest entry dict
    """
    rng = np.random.RandomState(seed)

    # 生成随机电缆
    cable = generate_random_cable(rng)
    cable.has_joint_reflections = True

    # 生成 S11
    freq_hz, s11 = generate_s11(cable, sweep, rng=rng,
                                add_noise=True, inject_joints=True)

    # IFFT → 距离域
    distance, impulse, step, Z = s11_to_responses(
        freq_hz, s11, epsr=cable.epsr, window='hann'
    )

    # 固定网格重采样
    grid, imp_grid, step_grid = to_fixed_distance_grid(
        distance, impulse, step, d_max=D_MAX, dd=DD
    )

    # 标签
    defect_info = cable.defect_info
    defect_positions = [d["position"] for d in defect_info]
    severities = [d["severity"] for d in defect_info]
    joint_positions = getattr(cable, 'joint_positions', [])
    label = build_label_vector(
        defect_positions, severities,
        cable.total_length, grid,
        joint_positions=joint_positions,
    )

    # ─── 保存文件 ───
    split_dir = output_dir / "raw" / split
    label_dir = output_dir / "labels" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = split_dir / f"{sample_id}.csv"
    save_sample_csv(csv_path, freq_hz, s11, distance, impulse, step, cable.total_length)

    # YAML 元数据
    yaml_path = split_dir / f"{sample_id}.yaml"
    save_metadata_yaml(yaml_path, sample_id, cable, defect_info,
                      cable.total_length, cable.epsr, seed, split,
                      joint_positions=joint_positions)

    # 标签 npy
    npy_path = label_dir / f"{sample_id}.npy"
    np.save(npy_path, label.astype(np.float32))

    return {
        "sample_id": sample_id,
        "total_length_m": float(round(cable.total_length, 2)),
        "epsr": float(round(cable.epsr, 3)),
        "n_defects": int(len(defect_info)),
        "n_joints": int(len(joint_positions)),
        "seed": int(seed),
        "split": split,
    }


def main():
    parser = argparse.ArgumentParser(description="合成电缆缺陷数据集生成器")
    parser.add_argument("--output_dir", type=str, default=str(SCRIPT_DIR.parent / "DataSet"),
                        help="数据集输出目录")
    parser.add_argument("--n_total", type=int, default=3000, help="总样本数")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.133, help="验证集比例")
    parser.add_argument("--seed", type=int, default=2024, help="主随机种子")
    parser.add_argument("--start_seed", type=int, default=10000, help="样本种子起始值")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    n_total = args.n_total
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    n_test = n_total - n_train - n_val

    print(f"数据集生成配置:")
    print(f"  输出目录: {output_dir}")
    print(f"  总样本数: {n_total} (train={n_train}, val={n_val}, test={n_test})")
    print(f"  主种子: {args.seed}")

    sweep = SweepConfig()
    fixed_grid = np.linspace(0, D_MAX, N_GRID, endpoint=False) + DD / 2

    master_rng = np.random.RandomState(args.seed)
    seeds = master_rng.randint(0, 2**31, size=n_total)
    # 样本ID用起始种子偏移
    sample_ids = [f"s{args.start_seed + i:06d}" for i in range(n_total)]

    # 划分
    indices = np.random.RandomState(args.seed).permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    split_map = {}
    for i in train_idx:
        split_map[i] = "train"
    for i in val_idx:
        split_map[i] = "val"
    for i in test_idx:
        split_map[i] = "test"

    manifest = {
        "n_total": n_total,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "seed": args.seed,
        "start_seed": args.start_seed,
        "fixed_grid": {"d_max_m": D_MAX, "dd_m": DD, "n_points": N_GRID},
        "sweep": {"start_hz": sweep.start_hz, "stop_hz": sweep.stop_hz,
                  "n_points": sweep.n_points},
        "window": "hann",
        "samples": [],
    }

    t0 = time.time()
    for i in range(n_total):
        split = split_map[i]
        sid = sample_ids[i]
        seed = seeds[i]

        entry = generate_one_sample(seed, sid, split, output_dir, sweep, fixed_grid)
        manifest["samples"].append(entry)

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_total - i - 1) / rate
            print(f"  [{i+1:4d}/{n_total}] {sid} ({split}) "
                  f"L={entry['total_length_m']:.0f}m, "
                  f"defects={entry['n_defects']}, "
                  f"rate={rate:.1f}/s, ETA={eta:.0f}s")

    # 保存 manifest
    manifest_path = output_dir / "manifest.yaml"
    with manifest_path.open("w", encoding="utf-8") as f:
        yaml.dump(manifest, f, allow_unicode=True, default_flow_style=False)

    elapsed = time.time() - t0
    print(f"\n✅ 数据集生成完成！")
    print(f"  耗时: {elapsed:.1f}s ({n_total/elapsed:.1f} samples/s)")
    print(f"  输出: {output_dir}")
    print(f"    raw/train/: {n_train} CSV + YAML")
    print(f"    raw/val/:   {n_val} CSV + YAML")
    print(f"    raw/test/:  {n_test} CSV + YAML")
    print(f"    labels/{split}/: 对应 .npy")
    print(f"    manifest.yaml")


if __name__ == "__main__":
    main()
