"""
dataset.py — PyTorch 数据集类

加载合成数据集，使用共享信号库从 S11 重算响应（保证训练==推理代码路径），
重采样到固定距离网格，返回 (input_tensor, label_tensor)。

输入通道可配置（默认 [impulse, step] 双通道）。

用法:
    from core.dataset import CableDefectDataset
    ds = CableDefectDataset(manifest_path="DataSet/manifest.yaml", split="train")
    inputs, labels = ds[0]  # inputs: [2, 2400], labels: [2400]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import yaml

# 添加 Src 到路径
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from core.tdr_signal import s11_to_responses, to_fixed_distance_grid

# ═══════════════════════════════════════════════
# 默认参数
# ═══════════════════════════════════════════════
D_MAX = 1200.0
DD = 0.5
N_GRID = int(round(D_MAX / DD))  # 2400
WINDOW = "hann"


class CableDefectDataset(Dataset):
    """
    电缆缺陷检测数据集。

    从 CSV 文件读取 S11，通过共享信号库重算脉冲/阶跃响应，
    重采样到固定距离网格，返回多通道输入 + 置信度标签。

    Args:
        manifest_path: manifest.yaml 路径
        split: "train" / "val" / "test"
        channels: 输入通道列表，可选 "impulse", "step"
        d_max: 最大距离 (m)
        dd: 距离步长 (m)
        window: IFFT 窗函数
        augment: 是否启用数据增强（微小噪声）
    """

    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        channels: List[str] = None,
        d_max: float = D_MAX,
        dd: float = DD,
        window: str = WINDOW,
        augment: bool = False,
    ):
        if channels is None:
            channels = ["impulse", "step"]

        self.manifest_path = Path(manifest_path)
        self.split = split
        self.channels = channels
        self.d_max = d_max
        self.dd = dd
        self.n_grid = int(round(d_max / dd))
        self.window = window
        self.augment = augment

        # 加载 manifest
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest = yaml.safe_load(f)

        # 筛选当前 split 的样本
        all_samples = self.manifest["samples"]
        self.samples = [s for s in all_samples if s["split"] == split]

        # 确定 raw 和 labels 目录
        self.raw_dir = self.manifest_path.parent / "raw" / split
        self.label_dir = self.manifest_path.parent / "labels" / split

        print(f"CableDefectDataset: split={split}, {len(self.samples)} samples, "
              f"channels={channels}, grid={self.n_grid}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        sample_id = sample["sample_id"]

        # 读取 S11 CSV
        csv_path = self.raw_dir / f"{sample_id}.csv"
        freqs, S = self._read_csv(csv_path)

        # 读取 epsr（从 YAML 或 manifest）
        epsr = self._get_epsr(sample_id, sample)

        # 共享信号库处理 S11 → 响应
        distance, impulse, step, Z = s11_to_responses(
            freqs, S, epsr=epsr, window=self.window
        )

        # 固定网格重采样
        grid, imp_grid, step_grid = to_fixed_distance_grid(
            distance, impulse, step, d_max=self.d_max, dd=self.dd
        )

        # 组装多通道输入
        channel_data = []
        skip_idx = int(15.0 / self.dd)  # 跳过前15m（起始端强反射区域）

        for ch in self.channels:
            if ch == "impulse":
                raw = imp_grid
            elif ch == "step":
                raw = step_grid
            else:
                raise ValueError(f"Unknown channel: {ch}")

            # 通道A: 全局 max-abs 归一化
            global_max = max(np.abs(raw).max(), 1e-10)
            ch_global = raw / global_max
            channel_data.append(ch_global)

            # 通道B: 局部归一化（跳过前15m，放大弱反射信号）
            local_max = max(np.abs(raw[skip_idx:]).max(), 1e-10)
            ch_local = raw / local_max
            ch_local[:skip_idx] = np.clip(ch_local[:skip_idx], -1.0, 1.0)
            channel_data.append(ch_local)

        input_tensor = torch.tensor(
            np.stack(channel_data, axis=0), dtype=torch.float32
        )

        # 加载标签
        label_path = self.label_dir / f"{sample_id}.npy"
        label = np.load(label_path).astype(np.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        # 数据增强
        if self.augment and self.split == "train":
            input_tensor = self._augment(input_tensor)

        return input_tensor, label_tensor

    def _read_csv(self, csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """读取 S11 CSV（兼容多种格式）"""
        import pandas as pd
        df = pd.read_csv(csv_path, header=0)
        freq_col = [c for c in df.columns if "freq" in c.lower()][0]
        real_col = [c for c in df.columns if "real" in c.lower()][0]
        im_cols = [c for c in df.columns if "imag" in c.lower()]

        freqs = df[freq_col].astype(float).values
        re = df[real_col].astype(float).values
        if im_cols:
            im = df[im_cols[0]].astype(float).values
            S = re + 1j * im
        else:
            S = re.astype(np.complex128)

        # 跳过空频率行
        valid = ~np.isnan(freqs) & (freqs > 0)
        return freqs[valid], S[valid]

    def _get_epsr(self, sample_id: str, sample_entry: dict) -> float:
        """获取样本的 epsr"""
        # 优先从 YAML 读
        yaml_path = self.raw_dir / f"{sample_id}.yaml"
        if yaml_path.exists():
            try:
                with open(yaml_path, "r", encoding="utf-8") as f:
                    meta = yaml.safe_load(f)
                if meta and "epsr" in meta:
                    return meta["epsr"]
            except Exception:
                pass
        # 退回到 manifest 或默认
        return sample_entry.get("epsr", 2.23)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """V3 增强: 降低噪声以保留弱反射信号"""
        # 高斯噪声 (V3: sigma 0.02 -> 0.01, 保留弱BNC反射)
        noise = torch.randn_like(x) * 0.01
        x = x + noise
        # 幅度缩放 +-8% (per channel)
        for c in range(x.shape[0]):
            scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.16  # [0.92, 1.08]
            x[c] = x[c] * scale
        # 距离轴抖动 +-2 格
        shift = torch.randint(-2, 3, (1,)).item()
        if shift != 0:
            x = torch.roll(x, shifts=shift, dims=-1)
        # 信号衰减模拟 (V3: 概率从50%降到30%)
        if torch.rand(1).item() < 0.3:
            alpha = torch.rand(1).item() * 0.1  # alpha in [0, 0.1]
            L = x.shape[-1]
            positions = torch.linspace(0, 1, L, device=x.device)
            decay = torch.exp(-alpha * positions)
            x = x * decay.unsqueeze(0)
        return x
