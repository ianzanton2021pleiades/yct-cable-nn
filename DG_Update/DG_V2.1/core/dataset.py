"""PyTorch dataset loader for DG V2.1 outputs.

The loader always recomputes the distance responses from the selected S11 CSV
with the shared Client IFFT.  DG V2.1 guarantees that these responses are the same
as the response columns exported by the generator.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import yaml

from core.tdr_signal import s11_to_responses, to_fixed_distance_grid

D_MAX = 2500.0
DD = 0.25
N_GRID = int(round(D_MAX / DD))
WINDOW = "hann"


class CableDefectDataset(Dataset):
    """Load one DG V2.1 split and return aligned input/label tensors.

    Args:
        manifest_path: path to ``manifest.yaml``.
        split: train/val/test.
        channels: any subset of ``impulse`` and ``step``.  Each requested
            channel is returned twice: global-normalized and local-normalized.
        band: ``1GHz`` or ``200MHz``.
        d_max, dd: fixed distance grid; defaults match the V2.1 label grid.
        window: IFFT window.  Keep this equal to the generation window recorded
            in metadata unless intentionally training a window-robust model.
        augment: enable train-time signal augmentation.
    """

    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        channels: List[str] | None = None,
        band: str = "1GHz",
        d_max: float = D_MAX,
        dd: float = DD,
        window: str = WINDOW,
        augment: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.split = str(split)
        self.channels = list(channels or ["impulse", "step"])
        self.band = "200MHz" if str(band).lower().replace(" ", "") in {"200mhz", "200m", "0.2ghz"} else "1GHz"
        self.d_max = float(d_max)
        self.dd = float(dd)
        self.n_grid = int(round(self.d_max / self.dd))
        self.window = str(window)
        self.augment = bool(augment)

        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest = yaml.safe_load(handle)
        all_samples = self.manifest.get("samples", [])
        self.samples = [sample for sample in all_samples if sample.get("split") == self.split]
        self.raw_dir = self.manifest_path.parent / "raw" / self.split
        self.label_dir = self.manifest_path.parent / "labels" / self.split

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        sample_id = str(sample["sample_id"])
        csv_path = self._csv_path(sample, sample_id)
        freqs, s11 = self._read_csv(csv_path)
        epsr = self._get_epsr(sample_id, sample)

        distance, impulse, step, _ = s11_to_responses(
            freqs, s11, epsr=epsr, window=self.window
        )
        _, imp_grid, step_grid = to_fixed_distance_grid(
            distance, impulse, step, d_max=self.d_max, dd=self.dd
        )

        channel_data: list[np.ndarray] = []
        skip_idx = min(int(round(15.0 / self.dd)), self.n_grid)
        for channel in self.channels:
            if channel == "impulse":
                raw = imp_grid.copy()
            elif channel == "step":
                raw = step_grid.copy()
            else:
                raise ValueError(f"Unknown channel: {channel}")

            global_max = max(float(np.max(np.abs(raw))), 1e-10)
            channel_data.append(raw / global_max)
            tail = raw[skip_idx:] if skip_idx < len(raw) else raw
            local_max = max(float(np.max(np.abs(tail))), 1e-10)
            local = raw / local_max
            local[:skip_idx] = np.clip(local[:skip_idx], -1.0, 1.0)
            channel_data.append(local)

        inputs = torch.as_tensor(np.stack(channel_data), dtype=torch.float32)
        label_path = self._label_path(sample, sample_id)
        label = np.load(label_path).astype(np.float32)
        if len(label) != self.n_grid:
            raise ValueError(
                f"Label/input grid mismatch for {sample_id}: label={len(label)}, input={self.n_grid}. "
                "Use d_max=2500 and dd=0.25 for the standard DG V2.1 dataset."
            )
        labels = torch.as_tensor(label, dtype=torch.float32)

        if self.augment and self.split == "train":
            inputs, labels = self._augment(inputs, labels)
        return inputs, labels

    def _csv_path(self, sample: dict, sample_id: str) -> Path:
        key = "csv_200mhz" if self.band == "200MHz" else "csv_1ghz"
        relative = sample.get(key)
        if relative:
            path = self.manifest_path.parent / str(relative)
        else:
            suffix = "200MHz" if self.band == "200MHz" else "1GHz"
            path = self.raw_dir / f"{sample_id}_{suffix}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def _label_path(self, sample: dict, sample_id: str) -> Path:
        relative = sample.get("label")
        path = self.manifest_path.parent / str(relative) if relative else self.label_dir / f"{sample_id}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    @staticmethod
    def _read_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        import pandas as pd

        frame = pd.read_csv(csv_path, header=0)
        freq_col = next(column for column in frame.columns if "freq" in column.lower())
        real_col = next(column for column in frame.columns if "real" in column.lower())
        imag_col = next((column for column in frame.columns if "imag" in column.lower()), None)
        freqs = frame[freq_col].to_numpy(dtype=np.float64)
        real = frame[real_col].to_numpy(dtype=np.float64)
        imag = frame[imag_col].to_numpy(dtype=np.float64) if imag_col else np.zeros_like(real)
        valid = np.isfinite(freqs) & (freqs > 0) & np.isfinite(real) & np.isfinite(imag)
        return freqs[valid], real[valid] + 1j * imag[valid]

    def _get_epsr(self, sample_id: str, sample: dict) -> float:
        yaml_path = self.raw_dir / f"{sample_id}.yaml"
        if yaml_path.exists():
            try:
                with yaml_path.open("r", encoding="utf-8") as handle:
                    metadata = yaml.safe_load(handle)
                if metadata and "epsr" in metadata:
                    return float(metadata["epsr"])
            except (OSError, TypeError, ValueError, yaml.YAMLError):
                pass
        return float(sample.get("epsr", 2.23))

    @staticmethod
    def _augment(inputs: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = inputs + torch.randn_like(inputs) * 0.01
        for channel in range(inputs.shape[0]):
            inputs[channel] *= 1.0 + (torch.rand(1).item() - 0.5) * 0.16

        # Distance shifts must be applied to labels as well as inputs.  V1 only
        # shifted the input, silently creating incorrect supervision.
        shift = int(torch.randint(-2, 3, (1,)).item())
        if shift:
            inputs = torch.roll(inputs, shifts=shift, dims=-1)
            labels = torch.roll(labels, shifts=shift, dims=-1)
            if shift > 0:
                labels[:shift] = 0.0
            else:
                labels[shift:] = 0.0

        if torch.rand(1).item() < 0.3:
            alpha = torch.rand(1).item() * 0.1
            positions = torch.linspace(0, 1, inputs.shape[-1], device=inputs.device)
            inputs = inputs * torch.exp(-alpha * positions).unsqueeze(0)
        return inputs, labels
