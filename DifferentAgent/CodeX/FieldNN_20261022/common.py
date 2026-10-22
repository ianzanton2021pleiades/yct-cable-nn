"""Shared S11 -> fixed distance-grid inputs and geometric supervision.

No true cable length is used in preprocessing. Scale channels retain amplitude
information lost by per-channel RMS normalization.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FIELD = HERE.parent / "Core16_Field"
sys.path.insert(0, str(FIELD))
from field_responses import algorithm1

DEFAULT_DATA = Path(r"E:\CableNNDataSet\CodeX\FieldNN_20261022")
SOURCE_ROOT = Path(r"E:\20261010截至的全部fdr文件")


def load_config(path=None):
    cfg = json.loads(Path(path or HERE / "config.json").read_text(encoding="utf-8"))
    defaults = HERE / "randomization_defaults.json"
    if defaults.exists():
        value = json.loads(defaults.read_text(encoding="utf-8"))
        values = value.get("randomization", value)
        cfg["randomization"] = {**values, **cfg.get("randomization", {})}
    return cfg


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def distance_grid(cfg):
    r = cfg["response"]
    return np.linspace(0, r["distance_max_m"], r["points"], dtype=np.float64)


def frequency_grid(cfg):
    f = cfg["frequency"]
    grid = np.geomspace if f.get("spacing", "linear") == "log" else np.linspace
    return grid(f["start_hz"], f["stop_hz"], f["points"], dtype=np.float64)


def preprocess_s11(frequency_hz, s11, cfg):
    f, g = np.asarray(frequency_hz, dtype=float), np.asarray(s11, dtype=complex)
    if f.ndim != 1 or g.shape != f.shape or len(f) < 4:
        raise ValueError("S11 must contain at least four frequency/complex pairs")
    if not np.isfinite(f).all() or not np.isfinite(g).all() or np.any(np.diff(f) <= 0):
        raise ValueError("S11 frequency must be finite and strictly increasing; S11 must be finite")
    band = cfg["frequency"]
    if f[0] > band["start_hz"] * (1 + 1e-9) or f[-1] < band["stop_hz"] * (1 - 1e-9):
        raise ValueError("Input does not cover the configured frequency band; do not invent missing frequencies")
    keep = (f >= band["start_hz"]) & (f <= band["stop_hz"])
    # Include the declared edges by interpolation; no extension outside measured data.
    fi = np.unique(np.r_[band["start_hz"], f[keep], band["stop_hz"]])
    gi = np.interp(fi, f, g.real) + 1j * np.interp(fi, f, g.imag)
    r = cfg["response"]
    result = algorithm1(fi, gi, velocity_m_s=r["velocity_m_s"], distance_max_m=r["distance_max_m"])
    d = distance_grid(cfg)
    raw = np.stack([np.interp(d, result["distance_m"], result[key])
                    for key in ("impulse_raw", "step_rho")])
    rms = np.sqrt(np.mean(raw * raw, axis=1)).clip(min=1e-12)
    normalized = raw / rms[:, None]
    scale = np.broadcast_to(np.log10(rms)[:, None], raw.shape)
    return np.concatenate([normalized, scale]).astype(np.float32)


def make_targets(meta, cfg):
    d = distance_grid(cfg)
    sigma = cfg["response"]["event_sigma_m"]
    target = np.zeros((3, len(d)), dtype=np.float32)
    for position in meta["event_positions_m"]:
        target[0] = np.maximum(target[0], np.exp(-0.5 * ((d - position) / sigma) ** 2))
    for start, end in meta["defect_intervals_m"]:
        target[1, (d >= start) & (d <= end)] = 1
    target[2] = np.exp(-0.5 * ((d - meta["terminal_position_m"]) / sigma) ** 2)
    return target


def read_s11(path, sweep=None):
    """Read source files only; FDR decryption lives entirely in memory."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".fdr", ".db8"}:
        if sweep is None:
            raise ValueError("FDR input requires --sweep ROWID; do not guess the phase")
        from analyze_corpus import load_reader_module
        plaintext, _ = load_reader_module().FdrCrypto().decrypt(path.read_bytes())
        with sqlite3.connect(":memory:") as con:
            con.deserialize(plaintext)
            data = np.asarray(con.execute(f'SELECT Freq, Real, Image FROM "MeasureData_{int(sweep)}" ORDER BY Freq').fetchall(), dtype=float)
        return data[:, 0], data[:, 1] + 1j * data[:, 2]
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            if "frequency_hz" in z and "s11" in z:
                return z["frequency_hz"], z["s11"]
            return z["frequency_hz"], z["s11_real"] + 1j * z["s11_imag"]
    data = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8-sig")
    return np.asarray(data["Frequency"]), np.asarray(data["S11_Real"]) + 1j * np.asarray(data["S11_Imaginary"])


class ShardDataset:
    """Read-only NumPy shards; torch conversion occurs in the DataLoader."""
    def __init__(self, root, split):
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.shards = [s for s in manifest["shards"] if s["split"] == split]
        self.ends = np.cumsum([s["count"] for s in self.shards])
        self._open = {}
        self.metadata = []
        for shard in self.shards:
            self.metadata.extend(json.loads(line) for line in (self.root / shard["path"] / "metadata.jsonl").read_text(encoding="utf-8").splitlines())
        if len(self.metadata) != len(self):
            raise ValueError("Shard metadata count disagrees with manifest")

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def __getitem__(self, index):
        shard_index = int(np.searchsorted(self.ends, index, side="right"))
        local = index - (int(self.ends[shard_index - 1]) if shard_index else 0)
        if shard_index not in self._open:
            folder = self.root / self.shards[shard_index]["path"]
            self._open[shard_index] = (np.load(folder / "inputs.npy", mmap_mode="r"), np.load(folder / "targets.npy", mmap_mode="r"))
        x, y = self._open[shard_index]
        return x[local].copy(), y[local].copy(), int(index)
