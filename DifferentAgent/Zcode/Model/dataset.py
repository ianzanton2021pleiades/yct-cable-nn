"""Dataset v3 for DG_V5 synthetic npz samples + field_train_v2 field data.

Per sample: 4-channel input on the fixed GRID_BINS grid (optional 5th
signed-log channel via extra_log), per-class soft targets built from the
event anchors (sigma = max(3 m, 2 % of distance)), and a supervision mask.

v3 changes over round 1:
- valid_len always comes from meta["valid_len"] (field curves stop masking at
  the labelled valid length instead of supervising the zero-padded tail;
  legacy DG_V4 meta without the key falls back to the array length).
- per-pixel positive weight map w(d) = max weight of events covering the
  pixel (default 1.0) — field positives are weighted by engineer/site label
  confidence.
- z0 target: synthetic samples load z0_dev; field samples carry NaN target
  with zero weight (z0 loss contributes 0).
- domain-aware supervision masks (role_cm / mech_cm): field samples learn
  joint/defect role channels and the whole mech head from positive support
  only (no negative push, no hard-negative mining on those channels);
  terminal + background channels are supervised as usual. Synthetic samples
  supervise everything, regardless of their profile ("field"/"rg58").
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

GRID_BINS = 11008
DIST_STEP = 0.25
GRID_M = GRID_BINS * DIST_STEP
N_ROLE, N_MECH = 4, 11
ROLE_ID = {"background": 0, "joint": 1, "terminal": 2, "defect": 3}
# mechanism channel index -> role channel index
MECH2ROLE = {1: 1, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 2, 9: 2, 10: 2}
MECH_ID = {"joint": 1, "moisture_local": 2, "moisture_distributed": 3,
           "aging": 4, "low_resistance_short": 5, "series_resistance": 6,
           "void": 7}


def sigma_bins(axis_pos_m: float) -> float:
    return max(3.0, 0.02 * axis_pos_m) / DIST_STEP


def mech_channel(mechanism) -> int | None:
    """Mechanism head channel for an event; None events get no mech target."""
    if mechanism is None:
        return None
    if mechanism in MECH_ID:
        return MECH_ID[mechanism]
    if mechanism == "open":
        return 8
    if mechanism == "short":
        return 9
    return 10


def build_targets(events: list[dict], valid_len: int,
                  domain: str = "synth") -> tuple[np.ndarray, ...]:
    """Soft targets + valid mask + positive weight map + per-channel masks."""
    role_t = np.zeros((N_ROLE, GRID_BINS), dtype=np.float32)
    mech_t = np.zeros((N_MECH, GRID_BINS), dtype=np.float32)
    mask = np.zeros(GRID_BINS, dtype=np.float32)
    mask[:valid_len] = 1.0
    w_map = np.ones(GRID_BINS, dtype=np.float32)
    x = np.arange(GRID_BINS, dtype=np.float32)
    for ev in events:
        pos = float(ev["anchor_m"]) / DIST_STEP
        sig = sigma_bins(float(ev["axis_pos_m"]))
        mech_id = mech_channel(ev.get("mechanism"))
        lo = int(max(0, pos - 3 * sig))
        hi = int(min(GRID_BINS - 1, pos + 3 * sig))
        soft = np.exp(-0.5 * ((x[lo:hi + 1] - pos) / max(sig, 1e-3)) ** 2)
        if ev.get("observable", True):
            role_t[ROLE_ID[ev["role"]], lo:hi + 1] = np.maximum(
                role_t[ROLE_ID[ev["role"]], lo:hi + 1], soft)
            if mech_id is not None:
                mech_t[mech_id, lo:hi + 1] = np.maximum(
                    mech_t[mech_id, lo:hi + 1], soft)
        else:
            mask[lo:hi + 1] = 0.0          # invisible event: never supervise
        w_map[lo:hi + 1] = np.maximum(w_map[lo:hi + 1],
                                      float(ev.get("weight", 1.0)))
    # role targets where mech fires but role soft missed (extended defects)
    for m_id, r_id in MECH2ROLE.items():
        role_t[r_id] = np.maximum(role_t[r_id], mech_t[m_id])
    role_cm = np.ones((N_ROLE, GRID_BINS), dtype=np.float32)
    mech_cm = np.ones((N_MECH, GRID_BINS), dtype=np.float32)
    if domain == "field":
        # joint/defect: positive support only; mech: positive support only
        role_cm[1] = (role_t[1] > 0).astype(np.float32)
        role_cm[3] = (role_t[3] > 0).astype(np.float32)
        mech_cm = (mech_t > 0).astype(np.float32)
    return role_t, mech_t, mask, w_map, role_cm, mech_cm


def make_input(impulse: np.ndarray, step: np.ndarray, valid_len: int,
               extra_log: bool = False) -> np.ndarray:
    n = impulse.size
    imp = np.zeros(GRID_BINS, dtype=np.float32)
    stp = np.zeros(GRID_BINS, dtype=np.float32)
    imp[:n] = impulse[:GRID_BINS]
    stp[:n] = step[:GRID_BINS]
    logc = np.sign(imp) * np.log1p(np.abs(imp) * 200.0)
    dist = (np.arange(GRID_BINS, dtype=np.float32) * DIST_STEP) / GRID_M
    chans = [imp, logc, stp, dist]
    if extra_log:
        chans.append(np.sign(imp) * np.log1p(np.abs(imp) * 2000.0))
    x = np.stack(chans)
    x[:, valid_len:] = 0.0
    return x


class DGDataset(Dataset):
    def __init__(self, manifest_rows: list[dict], data_dir: Path,
                 augment: bool = False, domain: str = "synth",
                 extra_log: bool = False):
        self.rows = manifest_rows
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.domain = domain          # "synth" | "field" (by data origin)
        self.extra_log = extra_log
        self.rng = np.random.default_rng(0)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        with np.load(self.data_dir / "phases" / row["file"], allow_pickle=False) as z:
            impulse = z["impulse"].astype(np.float32)
            step = z["step"].astype(np.float32)
            meta = json.loads(str(z["meta"]))
            z0_dev = z["z0_dev"].astype(np.float32) \
                if self.domain == "synth" and "z0_dev" in z else None
        valid_len = int(meta["valid_len"]) if "valid_len" in meta else int(impulse.size)
        x = make_input(impulse, step, valid_len, self.extra_log)
        if self.augment:
            x = self._augment(x, valid_len)
        role_t, mech_t, mask, w_map, role_cm, mech_cm = build_targets(
            meta["events"], valid_len, self.domain)
        term = np.zeros(3, dtype=np.float32)
        visible = 0.0
        for ev in meta["events"]:
            if ev["role"] == "terminal" and ev.get("observable", True):
                term[0] = float(ev["anchor_m"]) / GRID_M
                visible = 1.0
        term[1] = visible
        if z0_dev is not None:
            z0 = np.zeros(GRID_BINS, dtype=np.float32)
            v = min(valid_len, z0_dev.size)
            z0[:v] = z0_dev[:v]
            z0_w = np.zeros(GRID_BINS, dtype=np.float32)
            z0_w[:valid_len] = 1.0
        else:
            z0 = np.full(GRID_BINS, np.nan, dtype=np.float32)
            z0_w = np.zeros(GRID_BINS, dtype=np.float32)
        l_full = meta.get("terminal_full_length_m")
        return dict(x=torch.from_numpy(x), role=torch.from_numpy(role_t),
                    mech=torch.from_numpy(mech_t), mask=torch.from_numpy(mask),
                    term=torch.from_numpy(term), z0=torch.from_numpy(z0),
                    z0_w=torch.from_numpy(z0_w), w_map=torch.from_numpy(w_map),
                    role_cm=torch.from_numpy(role_cm),
                    mech_cm=torch.from_numpy(mech_cm),
                    domain=0 if self.domain == "synth" else 1,
                    l_full=float(l_full) if l_full is not None else float("nan"),
                    sample_id=row["file"], valid_len=valid_len,
                    events=meta["events"])

    def _augment(self, x: np.ndarray, valid_len: int) -> np.ndarray:
        r = self.rng
        noise = (r.standard_normal((2, valid_len)) * 0.004 * float(np.abs(x[0, :valid_len]).max() + 1e-6))
        x[0, :valid_len] += noise[0]
        x[2, :valid_len] += np.cumsum(noise[1]) * 0.01
        gain = 1.0 + r.uniform(-0.15, 0.15)
        x[0, :valid_len] *= gain
        x[1, :valid_len] *= gain
        knots = np.linspace(0, valid_len, 8)
        ripple = np.interp(np.arange(valid_len), knots, r.uniform(0.92, 1.08, 8))
        x[0, :valid_len] *= ripple
        x[1, :valid_len] *= ripple
        return x


def collate(batch: list[dict]) -> dict:
    """Pad/crop to the batch max valid length (rounded to 256) — samples are
    bucketed by length so short cables train at their own resolution."""
    n = min(GRID_BINS, max(256, -(-max(b["valid_len"] for b in batch) // 256) * 256))

    def crop(t: torch.Tensor) -> torch.Tensor:
        return t[:, :n] if t.dim() == 2 else t[:n]

    out = {k: torch.stack([crop(b[k]) for b in batch])
           for k in ("x", "role", "mech", "mask", "role_cm", "mech_cm",
                     "w_map", "z0", "z0_w")}
    out["term"] = torch.stack([b["term"] for b in batch])
    out["domain"] = torch.tensor([b["domain"] for b in batch], dtype=torch.long)
    out["l_full"] = torch.tensor([b["l_full"] for b in batch], dtype=torch.float32)
    out["sample_id"] = [b["sample_id"] for b in batch]
    out["valid_len"] = [b["valid_len"] for b in batch]
    out["events"] = [b["events"] for b in batch]
    return out


class BucketBatchSampler(torch.utils.data.Sampler):
    """Batches of similar valid length; batch order shuffled every epoch."""

    def __init__(self, valid_lens: list[int], batch_size: int, seed: int = 0):
        self.order = np.argsort(np.asarray(valid_lens), kind="stable")
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return (len(self.order) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = [self.order[i:i + self.batch_size]
                   for i in range(0, len(self.order), self.batch_size)]
        for b in rng.permutation(len(batches)):
            yield [int(i) for i in batches[b]]


class MixedBatchSampler(torch.utils.data.Sampler):
    """field_train indices + a fresh synthetic replay subset each epoch,
    bucketed by length.  replay_frac is the synth share of the merged set
    (0.25 -> ~n_field/3 synth samples per epoch)."""

    def __init__(self, field_lens: list[int], synth_lens: list[int],
                 batch_size: int, replay_frac: float = 0.25, seed: int = 0):
        self.field_lens = field_lens
        self.synth_lens = synth_lens
        self.n_field = len(field_lens)
        self.batch_size = batch_size
        self.replay_frac = replay_frac
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        n = self.n_field + self.n_replay()
        return (n + self.batch_size - 1) // self.batch_size

    def n_replay(self) -> int:
        n = int(round(self.n_field * self.replay_frac / max(1.0 - self.replay_frac, 1e-6)))
        return min(n, len(self.synth_lens))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        sel = rng.choice(len(self.synth_lens), size=self.n_replay(), replace=False)
        idx = list(range(self.n_field)) + [self.n_field + int(i) for i in sel]
        lens = list(self.field_lens) + [self.synth_lens[int(i)] for i in sel]
        order = np.argsort(np.asarray(lens), kind="stable")
        batches = [order[i:i + self.batch_size]
                   for i in range(0, len(order), self.batch_size)]
        for b in rng.permutation(len(batches)):
            yield [int(idx[j]) for j in batches[b]]


def load_manifest(data_dir: Path, split: str) -> list[dict]:
    import csv
    rows = []
    with (Path(data_dir) / "manifest.csv").open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                rows.append(row)
    return rows
