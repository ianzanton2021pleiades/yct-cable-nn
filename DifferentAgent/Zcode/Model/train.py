r"""Train v3: two-stage pipeline (synthetic pretrain -> mixed field fine-tune).

Stage pretrain (default): synthetic train split only, AdamW + warmup + cosine
lr 2e-3, selection = synth_val loss.  field_val F1@20 is printed every epoch
as a transfer monitor (never used for selection, protocol v2 9.5).

    python train.py --stage pretrain --synth-data E:\CableNNDataSet\ZCode\DG_V5_S1 \
        --field-data E:\CableNNDataSet\ZCode\field_train_v2 --epochs 70 --out runs\v3_pretrain

Stage mixed: field_train (all) + a fresh ~25% synthetic replay subset per
epoch (MixedBatchSampler, batches naturally mixed), lr 2e-4, selection =
field_val F1@20; synth_val loss printed as a forgetting monitor.

    python train.py --stage mixed --init runs\v3_pretrain\best.pt \
        --synth-data E:\CableNNDataSet\ZCode\DG_V5_S1 \
        --field-data E:\CableNNDataSet\ZCode\field_train_v2 --epochs 35 --out runs\v3_mixed

Loss: role + mech + 0.3*term + 0.3*z0 (SmoothL1, synthetic valid pixels only;
field z0 carries NaN target / zero weight).  Field positives are scaled by
the event weight map; field joint/defect role channels and the whole mech
head see no negative supervision and stay out of hard-negative mining (v3
channel masks from dataset.build_targets).  Val loss keeps the canonical
unweighted formula (pos_weight=1, hard_k=256, mech weight 1) with the same
masks.  Resume: --resume continues from <out>/last.pt.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import (DGDataset, BucketBatchSampler, MixedBatchSampler,
                     GRID_BINS, collate, load_manifest)
from evaluate import decode_f1
from model_unet import UNet1D, UNet1DV2, count_params


def valid_len_est(row: dict) -> int:
    return int(min(GRID_BINS, (1.3 * float(row["length_m"] or 0) + 10.0) / 0.25 + 1))


def masked_head_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                     hard_k: int = 256, pos_weight: torch.Tensor | None = None,
                     adaptive: bool = False, ch_mask: torch.Tensor | None = None,
                     pos_w_map: torch.Tensor | None = None) -> torch.Tensor:
    """Masked BCE with hard-negative mining.

    ch_mask: (B,C,N) per-channel supervision mask (field samples zero the
    joint/defect role negatives and all mech negatives) — combined with the
    shared valid-length mask, so those channels never enter the top-256
    background mining either (their negatives carry zero loss).
    pos_w_map: (B,N) per-pixel positive-term weight from event weights.
    """
    pw = pos_weight.view(1, -1, 1) if pos_weight is not None else 1.0
    w = pos_w_map.unsqueeze(1) if pos_w_map is not None else 1.0
    logp = F.logsigmoid(logits)
    log1m = F.logsigmoid(-logits)
    per_ch = -(pw * w * target * logp + (1.0 - target) * log1m)  # (B,C,N)
    m = mask.unsqueeze(1) if ch_mask is None else mask.unsqueeze(1) * ch_mask
    per_pixel = (per_ch * m).sum(dim=1)
    pos_support = (target.amax(dim=1) > 0.05).float() * mask
    bg = (1.0 - pos_support) * mask
    n_bg = int(bg.sum().item())
    if adaptive:
        hard_k = int(min(2048, max(64, 0.01 * n_bg)))
    topk = (per_pixel * bg).topk(min(hard_k, n_bg or 1), dim=1).indices
    hard_bg = torch.zeros_like(per_pixel)
    hard_bg.scatter_(1, topk, 1.0)
    keep = torch.max(pos_support, hard_bg)
    return (per_pixel * keep).sum() / keep.sum().clamp(min=1.0)


def z0_loss(pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """SmoothL1 on synthetic valid pixels only (field w == 0 everywhere)."""
    sel = w > 0.5
    if not bool(sel.any()):
        return torch.zeros((), device=pred.device)
    return F.smooth_l1_loss(pred[sel], target[sel])


def terminal_loss(term_logits: torch.Tensor, term: torch.Tensor):
    visible = term[:, 1]
    l_vis = F.binary_cross_entropy_with_logits(term_logits[:, 1], visible)
    if visible.sum() > 0:
        sel = visible > 0.5
        l_pos = F.smooth_l1_loss(term_logits[sel, 0], term[sel, 0])
    else:
        l_pos = torch.zeros((), device=term_logits.device)
    return l_vis + l_pos, l_vis.detach(), l_pos.detach()


def run_epoch(model, loader, opt, device, train: bool, cfg: dict | None):
    """cfg (train): role_pw (C,) tensor, mech_w float, hard_k int (0=adaptive),
    adaptive bool, z0_w float.  Validation always uses the canonical
    unweighted loss (pos_weight=1, hard_k=256, mech weight 1, z0 weight 0.3)
    with the same domain masks."""
    model.train(train)
    sums = dict(loss=0.0, role=0.0, mech=0.0, term=0.0, z0=0.0, n=0)
    f1s, terrs = [], []
    iterator = loader
    if train:
        from tqdm import tqdm
        iterator = tqdm(loader, desc="train", leave=False, mininterval=2.0)
    ctx = torch.enable_grad() if train else torch.no_grad()
    for batch in iterator:
        x = batch["x"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        role_t = batch["role"].to(device, non_blocking=True)
        mech_t = batch["mech"].to(device, non_blocking=True)
        term = batch["term"].to(device, non_blocking=True)
        role_cm = batch["role_cm"].to(device, non_blocking=True)
        mech_cm = batch["mech_cm"].to(device, non_blocking=True)
        w_map = batch["w_map"].to(device, non_blocking=True)
        z0_t = batch["z0"].to(device, non_blocking=True)
        z0_w = batch["z0_w"].to(device, non_blocking=True)
        with ctx, torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(x)
            l_term, l_vis, l_pos = terminal_loss(out["term"].float(), term)
            l_z0 = z0_loss(out["z0"].float(), z0_t, z0_w)
            if train:
                hk = cfg["hard_k"]
                l_role = masked_head_loss(out["role"].float(), role_t, mask, hk,
                                          cfg["role_pw"], cfg["adaptive"],
                                          role_cm, w_map)
                l_mech = masked_head_loss(out["mech"].float(), mech_t, mask, hk,
                                          None, cfg["adaptive"], mech_cm, w_map)
                loss = (l_role + cfg["mech_w"] * l_mech + 0.3 * l_term
                        + cfg["z0_w"] * l_z0)
            else:
                l_role = masked_head_loss(out["role"].float(), role_t, mask, 256,
                                          None, False, role_cm, w_map)
                l_mech = masked_head_loss(out["mech"].float(), mech_t, mask, 256,
                                          None, False, mech_cm, w_map)
                loss = l_role + l_mech + 0.3 * l_term + 0.3 * l_z0
            if not train:
                f1, terr = decode_f1(out["role"].float(), out["term"].float(), batch)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        b = x.size(0)
        sums["loss"] += loss.item() * b
        sums["role"] += l_role.item() * b
        sums["mech"] += l_mech.item() * b
        sums["term"] += l_term.item() * b
        sums["z0"] += l_z0.item() * b
        sums["n"] += b
        if not train:
            f1s.append(f1)
            if terr == terr:
                terrs.append(terr)
    means = {k: v / max(sums["n"], 1) for k, v in sums.items() if k != "n"}
    if not train:
        means["f1_20"] = float(np.mean(f1s)) if f1s else float("nan")
        means["term_mae_m"] = float(np.mean(terrs)) if terrs else float("nan")
    return means


def build_model(args, device):
    cin = 5 if args.extra_log else 4
    if args.arch == "v1":
        return UNet1D(width=args.width, cin=cin).to(device)
    return UNet1DV2(width=args.width, dsconv=args.dsconv, cin=cin).to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("pretrain", "mixed"), default="pretrain")
    ap.add_argument("--synth-data", default=r"E:\CableNNDataSet\ZCode\DG_V5_S1")
    ap.add_argument("--field-data", default=r"E:\CableNNDataSet\ZCode\field_train_v2")
    ap.add_argument("--out", default="runs/v3_pretrain")
    ap.add_argument("--epochs", type=int, default=0,
                    help="0 = stage default (pretrain 70 / mixed 35)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.0,
                    help="0 = stage default (pretrain 2e-3 / mixed 2e-4)")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--arch", choices=("v1", "v2"), default="v1",
                    help="v1 = UNet1D (default); v2 = UNet1DV2 (multi-scale front)")
    ap.add_argument("--dsconv", action="store_true",
                    help="v2 only: depthwise-separable blocks (use with --width 48)")
    ap.add_argument("--extra-log", action="store_true",
                    help="add input ch4 = sign*log1p(2000|x|) (cin=5)")
    ap.add_argument("--pos-weights", default="",
                    help="role-head positive weights 'joint,terminal,defect'; "
                         "empty = auto (v1: 1,1,1  v2: 4,5,6)")
    ap.add_argument("--mech-weight", type=float, default=-1.0,
                    help="mechanism-head loss weight; <0 = auto (v1: 1.0, v2: 0.5)")
    ap.add_argument("--z0-weight", type=float, default=0.3)
    ap.add_argument("--hard-k", type=int, default=256,
                    help="hard-negative budget per batch; 0 = adaptive "
                         "(1%% of batch background pixels, clamp 64..2048)")
    ap.add_argument("--replay-frac", type=float, default=0.25,
                    help="mixed stage: synth share of the merged per-epoch set")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="cap train samples (smoke)")
    ap.add_argument("--val-limit", type=int, default=0,
                    help="cap the synth val split (0 = all); field_val is always full")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="synth_val")
    ap.add_argument("--init", default="", help="pretrained .pt to load before training")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    epochs = args.epochs or (70 if args.stage == "pretrain" else 35)
    lr = args.lr or (2e-3 if args.stage == "pretrain" else 2e-4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    synth_train_rows = load_manifest(args.synth_data, args.train_split)
    synth_val_rows = load_manifest(args.synth_data, args.val_split)
    field_train_rows = load_manifest(args.field_data, "field_train")
    field_val_rows = load_manifest(args.field_data, "field_val")
    if args.stage == "pretrain":
        train_rows = synth_train_rows[:args.limit] if args.limit else synth_train_rows
        val_rows = synth_val_rows[:args.val_limit] if args.val_limit else synth_val_rows
        train_domain = "synth"
    else:
        train_rows = field_train_rows[:args.limit] if args.limit else field_train_rows
        val_rows = field_val_rows
        train_domain = "field"
    if args.val_limit and args.stage == "pretrain":
        synth_val_rows = synth_val_rows[:args.val_limit]
    print(f"stage={args.stage} train={len(train_rows)} "
          f"(field={len(field_train_rows)} synth_pool={len(synth_train_rows)}) "
          f"val={len(val_rows)} field_val={len(field_val_rows)} device={device}")

    if args.stage == "pretrain":
        train_ds = DGDataset(train_rows, args.synth_data, augment=True,
                             domain="synth", extra_log=args.extra_log)
        train_sampler = BucketBatchSampler([valid_len_est(r) for r in train_rows],
                                           args.batch, seed=17)
    else:
        field_ds = DGDataset(train_rows, args.field_data, augment=True,
                             domain="field", extra_log=args.extra_log)
        synth_ds = DGDataset(synth_train_rows, args.synth_data, augment=True,
                             domain="synth", extra_log=args.extra_log)
        train_ds = ConcatDataset([field_ds, synth_ds])
        train_sampler = MixedBatchSampler(
            [valid_len_est(r) for r in train_rows],
            [valid_len_est(r) for r in synth_train_rows],
            args.batch, args.replay_frac, seed=23)
    val_ds = DGDataset(val_rows,
                       args.synth_data if args.stage == "pretrain" else args.field_data,
                       augment=False,
                       domain="synth" if args.stage == "pretrain" else "field",
                       extra_log=args.extra_log)
    field_val_ds = DGDataset(field_val_rows, args.field_data, augment=False,
                             domain="field", extra_log=args.extra_log)
    synth_val_ds = DGDataset(synth_val_rows, args.synth_data, augment=False,
                             domain="synth", extra_log=args.extra_log)
    val_sampler = BucketBatchSampler([valid_len_est(r) for r in val_rows],
                                     args.batch, seed=99)
    val_sampler.shuffle_enabled = False
    fv_sampler = BucketBatchSampler([valid_len_est(r) for r in field_val_rows],
                                    8, seed=98)
    fv_sampler.shuffle_enabled = False
    sv_sampler = BucketBatchSampler([valid_len_est(r) for r in synth_val_rows],
                                    args.batch, seed=97)
    sv_sampler.shuffle_enabled = False
    train_loader = DataLoader(train_ds, num_workers=args.workers,
                              pin_memory=True, persistent_workers=args.workers > 0,
                              batch_sampler=train_sampler, collate_fn=collate)
    val_loader = DataLoader(val_ds, num_workers=2, pin_memory=True,
                            batch_sampler=val_sampler, collate_fn=collate)
    field_val_loader = DataLoader(field_val_ds, num_workers=2, pin_memory=True,
                                  batch_sampler=fv_sampler, collate_fn=collate)
    synth_val_loader = DataLoader(synth_val_ds, num_workers=2, pin_memory=True,
                                  batch_sampler=sv_sampler, collate_fn=collate)

    if args.pos_weights:
        wj, wt, wd = (float(v) for v in args.pos_weights.split(","))
    else:
        wj, wt, wd = (1.0, 1.0, 1.0) if args.arch == "v1" else (4.0, 5.0, 6.0)
    mech_w = args.mech_weight if args.mech_weight >= 0 else (1.0 if args.arch == "v1" else 0.5)
    cfg = dict(role_pw=torch.tensor([1.0, wj, wt, wd], device=device),
               mech_w=mech_w, hard_k=args.hard_k, adaptive=args.hard_k == 0,
               z0_w=args.z0_weight)
    print(f"arch={args.arch} dsconv={args.dsconv} extra_log={args.extra_log} "
          f"pos_w=[1,{wj:g},{wt:g},{wd:g}] mech_w={mech_w:g} z0_w={args.z0_weight:g} "
          f"hard_k={'adaptive' if args.hard_k == 0 else args.hard_k} lr={lr:g} epochs={epochs}")

    model = build_model(args, device)
    print(f"params: {count_params(model):,}")
    if args.init:
        if not Path(args.init).exists():
            raise FileNotFoundError(f"--init not found: {args.init}")
        ck = torch.load(args.init, map_location=device, weights_only=False)
        model.load_state_dict(ck.get("model", ck))
        print(f"initialized from {args.init}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    start_epoch = 0
    # pretrain selects on val loss (lower better); mixed on field F1@20 (higher better)
    best_val = float("inf") if args.stage == "pretrain" else float("-inf")
    history = []
    if args.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"]
        best_val = ck["best_val"]
        history = ck["history"]
        print(f"resumed at epoch {start_epoch}, best_val={best_val:.4f}")

    warmup = max(2, epochs // 30)

    def lr_at(ep):
        if ep < warmup:
            return lr * (ep + 1) / warmup
        t = (ep - warmup) / max(epochs - warmup, 1)
        return lr * 0.5 * (1 + math.cos(math.pi * t))

    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        for g in opt.param_groups:
            g["lr"] = lr_at(epoch)
        t0 = time.time()
        tr = run_epoch(model, train_loader, opt, device, True, cfg)
        va = run_epoch(model, val_loader, opt, device, False, None)
        if args.stage == "pretrain":
            fv = run_epoch(model, field_val_loader, opt, device, False, None)
            sv_loss = va["loss"]                      # val split IS synth_val
        else:
            fv = va                                   # val split IS field_val
            sv = run_epoch(model, synth_val_loader, opt, device, False, None)
            sv_loss = sv["loss"]
        dt = time.time() - t0
        rec = dict(epoch=epoch + 1, lr=round(opt.param_groups[0]["lr"], 8),
                   time_s=round(dt, 1),
                   **{f"train_{k}": round(v, 5) for k, v in tr.items()},
                   **{f"val_{k}": round(v, 5) for k, v in va.items()},
                   synth_val_loss=round(sv_loss, 5),
                   field_f1_20=round(fv["f1_20"], 5),
                   field_val_loss=round(fv["loss"], 5))
        history.append(rec)
        print(f"ep {epoch + 1}/{epochs} {dt:.0f}s lr={rec['lr']:.2e} "
              f"train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} "
              f"val_f1@20={va.get('f1_20', float('nan')):.3f} "
              f"val_term={va.get('term_mae_m', float('nan')):.1f}m "
              f"field_f1@20={rec['field_f1_20']:.3f} "
              f"synth_val_loss={rec['synth_val_loss']:.4f}", flush=True)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        epoch=epoch + 1, best_val=best_val, history=history,
                        args=vars(args)),
                   out / "last.pt")
        metric = va["loss"] if args.stage == "pretrain" else va["f1_20"]
        improved = metric < best_val if args.stage == "pretrain" else metric > best_val
        if improved:
            best_val = metric
            torch.save(dict(model=model.state_dict(), epoch=epoch + 1,
                            val=va, field_f1_20=fv["f1_20"], args=vars(args)),
                       out / "best.pt")
        (out / "history.json").write_text(json.dumps(history, indent=1),
                                          encoding="utf-8")
    print(f"done, best_{ 'val_loss' if args.stage == 'pretrain' else 'field_f1_20' }"
          f"={best_val:.4f} -> {out}")


if __name__ == "__main__":
    main()
