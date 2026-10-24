# Codex session memory: Iter8 U-Net result export

## Scope and constraints

- Prepare a reproducible CPU-only exporter and Chinese model card for the existing FieldNN Iter8 U-Net.
- Reuse existing evaluation JSON, field predictions, Iter8 field cache, and checkpoint; do not train or run inference.
- Select 2–4 successful examples from distinct non-wet held-out field groups, excluding RG58. Unknown field locations remain unlabelled; unmatched candidates remain unverified.
- Generated CSV/Markdown/figures belong under `DifferentAgent/CodeX/FieldNN_Iter10_20261023/output/` and are regenerable. The model card and exporter are retained source files.

## Read-only evidence inspected

- Iter8 summary: `DifferentAgent/CodeX/FieldNN_Iter8_20261022/output/iter8/iteration8_summary.json`.
- Iter8 U-Net fixed predictions: `output/iter8/paired/candidate/unet/field_fixed_e000/predictions.jsonl`.
- Iter8 Transformer fixed predictions and Iter5 U-Net baseline predictions are available for the comparison table.
- Iter8 field/calibration cache on E: contains 138 labeled field sweep feature rows, grouped positive-only bags, and 63 short/no-cable calibration references. The full FDR inference prediction file includes 213 curves.
- Validation-selected field event thresholds are U-Net 0.94 and Transformer 0.98. Iter8 U-Net non-wet held-out summary at 0.94 reports 31 curves, 6 groups, 40 known position claims, 93 candidate events, and 17/40 hit@20 (group macro 0.5094). This is not complete field truth.

## Work log

- 2026-10-23: Began static source audit and reported input fields and selection criteria to the parent agent before implementation. No training, GPU inference, or tests have been run.
- Added `DifferentAgent/CodeX/FieldNN_Iter10_20261023/export_existing.py`, `.gitignore`, and the static `MODEL_CARD_Iter8.md`.
- Ran the exporter once with `python -X utf8 -B` on CPU only. It filtered existing predictions by confidence; no model forward or inference was run. Outputs are `output/existing_iter8/performance.csv`, `performance.md`, `MODEL_RESULTS.md`, `selected_cases.csv`, and three 200 dpi PNGs under `output/existing_iter8/assets/`.
- At the validation-selected U-Net event threshold 0.94, the non-wet test population has 6 groups, 40 known-position claims, 17/40 hit@20m, group macro 0.5094, and 93 candidate events over 31 sweeps. Selected distinct non-RG58 groups: F65 4/4 in one B-phase sweep (mean error 4.014 m); Nanling F55 3/3 in one C-phase sweep (6.421 m; group total 6/8); Huzhong F23 2/3 in one A-phase sweep (13.668 m).
- Full-field candidate burden at threshold 0.94 is 565 events over 213 prediction rows (2.653/row); these are not false-positive counts. Calibration references report 0 event false events for train/val/test (39/12/12) after confidence filtering. Complete field accuracy/F1 and inference latency remain unmeasured.
- Case plots show full-range pulse/step plus a local pulse magnification, known-position ±20 m bands, matched and unverified candidate styles, inward ticks, Times New Roman/SimHei, and 200 dpi. No tests, training, GPU, or inference were run.
- Added the Iter3 seven-scale U-Net architecture detail to the static model card after reading source: widths 32/64/128/192/192/192/192, dilation bottleneck 1/2/4/8 plus global-mean projection, linear upsampling/skip connections, and three 1×1 output heads.
- Narrow correction from root: `selected_cases.csv.hit_rate_at_20m` is the selected single-sweep rate; `group_hit_rate_at_20m` remains the group rate. Nanling F55 is now 3/3=1.0 for that sweep and .75 for the full group; the performance CSV group metric stays .75.
- Added reusable `plot_table.py` with `--input`/`--output`, then generated `output/existing_iter8/performance_table.png` at 200 dpi. Existing `plot_curves.py` already plots recorded train/val MIL loss and fixed-.94 hit@20 from all `history_steps.csv` observations without smoothing; it was left unchanged.
- Added `benchmark_models.py` (126 lines) for root-run-only NN latency: repeated `LABEL=PATH` checkpoints, I3/I8 factory selected by 8/12 input channels, fixed cached field curve 0, batch 1, 20 warmups, 100 synchronized wall-time samples, FP16 autocast on CUDA, and raw/median/p95 plus tensor/checkpoint sizes. It has not been run; root will invoke it after the two formal runs.
- `plot_table.py` accepts optional `--benchmark` JSON and overlays rows by U5/U8/T8/I10/I11 labels so the later five-model comparison can fill measured times without changing its CSV source.
