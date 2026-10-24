# Codex session memory: FieldNN Iter9 implementation

## Scope and constraints

- Continue the Iter9 patch-scoring branch described by `DifferentAgent/CodeX/FieldNN_Iter9_20261022/CONTRACT.md`.
- Implement `proposals.py`, `models.py`, and `prepare.py` in the Iter9 experiment directory.
- The contract forbids agent-side GPU use and post-edit tests. The user will run full cache preparation and later training/evaluation.
- Do not modify Iter3–Iter8 experiment code, the GUI, source FDR files, or the existing user changes elsewhere in the repository.

## Starting state

- The working tree contains substantial pre-existing user/agent changes; the current branch is `main` and is left unchanged.
- The Iter9 directory initially contained only `CONTRACT.md` and a `.gitignore` that covers `output/`, `assets/`, `s11_output/`, `__pycache__/`, `*.pyc`, and `smoke_*/`.
- Read-only inspection confirmed the Iter4 raw cache and Iter8 field/calibration caches exist at the contract paths on E:.

## Work log

- 2026-10-22: Began static implementation from the Iter9 contract. No cache, smoke, GPU, test, or build command has been run.
- Added physical proposal scoring with per-map peak normalization before cross-map NMS, plus center-relative two-view patch extraction.
- Added scalar-logit local U-Net and convolution-tokenized Transformer models for `[B, 20, 256]` patches.
- Added CPU preparation for synthetic, 213-curve field, and calibration patch caches. Synthetic work is parallelized by curve in bounded `workers * 4` batches; field bag indices are remapped to the full manifest and source/report distances use `nominal * source_scale_ratio`.
- Preserved missing field proposals as explicit entries in `positive_bags.json`. No test, cache preparation, smoke run, GPU work, or build was performed, as required by the contract.
