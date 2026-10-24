# Iter9 implementation contract

> 归档说明：Iter9 已停止开发。本文件保留为方法记录；模型训练/评价 runner 将由清理脚本移除。

Purpose: event localization only, using physical candidate proposals and local U-Net / Transformer patch scorers, with user-approved weak positive / unlabeled learning. No GPU or post-edit tests by agents; user runs the full command. No edits to CMDC or previous experiment code/data.

## Inputs and representation

Reuse Iter4 synthetic raw cache and Iter8 field/calibration cache (read-only). Fixed source/report-axis mapping and Iter6 connected group split. Model-axis grid remains 0..3250m/8192 points; valid proposal centers 30..3240 nominal m, same near/right exclusion as old comparison. Physical cable scope remains <=2500m; no true-length crop.

`proposals.py` APIs:

- `propose(features, distance)` -> sorted nominal positions array. Four score maps: abs pulse/local RMS30 (feature4), pulse/local RMS90 from feature0, abs step detrend90 (feature8), abs step detrend300 (feature10). Each finds local maxima >=10m apart, top20 per map, merge candidates within10m, keep strongest center. No label-based candidates on validation/test. No minimum score tuned on test.
- `extract_patch(features, distance, center)` -> float32 `[20,256]`: views +/-50m and +/-200m, resampled256 points; select original channels `[0,1,4,5,6,8,9,10,11]`, append relative coordinate per view. Normalize pulse per view by own RMS, center step by median and own RMS. Clip normalized wave channels to +/-6, divide3. No absolute distance or constant global RMS channels.

## Cache (root default E:/CableNNDataSet/CodeX/FieldNN_Iter9_20261022)

- `config.json`, `prepare_summary.json` (status complete only at end).
- `synthetic_{train,val,test}_patches.npy`: float16 `[N,20,256]`; `synthetic_{split}_labels.npy`: int8 0/1. Each original curve uses physical proposals; keep up to4 positive (<=20m to observed event) and4 negative. Negatives: >40m from events/terminal with event target at center>=0; include up to1 observable terminal-near candidate as NON-DEFECT negative when not overlapping a known event (terminal is distinct from a defect). Others ignored. Metadata split unchanged, deterministic mixed view from Iter5; no true-position injection.
- `field_patches.npy`: float16 flattened candidate patches; `field_candidates.jsonl`: `{index,curve_index,distance_m}` where distance_m is SOURCE/REPORT axis. May also store nominal distance.
- `field_records.jsonl`: all213 source curves, continuous index, split/group/source fields and `source_scale_ratio`; no unknown negative labels.
- `field_bags.json`: dict with `bags`, existing Iter8 bag schema `{indices:[curve_indices], positions_source_m, split,group,case_group,known_lab,...}`, remap indices from Iter8 records to full213.
- `positive_bags.json`: list `{indices:[candidate_indices],split,group,position_source_m}` each known bag-position; only candidates within20 SOURCE m on applicable sweeps. Missing proposal positions recorded, never silently removed from evaluation truth.
- `calibration_patches.npy`: float16 `[N,20,256]`; `calibration_candidates.jsonl`: `{index,curve_index,distance_m}`; `calibration_records.jsonl`: source calibration records with continuous curve index and split/group, including zero-proposal references.

Field all213: reuse cached Iter8 features for138 matching keys; read remaining FDR via frozen reader/preprocess for remainder, group assignment from frozen Iter6 helpers. Synthetic source has12000 curves; float16 patch cache about<1GB. `--smoke` uses first <=8 synthetic curves/split and <=2 field groups/split AFTER full group split; all included original bags remain valid via remapping. Calibration <=2 sources/split. Smoke still has train/val/test and known_lab if present. `--workers` default7.

## Models and training

`models.build_model(name,input_channels=20,width=32,**ignored)` -> `[B]` logits, names unet/transformer only. Small 1D U-Net preserves center features; Transformer conv-tokenizer+relative learned positions+CLS score. No old4M weight loading, no absolute distance feature.

`train.py --data ROOT --output OUT --model unet|transformer --epochs 35 --batch-size 512 --workers 6 --pi .1 --resume --max-batches N`.

Synthetic BCE +0.5 positive-bag MIL BCE +1.0 weak-bag nnPU +0.25 confirmed calibration-negative BCE. Field P bags use max logit per bag; U sampled from ALL TRAIN field candidates including possible positives. Risk = pi*mean softplus(-Pmax) + max(0, mean softplus(U)-pi*mean softplus(Pmax)). This is approximate weak-bag PU: P selection and pi assumptions are disclosed, no unbiased-risk claim. Draw16 P bags, up to256 U and64 calibration patches each step; group-balanced P/U sampling preferred. No field test training. Dropout.1, AdamW lr3e-4 wd.01, AMP init1024, clip1, early-stop6. Log components, grad, AMP skips, seconds, peak memory, validation scores. Save best/last/config/history/RNG resume.

Checkpoint selection: score candidates on FIELD VAL only each epoch; threshold grid .05.. .99, select maximum group macro one-to-one hit@20 under Iter5 U-Net candidate density on same annotated VAL curves. Also measure true calibration false event peaks. Epoch score=val macro-.05*cal false events/reference; tie lower synthetic val BCE. No test use until best selected. Both model names use same U5 density budget for fair comparison.

## Evaluation interface (evaluator owns)

`evaluate.py` exports `validate_model(model,data_root,device,batch_size=512)` returning `{selection_score,threshold,field_macro_hit20,field_candidates_per_curve,calibration_false_events_per_reference,synthetic_bce,...}`; no files needed. Can reuse frozen Iter6 evaluator on generated SOURCE-axis predicted events; nearest candidates within3m NMS after scoring. Counts include missing proposals as misses and use all bag truth.

CLI `evaluate.py --data ROOT --checkpoint PATH --output DIR --device cuda|cpu` scores all field candidates, writes predictions.jsonl (old `{record,status,prediction:{events,intervals:[],terminal:null}}` schema), field_report.json and calibration_report.json. Threshold from checkpoint `validation.threshold`; no test calibration. Emit proposal coverage/upper-bound and all-group/missing/unknown statistics. Synthetic reports are candidate-patch classification, not comparable to old full-axis F1 or terminal.

Runner `run_iteration9.ps1/.py`: prepare -> sequential model train -> evaluator -> fixed prior I5/I8 source-axis event comparisons + summary. No GUI/GPU configuration/process changes. One physical core reserved (reuse old helper), all long tasks user-run. No automatic GPU/tests during authoring. Baseline I8 is frozen and remains available if new branch worse.
