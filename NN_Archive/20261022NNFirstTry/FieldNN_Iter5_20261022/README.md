> 已归档，现存权重、结果与复现缺口以 [归档索引](../README.md) 为准；下面保留实验原说明。

# Iter5：宽反射标签对齐 + 首端抑制 + 精度优先工作点

本目录是 CMDC 对 CodeX Iter4 的接续实验。目标是在不重造 S11 的前提下，提高 ≤2.5 km 现场 FDR 的可用性，重点是**减少首端固定伪候选**、让宽/弱反射更容易被检出，并把默认工作点调到精度优先。

## 输入与边界

- 数据：复用 `E:\CableNNDataSet\CodeX\FieldNN_Iter4_20261022` 的 raw impulse/step/targets 缓存（只读，不修改 Iter3/Iter4 数据）。
- 初值：`FieldNN_Iter4_20261022/output/adaptation4/{unet,transformer}/best.pt`，只加载权重。
- 现场数据：`E:\20261010截至的全部fdr文件` 的 213 条曲线仅用于评价，不参与训练。
- 现场已知标签：1500 m 浸水试验 11 个接头位置（100–1400 m），命中容差 10/20/30 m。
- 资源：`gpushare_cu124`，CMP 90HX 10 GiB，7 个训练 worker，保留 1 个物理核心。

## 相对 Iter4 的改动

1. **标签宽度跟随输入展宽**：增强样本的 event/terminal 目标由
   `sigma = min(cap, sqrt(5² + (0.6·sigma_input)²))` 重新渲染，event cap 20 m、terminal cap 22 m。
   Iter4 只展宽输入、标签始终 5 m，模型被要求从宽鼓包中输出窄峰；本版让标签与输入宽度一致，宽事件不会被当成负样本压制。
   干净回放（35%）仍使用原始 5 m 标签。
2. **夹具扰动增强**：真实 CalibrationData bank 的注入尺度从 `0.5–3×` 提高到 `1–6×` 电缆近端 RMS，更接近现场首端振荡占主导的情况。
3. **首端抑制**：解码时抑制响应轴前 **30 m** 的 event/interval 候选（`decode.py`）。依据是 213 条现场曲线中 20–24 m 候选重复出现在 36–42% 的不同电缆上，属于夹具/仪器固定伪峰。该值可由 `cfg["evaluation"]["near_exclusion_m"]` 覆盖；模型输入不裁剪。
4. **峰值显著性**：`event_prominence` 从 0.05 提高到 0.10。
5. **精度优先阈值**：val 校准目标 precision 0.90（Iter4 为 0.85），test 阈值冻结后再做现场评价。

## 运行

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File "D:\GitRepository\Cable-NN\NN_Archive\20261022NNFirstTry\FieldNN_Iter5_20261022\run_iteration5.ps1"
```

默认 20 epoch、batch 32、7 workers、两模型顺序训练。中断加 `-Resume`；只跑一个模型用 `-Models unet`。

结果入口：`output/iter5/iteration5_summary.json`；逐阶段日志 `output/iter5/logs/`；每模型 `history.csv`、`best.pt`、`thresholds.json`、`evaluation_test_{mixed,clean}/`、`field_e###/`。

## 评价口径

- 合成 test 1200 条：mixed（半增强）与 clean 两个视图，阈值来自 val，不在 test 上调参。
- 现场 213 条：known-positive 命中率、候选/曲线、按 cable_group 的中位数；无完整真值，不报告现场 precision/recall。
- 与 Iter4 对比时，必须区分“同权重同增强集基线”和“新训练结果”；不同标签/解码口径的数值不能直接相减。

## 正式结果（2026-10-22）

训练：U-Net 从 Iter4 best 微调 16 轮（best epoch 10），Transformer 微调 13 轮（best epoch 7）；batch 32、7 workers、AMP。阈值：val precision target 0.90；U-Net 锁定 0.6，Transformer 校准得 0.6。两模型 interval 0.99、terminal 0.4。

### 合成 test（1200 条，@5 m）

| 模型 | 视图 | P | R | F1 | interval F1 | terminal F1 | 事件/条 |
|---|---|---|---|---|---|---|---|
| U-Net | mixed | .945 | .703 | .806 | .906 | .998 | 1.53 |
| U-Net | clean | .934 | .836 | .882 | .930 | .999 | 1.85 |
| U-Net Iter4 (mixed) | mixed | .862 | .768 | .812 | .865 | .9995 | 2.3 |
| Transformer | mixed | .923 | .693 | .792 | .881 | .997 | 1.55 |
| Transformer | clean | .910 | .854 | .881 | .921 | 1.000 | 1.94 |
| Transformer Iter4 (mixed) | mixed | .858 | .762 | .807 | .875 | .998 | 2.2 |

合成 mixed 的 F1 与 Iter4 基本持平（-0.6 / -1.5 个点），但 precision 提高 6-8 个点；clean 视图反而略升。这是精度优先阈值的直接结果。

### 现场 213 条

| 模型 | 版本 | 1500 m 已知命中@10 m | 已知序列额外候选 | 候选/条（均值） | 中位 | P90 |
|---|---|---|---|---|---|---|
| U-Net | Iter4 | 7/11 | 88 | 4.56 | 4 | 8 |
| U-Net | Iter5 | **7/11** | **28** | **2.65** | 3 | 6 |
| Transformer | Iter4 | 6/11 | 41 | 3.75 | 4 | 7 |
| Transformer | Iter5 | **6/11** | **21** | **2.21** | 2 | 4 |

在命中数完全不变的前提下，U-Net 现场候选减少 42%，已知序列上的额外候选减少 68%；Transformer 分别减少 41% 和 49%。首端 20–24 m 的固定伪候选在 U-Net 上从 36–42% 的曲线出现降到接近 0（见 `field_e06` 的图）。

### 工程师记录覆盖（新台账）

`field_evidence.md/json` 由只读挖掘生成：177 条记录、50/60 个 case group 有位置证据（其中 49 个为 engineer_note），1500 m 的 11 个位置与 `KNOWN_LAB_SWEEPS` 完全一致。

对 161 条带位置注释的曲线做“case 级、跨 sweep 取最佳”的评价（注释是人读 FDR 的位置，不是独立真值；坐标冲突已逐条标注）：

| 模型 | 版本 | 注释位置数 | hit@10 m | hit@20 m | hit@30 m | 候选/注释曲线 |
|---|---|---|---|---|---|---|
| U-Net | Iter4 | 209 | 40.7% | 47.8% | 51.2% | 5.40 |
| U-Net | Iter5 | 209 | 32.5% | 42.1% | 45.5% | 3.27 |
| Transformer | Iter4 | 209 | 41.6% | 51.2% | 54.5% | 4.65 |
| Transformer | Iter5 | 209 | 34.9% | 42.6% | 44.5% | 2.75 |

hit@20 m 下降 6-9 个点，但每条注释曲线的候选减少 26-41%；**每个候选的命中率**从 11.5%/14.3% 提高到 16.7%/20.1%。少数 case 存在坐标轴/相别错位（台账 §4 已列出），不能把未命中直接当模型漏检。

## 已知边界

- 1500 m 的 900–1400 m 湿接头：脉冲峰值只有背景的 1.1–2.3 倍，1100 m 以上阶跃跳变也落到背景 95 分位附近，属于物理弱反射；本轮没有用阈值硬凑命中。
- 30 m 首端抑制是解码策略，不是模型输入裁剪；它对短电缆（<300 m）会隐藏前 30 m 的真实缺陷，使用时可改 `cfg["evaluation"]["near_exclusion_m"]`。
- 合成 mixed 的 recall 下降是精度优先的直接代价；若要召回优先，把 U-Net event 阈值调回 0.5（test mixed R=0.726、候选 3.99/条）。
- 现场标签不完整，本文所有现场数字都是“已知正例命中/候选密度”，不是现场 precision/recall。

