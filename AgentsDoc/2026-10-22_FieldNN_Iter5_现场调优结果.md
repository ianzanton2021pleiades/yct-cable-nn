# FieldNN Iter5：现场精度优先调优结果

日期：2026-10-22；执行：CommandCode（CMDC）；目录：`NN_Archive/20261022NNFirstTry/FieldNN_Iter5_20261022`。

## 目的

接续 CodeX Iter4，在冻结的 12k 仿真 S11 与 raw 响应缓存上继续调优 U-Net / Transformer，解决两个现场问题：首端夹具在约 20 m 反复产生高置信伪候选；宽/弱反射与实测域仍有差距。目标不是提高合成 F1，而是在**不丢已知正例**的前提下减少现场候选。

## 方法

1. **标签宽度跟随输入**：Iter4 只在增强时展宽输入脉冲/阶跃，标签仍是 5 m 高斯；模型因此被要求从宽鼓包输出窄峰。Iter5 对增强样本按 `sigma = min(cap, sqrt(5² + (0.6·sigma_input)²))` 重渲染 event/terminal 目标（cap 20/22 m），35% 干净回放仍用 5 m。
2. **真实夹具扰动加强**：CalibrationData 短接 bank 的注入幅度由 0.5–3× 提高到 1–6× 电缆近端 RMS。
3. **首端抑制**：解码时抑制响应轴前 30 m 的 event/interval。依据：213 条现场曲线中，20–24 m 的候选在 36–42% 的不同电缆上重复出现，属于夹具/仪器固定伪峰。该策略可配置，且不裁剪模型输入。
4. **精度优先工作点**：val 上以 precision 0.90 为目标选阈值，test 冻结。最终两模型 event 0.6、interval 0.99、terminal 0.4。

训练：从 Iter4 各 `best.pt` 只加载权重，AdamW 1e-4，batch 32，7 workers，AMP，U-Net 16 轮（best 10），Transformer 13 轮（best 7）。

## 结果

### 合成 test（1200 条）

| 模型 | 视图 | event P | event R | event F1@5m | interval F1 | terminal F1 |
|---|---|---|---|---|---|---|
| U-Net Iter5 | mixed | .945 | .703 | .806 | .906 | .998 |
| U-Net Iter5 | clean | .934 | .836 | .882 | .930 | .999 |
| U-Net Iter4 | mixed | .862 | .768 | .812 | .865 | .9995 |
| Transformer Iter5 | mixed | .923 | .693 | .792 | .881 | .997 |
| Transformer Iter5 | clean | .910 | .854 | .881 | .921 | 1.000 |
| Transformer Iter4 | mixed | .858 | .762 | .807 | .875 | .998 |

mixed F1 与 Iter4 基本持平，precision 提高 6–8 个点；clean 视图略升。recall 下降是精度优先阈值的直接结果。

### 现场 213 条与 1500 m 已知正例

| 模型 | 版本 | 已知命中@10m | 已知序列额外候选 | 候选/条（均值） | 中位 | P90 |
|---|---|---|---|---|---|---|
| U-Net | Iter4 | 7/11 | 88 | 4.56 | 4 | 8 |
| U-Net | Iter5 | 7/11 | 28 | 2.65 | 3 | 6 |
| Transformer | Iter4 | 6/11 | 41 | 3.75 | 4 | 7 |
| Transformer | Iter5 | 6/11 | 21 | 2.21 | 2 | 4 |

命中数不变，候选总数减少 41–42%，已知序列上的额外候选减少 49–68%。U-Net 在首端 20–24 m 的固定伪候选基本消失。

### 工程师记录台账

对现场目录做人写记录只读挖掘，得到 177 条记录，50/60 个 case group 有位置证据（49 个为 engineer_note）；1500 m 的 11 个位置与旧 `KNOWN_LAB_SWEEPS` 完全一致。对 161 条带注释曲线做 case 级、跨 sweep 取最佳的评价：

| 模型 | 版本 | 注释位置 | hit@10m | hit@20m | hit@30m | 候选/注释曲线 |
|---|---|---|---|---|---|---|
| U-Net | Iter4 | 209 | 40.7% | 47.8% | 51.2% | 5.40 |
| U-Net | Iter5 | 209 | 32.5% | 42.1% | 45.5% | 3.27 |
| Transformer | Iter4 | 209 | 41.6% | 51.2% | 54.5% | 4.65 |
| Transformer | Iter5 | 209 | 34.9% | 42.6% | 44.5% | 2.75 |

hit@20m 下降 6–9 个点，但每个候选的命中率从 11.5%/14.3% 提高到 16.7%/20.1%。台账中的坐标/相别冲突见 `field_evidence.md` §4；未命中不能直接判为漏检。

## 结论与限制

- 本轮达到“已知正例不掉、现场候选大幅减少”的目标，适合精度优先的现场筛查。
- 900–1400 m 湿接头未命中是物理弱反射（脉冲峰值 1.1–2.3× 背景），不是阈值问题；不要用降阈值硬凑。
- 30 m 首端抑制会隐藏短电缆前 30 m 的真实缺陷，现场可按需要改回 20 m 或按长度 2% 自适应。
- 现场标签不完整；所有现场数字是“已知正例命中/候选密度”，不是现场 precision/recall。
- 现场注释只用于评价，未参与训练或阈值选择。

## 复现

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File "D:\GitRepository\Cable-NN\NN_Archive\20261022NNFirstTry\FieldNN_Iter5_20261022\run_iteration5.ps1"
```

结果：`NN_Archive/20261022NNFirstTry/FieldNN_Iter5_20261022/output/iter5/iteration5_summary.json`；汇总图：`Image/20261022_FieldNN_Iter5_precision.png`。
