# FieldNN Hybrid：实测 S11 做底的现场数据集与训练结果

日期：2026-10-22；Agent：CommandCode（CMDC）；实验目录：`NN_Archive/20261022NNFirstTry/FieldNN_Hybrid_20261022`。

## 1. 问题

Iter8 的 `history.csv` 显示 `train_field_positive` 从 0.62 降到 0.002，而 `val_field_positive` 一直在 0.72–0.78，模型是在背 21 个现场正例袋；合成域 train/val 仅 0.30 vs 0.40。继续用“纯合成 S11 + 少量现场 MIL”无法跨过域差。

本轮把训练域改成**实测 S11 做底 + 在 S11 域注入已知反射**：真实夹具、噪声、衰减、相位、校准误差都保留；注入位置是精确标签；专家注释只做弱正例；未知位置不当负例。

## 2. 数据

- 现场：`E:\20261010截至的全部fdr文件` 的 213 条 sweep（60 个 case，97 个目录）。
- 频率：重采样到 9 kHz–200 MHz；距离轴 0–3250 m、8192 点、172 m/μs（与 Iter3/Iter5 一致）。
- 注入：`Γ(f)=A·exp(-2π²f²σ_t²)·exp(-j4πfd/v)`，σ_t=2σ_d/v；σ_d=3–35 m；A 按局部背景 RMS 定标，目标峰值/背景 1.2–8.0；避开工程师位置 40 m、事件间距 60 m。区间用两个反号反射，宽度 20–100 m。
- 扰动：平滑复增益/相位纹波、±2 m 时移、复高斯噪声。
- 标签：注入事件权重 1.0；高置信工程师位置权重 0.75（σ=15 m）；其他位置 -1（未知）；实测曲线 terminal 不监督；`CalibrationData` 短接在 d≥30 m 做事件/区间负例（用户确认无被测电缆）。
- 划分：按 `source_case_group` 随机 70/15/15（用户选择；同电缆不同日期可跨 split）；1500 m 湿接头和 5 个整电缆 strict holdout 强制 test。
- 规模：train 1,268 条注入 + 84 条原始现场弱正例；val 108 条注入；test 236 条注入；原始现场特征 213 条。

## 3. 训练

- 8×8192 输入；U-Net / Transformer 与 Iter3/Iter5 同拓扑；bottleneck dropout 0.1；从 Iter5 best 权重初始化（strict=True，dropout 无参数）。
- 混合采样：45% 合成 replay、45% 注入实测、5% 原始现场弱正例、5% 短接负例。
- AdamW 3e-5、weight decay 0.02、AMP 1024、batch 64、workers 0（6 workers 在本机触发 Windows 共享内存错误 1455）。
- EMA 0.999；每 epoch raw/EMA 都评估；选择分数：
  `0.5·syn_val + 0.5·hyb_val + 0.1·cal_val + 0.5·(1-hyb hit@20) + 0.5·(1-field hit@20) + 0.05·(hyb+field candidates/curve)`。
- 早停 patience 6，最多 24 epoch。U-Net/Transformer 的 best 都落在 epoch 1；后续 epoch 的 `val_syn`、`val_hyb`、候选密度持续上升，属于对注入样本的过拟合，被早停挡掉。最终模型不是最后一轮。

## 4. 阈值

- event 阈值在合成 val F1@5m、注入 val hit@20m、现场 val hit@20m 之间加权，并限制注入/现场候选 ≤3/curve；interval/terminal 用合成 precision-first。
- 0.6 精度点：event 0.6、interval 0.95、terminal 0.5（U-Net）/0.3（Transformer）/0.2（集成）。
- 0.5 平衡点：event 0.5、interval 0.95、terminal 同上。

## 5. 结果

### 5.1 精度点（event 0.6）

| 模型 | 合成 clean F1@5m | 合成 mixed F1@5m | 合成 interval F1 | 合成 terminal F1 | 现场 non-wet hit20 / 候选 | 现场 strict hit20 / 候选 | 1500m wet hit20 / 候选 | 短接伪事件 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| U-Net | .852 | .771 | .804 | .971 | .274 / 1.93 | .323 / 2.10 | .727 / 1.27 | 0/15 |
| Transformer | .850 | .750 | .783 | .968 | .342 / 2.05 | .384 / 2.43 | .636 / 1.55 | 0/15 |
| 集成 | .866 | .769 | .818 | .969 | .274 / 1.67 | .313 / 1.95 | .636 / 1.55 | 0/15 |

### 5.2 平衡点（event 0.5）

| 模型 | 合成 clean F1@5m | 合成 mixed F1@5m | 合成 interval F1 | 合成 terminal F1 | 现场 non-wet hit20 / 候选 | 现场 strict hit20 / 候选 | 1500m wet hit20 / 候选 | 短接伪事件 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| U-Net | .843 | .771 | .804 | .971 | **.416 / 3.51** | **.525 / 4.10** | **.818 / 3.09** | 0/15 |
| Transformer | .836 | .757 | .783 | .968 | .390 / 3.35 | .434 / 3.57 | .636 / 2.64 | 0/15 |
| 集成 | **.859** | **.781** | **.818** | .969 | .390 / **2.93** | .455 / 3.33 | .727 / **1.82** | 0/15 |

读法：

- 0.6 是精度优先点：现场候选 1.6–2.4/条，non-wet hit20 .22–.34，wet .64–.73。
- 0.5 是平衡点：U-Net 现场召回最高（non-wet .416、strict .525、wet .818），但候选 3.1–4.1/条；集成模型候选最低（2.93/3.33/1.82），合成 clean/interval 最好。
- 短接参考 0/15 伪事件，说明 30 m 外的短接负例和 30 m 首端抑制有效。
- 合成指标仍然强（clean F1 .84–.86、interval .80–.82、terminal .97），没有为了现场牺牲合成定位能力。

## 6. 边界

- 用户选择 case-level 随机划分，同一电缆不同日期可能跨 split；现场指标可能偏乐观。strict 子集只覆盖 5 个整电缆组 + 1500 m 湿接头，不是全新盲测。
- 工程师位置是弱正例（10–30 m 量级不确定）；未知位置不是负例。现场只报告已知正例命中率和候选密度，不报告 precision/recall。
- 注入是带限高斯反射，不是完整 RLGC 接头模型；它提供真实域背景 + 精确位置，但不是现场缺陷物理的完整复现。
- 1500 m 湿接头只有一个实验组；不能证明跨电缆泛化。
- `CalibrationData` 是夹具短接、无被测电缆；只能做 d≥30 m 的事件/区间负例，不是健康电缆。
- 6 个 DataLoader worker 在本机触发 Windows 共享内存错误 1455；正式训练使用 workers=0。

## 7. 建议

1. 组会展示：优先讲 0.5 平衡点的 U-Net（现场召回最高）和集成（候选最低、合成最好）；0.6 作为精度优先备选。
2. GUI 接入：如果要“少误报”，选集成 0.5；如果要“尽量找到”，选 U-Net 0.5；如果现场要求高精度，选 U-Net/集成 0.6。
3. 下一轮：增加真实弱反射的注入比例、把工程师位置扩展到更多 case、尝试完整的 RLGC 接头模型或 attention 变体；不要把 1500 m 单实验当作跨电缆泛化。

## 8. 长训练与最终综合指标（2026-10-22 晚）

### 8.1 训练

- 20 变体/训练 trace：2,530 条注入训练样本（v10 为 1,268），val 108、test 236；同一 split。
- 从 v10 的 U-Net/Transformer best 继续微调；lr 8e-6、cosine、30 epoch 上限、dropout 0.15、weight decay 0.03、域内 batch（45% 合成 / 45% 注入 / 5% 原始现场 / 5% 短接）。
- U-Net best epoch 2、Transformer best epoch 1；两者都在 epoch 9–10 早停。后续 `val_syn`、`val_hyb`、候选密度上升的 epoch 被丢弃，最终模型不是最后一轮。
- 损失曲线：`Image/20261022_FieldNN_Hybrid_loss_curves.png`（Iter5 合成预训练 train/val + 现场微调 synthetic/hybrid train/val + field hit/候选，标出 selected epoch）。

### 8.2 精度点 event 0.6（综合表）

| Method | Syn P@5m | Syn R@5m | Syn F1@5m | Syn interval F1 | Syn terminal F1 | Macro F1 | Field non-wet Sens@20 | P_ann | Cand/curve | Strict Sens@20 | Wet Sens@20 | Short specificity | Latency ms | Params MB | p vs U-Net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U-Net | .8753 | .8307 | .8524 | .8037 | .9722 | .8761 | .2737 | .6190 | 1.953 | .3232 | .7273 | 1.0000 | 7.450 | 14.83 | — |
| Transformer | .8707 | .8323 | .8511 | .7993 | .9700 | .8735 | .3474 | .7174 | 2.140 | .3939 | .6364 | 1.0000 | 7.432 | 16.40 | .0400 |
| Ensemble | **.9047** | .8319 | **.8668** | **.8252** | .9686 | **.8869** | .3053 | **.7160** | **1.884** | .3636 | .6364 | 1.0000 | 15.491 | 31.23 | **.0328** |

### 8.3 平衡点 event 0.5（综合表）

| Method | Syn P@5m | Syn R@5m | Syn F1@5m | Syn interval F1 | Syn terminal F1 | Macro F1 | Field non-wet Sens@20 | P_ann | Cand/curve | Strict Sens@20 | Wet Sens@20 | Short specificity | Latency ms | Params MB | p vs U-Net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U-Net | .8280 | .8558 | .8416 | .8037 | .9722 | .8725 | **.4263** | .5226 | 3.605 | **.5253** | **.8182** | 1.0000 | 7.530 | 14.83 | — |
| Transformer | .8178 | **.8598** | .8383 | .7993 | .9700 | .8692 | .4053 | .4695 | 3.814 | .4545 | .6364 | 1.0000 | 7.597 | 16.40 | .4158 |
| Ensemble | **.8534** | .8634 | **.8584** | **.8252** | .9686 | **.8841** | .4158 | **.5563** | **3.302** | .4949 | .7273 | 1.0000 | 15.964 | 31.23 | .9006 |

口径：

- `P_ann = hits@20m / all predicted events`（非湿现场）；未标注的真实事件也会被算进分母，所以是**下界**，不是完整 precision。
- `Field non-wet Sens@20` 是已知工程师位置的事件命中率；strict 是 5 条整电缆留出；wet 是 1500 m 湿接头 11 个 sweep。
- `Short specificity` = 1 − 短接参考中出现伪事件/伪区间的比例；15/15 为 1.0。
- `p vs U-Net` = 12 个非湿现场 case 的 per-case hit@20 配对 bootstrap（10,000 次，双侧）；未做多重比较校正。0.6 的 Transformer/Ensemble 约 .04/.03，0.5 不显著。
- 延迟在 CMP 90HX、batch=1、预热后 30 次平均；Ensemble 是两个模型串行，延迟约为单模型两倍。参数量按 fp32 计算。

### 8.4 四个现场案例

- `Image/20261022_FieldNN_Hybrid_case1.png`：1500 m 湿接头试验的 **1100 m 接头浸水** sweep（source 1100 m → nominal 946 m），U-Net 0.5 命中 1/1，3 个候选。
- `Image/20261022_FieldNN_Hybrid_case2.png`：1500 m 湿接头试验的 **900 m 接头浸水** sweep（source 900 m → nominal 774 m），U-Net 0.5 命中 1/1，4 个候选。
- `Image/20261022_FieldNN_Hybrid_case3.png`：**2025-8-20 山东驻马店 B 相**（strict 5 整电缆留出），已知接头 577/1177/1758/1885 m，U-Net 0.5 命中 4/4，6 个候选。
- `Image/20261022_FieldNN_Hybrid_case4.png`：**2026-6-12 消防#3 C 相**（strict 5 整电缆留出），已知接头 452/576/637 m，U-Net 0.5 命中 3/3，5 个候选。

## 8.5 50 → 100 轮续训结果（2026-10-23 凌晨）

### 8.5.1 续训方式

- 50 轮结果先完整备份：`output/pretrain50` 保持不动，另存 `output/pretrain50_saved`。
- 100 轮从 50 轮 `last.pt`（epoch 50）继续：新目录 `output/pretrain100`，复制 50 轮
  `last.pt`/`best.pt`/`history.csv`/`steps.csv` 后 `--resume --restart-scheduler`；
  新 cosine 周期 lr 5e-5 → 1e-6，epoch 51–100，共再 7,500 step。
- 50 轮目录没有被覆盖；100 轮是独立结果。

### 8.5.2 合成 test 对比

| Run | checkpoint epoch | Event P@5m | Event R@5m | Event F1@5m | Interval F1 | Terminal F1 | Train/Val loss | Val F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 epoch | 38 | .7786 | .7543 | .7663 | .6428 | .9731 | .445 / .543 | .7147 |
| 100 epoch | 81 | **.7865** | **.7770** | **.7817** | **.7137** | .9708 | .255 / .584 | **.7321** |

- 100 轮 test event F1 +.015、interval F1 +.071；terminal 略降 .002。
- 但 100 轮 train/val gap 从 .098 扩大到 .329（train .255 / val .584），val loss 后期已上升；
  best 在 epoch 81 左右，后段是过拟合。100 轮不是“无过拟合”的模型，只是 test F1 更高。
- 曲线：`Image/20261022_FieldNN_Hybrid_pretrain50_curves.png`（0–7,500 step）与
  `Image/20261022_FieldNN_Hybrid_pretrain100_curves.png`（0–15,000 step，包含 50→100 的第二段上升）。

### 8.5.3 100 best 的 field 微调

- 从 `output/pretrain100/unet/best.pt` 做 10 epoch field 微调（`config_finetune.json`，
  lr 1e-5、mixup 0.5）：best 只在 epoch 1，field non-wet .297 / 2.97 候选；epoch 4 的
  field hit .531 但候选 6.09，按统一选择分数没有保留。
- 没有超过 long U-Net 0.5（field non-wet .426 / 3.61、P_ann .523、strict .525、wet .818）。
- 结论：100 轮 pretrain 是更好的**纯合成模型**，但不是更好的**现场部署模型**；field 部署仍选
  long U-Net 0.5 / Ensemble 0.6。150 轮未继续：100 轮 gap 已 .329，继续 50 轮只会更过拟合；
  如需要，可从 `output/pretrain100/unet/last.pt` 再开新 cosine 周期续到 150。
- 案例图包含 normalized impulse、step、event probability、已知位置（绿=命中/红=漏检）、预测事件（红三角）、预测区间（浅红）和 terminal 候选（绿虚线）。

## 9. 50 epoch 长训练曲线与最终选型（2026-10-23 凌晨）

### 9.1 合成域 50 epoch 预训练（从头）

- `train_pretrain.py`：9,600 合成训练、1,200 验证；U-Net 从头训练 50 epoch（7,500 step）、batch 64、lr 3e-4、cosine、AMP 1024、EMA 0.999。
- 收敛：epoch 50 train loss .445、val loss .543；train event F1@5m .815、val event F1@5m .712、val terminal F1 .971；best val F1 .715（score .338）。
- 曲线：`Image/20261022_FieldNN_Hybrid_pretrain50_curves.png`，上 panel 是 train/val loss vs step，下 panel 是 train/val F1@5m + terminal F1 vs step；这是本轮最干净的长训练曲线。
- 说明：这个 scratch 预训练模型（F1 .715）弱于部署模型使用的 Iter5 合成 backbone（测试 clean F1 .84–.87）；它的价值是 50 epoch 的收敛曲线，不是最终现场模型。用它的 best 做 10 epoch 现场微调，best 只在 epoch 2（field non-wet .345 / 3.56 候选），没有超过 long U-Net 0.5。

### 9.2 冻结 encoder + mixup 的 field 长训练

- `config_stable.json`：从 long U-Net best 初始化，冻结 encoder，只训 decoder/heads；lr 2e-5、cosine、50 epoch、dropout 0.15、hybrid mixup 0.5。
- 前 6 epoch 稳定：val_syn .42–.46、val_hyb 1.05–1.15、field hit .42–.47、候选 3.7–5.4；之后 val_syn 升到 .52、val_hyb 升到 1.84、候选升到 7.3，best 停在 epoch 6。
- epoch 6 的 0.5 测试：field non-wet .432 / 4.19 候选、P_ann .456、strict .515 / 4.48、wet .818 / 3.73；0.7 精度点 field .194 / 0.89 候选。综合看没有超过 long U-Net 0.5（non-wet .426 / 3.61、P_ann .523、strict .525 / 4.19、wet .818 / 3.18）。
- 曲线：`Image/20261022_FieldNN_Hybrid_stable_curves.png`；标出 selected epoch 6，后续 over-fit 区域只作诊断，不作为最终模型。
- 结论：field 域只有 2,530 条注入样本，冻结 encoder 也只能把过拟合推迟到 epoch 6；继续堆 field-only epoch 不会带来稳定增益。

### 9.3 最终选型

- **综合/精度优先**：long **Ensemble 0.6**（field non-wet .305、P_ann .716、候选 1.88、strict .364、wet .636、Macro F1 .887、p=.033）。
- **现场召回优先**：long **U-Net 0.5**（field non-wet .426、strict .525、wet .818，候选 3.6/4.2/3.2；7.5 ms、14.8 MB）。
- **中间精度选项**：long **Transformer 0.6**（field non-wet .347、P_ann .717、候选 2.14、p=.040）。
- 50 epoch scratch 预训练和 stable field 长训练都保留为**曲线/对照实验**；部署表仍以第 8 节的 long U-Net/Transformer/Ensemble 为准。

### 9.4 四个现场案例（long U-Net 0.5）

- `Image/20261022_FieldNN_Hybrid_case1.png`：1500 m 湿接头 1100 m sweep，1/1、3 候选。
- `Image/20261022_FieldNN_Hybrid_case2.png`：1500 m 湿接头 900 m sweep，1/1、4 候选。
- `Image/20261022_FieldNN_Hybrid_case3.png`：2025-8-20 山东驻马店 B 相，4/4 接头、6 候选（strict 整电缆留出）。
- `Image/20261022_FieldNN_Hybrid_case4.png`：2026-6-12 消防#3 C 相，3/3 接头、5 候选（strict 整电缆留出）。
