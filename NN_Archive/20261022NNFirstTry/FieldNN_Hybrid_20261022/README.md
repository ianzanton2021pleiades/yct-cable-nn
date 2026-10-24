> 已归档，现存权重、结果与复现缺口以 [归档索引](../README.md) 为准；下面保留实验原说明。

# FieldNN Hybrid：实测 S11 做底的现场数据集

日期：2026-10-22；Agent：CommandCode（CMDC）。

## 为什么做这一轮

Iter8 的 `history.csv` 显示：`train_field_positive` 从 0.62 降到 0.002，而
`val_field_positive` 一直在 0.72–0.78，模型是在背 21 个现场正例袋，不是学会现场特征。
合成域 train/val 只是 0.30 vs 0.40，属于轻度。继续用“合成 S11 + 少量现场 MIL”很难
跨过这个域差。

本轮改为**实测 S11 做底，在 S11 域注入已知反射**：真实夹具、噪声、衰减、相位和
校准误差都保留；注入位置是精确标签；专家注释只做弱正例；未知位置一律不当负例。

## 数据生成

每条实测 sweep：

1. 从 `.fdr` 的 SQLite `MeasureData_<rowid>` 读取复数 S11，重采样到 9 kHz–200 MHz。
2. 估计电缆末端（高置信工程师全长 ×0.85；否则 REF 末端检测 ×0.85；再否则 1000 m），
   只在该范围内注入。
3. 注入：
   - 点事件：`Γ(f)=A·exp(-2π²f²σ_t²)·exp(-j4πfd/v)`，σ_t=2σ_d/v；
   - 区间：两个相反符号的反射，间隔 20–100 m；
   - σ_d=3–35 m；A 按局部背景 RMS 定标，目标峰值/背景比 1.3–15；
   - 避开已知工程师位置 40 m，事件间距 ≥60 m。
4. 整条曲线扰动：平滑复增益/相位纹波、±2 m 时移、复高斯噪声。
5. Algorithm 1 → 8 通道距离域特征（与 Iter3/Iter5 完全一致）。
6. 标签：
   - 注入事件：高斯，σ=max(2.5, 0.55σ_d)，权重 1.0；
   - 高置信工程师注释：σ=12 m，权重 0.5；
   - 其他位置：-1（未知）；
   - terminal：实测曲线不监督；
   - CalibrationData 短接：d≥30 m 的事件/区间负例（用户已确认无被测电缆），terminal 不监督。
7. 未知区 top-64 事件概率做弱稀疏惩罚（权重 0.02），抑制“到处报事件”。

划分：按 `source_case_group` 随机 70/15/15（用户选择；同一电缆不同日期可跨 split）。
1500 m 浸水试验和 5 个整电缆 strict holdout 强制 test。`strict` 指标单独报告。

## 训练

- 输入 8×8192；U-Net / Transformer 与 Iter3/Iter5 同拓扑；bottleneck 加 dropout 0.1。
- 从 Iter5 `best.pt` 初始化（strict=True，dropout 无参数）。
- 混合采样：45% 合成 replay、45% 注入实测、5% 原始现场弱正例、5% 短接负例。
- AdamW 3e-5、weight decay 0.02、AMP 1024、batch 64；Windows DataLoader 用 0 worker（6 worker 会触发共享内存错误 1455）。
- EMA 0.999；每个 epoch 同时评估 raw 和 EMA，按选择分数取较好者。
- 选择分数 = `0.5·syn_val + 0.5·hyb_val + 0.1·cal_val + 0.5·(1-hyb hit@20) + 0.5·(1-field hit@20) + 0.05·(hyb+field candidates/curve)`；现场 val 直接进入模型选择。
- 早停 patience 6，最多 24 epoch。

## 阈值与评价

- event 阈值：在合成 val F1@5m 与注入 val hit@20m 之间取平衡，并限制
  `candidates/curve ≤ 3`；interval/terminal 用合成 val 的 precision-first 规则。
- 测试集：
  - 合成 clean/mixed：event F1@2/5/10 m、interval F1、terminal F1；
  - 注入 hybrid test：注入事件 hit@10/20/30 m、candidates/curve、strict 子集；
  - 原始现场 test：工程师位置 hit@10/20/30 m、candidates/curve、strict 子集；
  - 1500 m 湿接头：11 个 sweep 的已知位置 hit、candidates/curve；
  - CalibrationData test：d≥30 m 的伪事件/伪区间计数（应接近 0）。
- 现场未知位置永远不是负例；不报告现场 precision/recall，只报告已知正例命中和候选密度。

## 运行

```powershell
# 完整流程：准备数据 -> 训练 U-Net/Transformer -> 校准 -> 评价 -> 集成
pwsh -NoProfile -ExecutionPolicy Bypass -File "D:\GitRepository\Cable-NN\NN_Archive\20261022NNFirstTry\FieldNN_Hybrid_20261022\run_hybrid.ps1"

# 只跑 U-Net
pwsh -NoProfile -ExecutionPolicy Bypass -File "...\run_hybrid.ps1" -Models unet

# 快速链路检查（2 case/split、2 变体、1 epoch、3 batch）
pwsh -NoProfile -ExecutionPolicy Bypass -File "...\run_hybrid.ps1" -Smoke

# 已经准备好数据，只训练/评价
pwsh -NoProfile -ExecutionPolicy Bypass -File "...\run_hybrid.ps1" -SkipPrepare
```

单步命令：

```powershell
$Python = 'E:\Anaconda\envs\gpushare_cu124\python.exe'
$Exp = 'D:\GitRepository\Cable-NN\NN_Archive\20261022NNFirstTry\FieldNN_Hybrid_20261022'
$Data = 'E:\CableNNDataSet\DeepSeek_CMDC\FieldNN_Hybrid_20261022'

& $Python -X utf8 -B "$Exp\prepare_hybrid.py" --output $Data --workers 7
& $Python -X utf8 -B "$Exp\train.py" --data $Data --model unet --output "$Exp\output\hybrid" --epochs 24 --init-checkpoint "D:\GitRepository\Cable-NN\NN_Archive\20261022NNFirstTry\FieldNN_Iter5_20261022\output\iter5\unet\best.pt"
& $Python -X utf8 -B "$Exp\calibrate.py" --data $Data --checkpoint "$Exp\output\hybrid\unet\best.pt" --output "$Exp\output\hybrid\unet"
& $Python -X utf8 -B "$Exp\evaluate.py" --data $Data --checkpoint "$Exp\output\hybrid\unet\best.pt" --output "$Exp\output\hybrid\unet\evaluation"
```

## 数据与产物

- 数据：`E:\CableNNDataSet\DeepSeek_CMDC\FieldNN_Hybrid_20261022`
  - `hybrid_features_{train,val,test}.npy`、`hybrid_targets_*.npy`、`hybrid_records_*.jsonl`
  - `field_features.npy`、`field_records.jsonl`（未注入的原始现场特征，用于评价）
  - `calibration_raw_imp.npy`、`calibration_raw_step.npy`、`calibration_records.jsonl`
  - `calibration_split.json`、`split.json`、`summary.json`
- 结果：`output/hybrid/{unet,transformer,ensemble}/evaluation/evaluation.json`
- 汇总：`output/hybrid/run_summary.json`
- 图：`python plot_results.py --run output/hybrid` 默认生成 0.5 平衡点图
  `Image/20261022_FieldNN_Hybrid_metrics_e05.png`；加 `--evaluation-name evaluation`
  生成 0.6 精度点图 `Image/20261022_FieldNN_Hybrid_metrics.png`。

## 正式结果（2026-10-22）

数据：213 条现场、0 错误；train 1,268 注入 + 84 原始现场正例、val 108、test 236。
U-Net/Transformer 的 best 都在 epoch 1；后续 `val_syn`、`val_hyb`、候选密度持续上升，
属于对注入样本的过拟合，被早停挡掉。

平衡点 event 0.5（完整两档表见 `AgentsDoc/2026-10-22_FieldNN_Hybrid_实测注入数据集.md`）：

| 模型 | 合成 clean F1@5m | 现场 non-wet hit20 / 候选 | 现场 strict hit20 / 候选 | 1500m wet hit20 / 候选 | 短接伪事件 |
|---|---:|---:|---:|---:|---:|
| U-Net | .843 | **.416 / 3.51** | **.525 / 4.10** | **.818 / 3.09** | 0/15 |
| Transformer | .836 | .390 / 3.35 | .434 / 3.57 | .636 / 2.64 | 0/15 |
| 集成 | **.859** | .390 / **2.93** | .455 / 3.33 | .727 / **1.82** | 0/15 |

精度点 event 0.6：现场 non-wet hit20 .22–.34、候选 1.6–2.4；wet .64–.73；短接 0/15。
U-Net 0.5 现场召回最高；集成 0.5 候选最低、合成最好；U-Net/集成 0.6 是精度优先点。

## 长训练（20 变体，2026-10-22 晚）

- 数据：`E:\CableNNDataSet\DeepSeek_CMDC\FieldNN_Hybrid_Long_20261022`；train 2,530 条注入、val 108、test 236。
- 配置：`config_long.json`（lr 8e-6、cosine、30 epoch 上限、dropout 0.15、weight decay 0.03、域内 batch）。
- U-Net best epoch 2、Transformer best epoch 1；均在 epoch 9–10 早停，后续 over-fit epoch 被丢弃。
- 0.6 精度点：Transformer/Ensemble 的现场 non-wet Sens@20 为 .347/.305，P_ann .717/.716，候选 2.14/1.88，p=.040/.033 vs U-Net；U-Net wet .727。
- 0.5 平衡点：U-Net 现场 non-wet .426、strict .525、wet .818（候选 3.6/4.2/3.2）；Ensemble .416/.495/.727（候选 3.3/3.6/2.1），Macro F1 .884。
- 完整综合表、P_ann 口径、延迟/参数量/p 值见
  `AgentsDoc/2026-10-22_FieldNN_Hybrid_实测注入数据集.md` §8；结果文件在
  `output/hybrid_long/metrics_table_e05.md` 与 `metrics_table_e06.md`。
- 损失曲线：`Image/20261022_FieldNN_Hybrid_loss_curves.png`；
  案例：`Image/20261022_FieldNN_Hybrid_case1.png`（1100 m 湿接头）、
  `Image/20261022_FieldNN_Hybrid_case2.png`（山东驻马店 4 个接头全中）。

长训练一键命令（数据已存在时可加 `-SkipPrepare`；此处直接给单步命令，便于复现）：

```powershell
$Python = 'E:\Anaconda\envs\gpushare_cu124\python.exe'
$Exp = 'D:\GitRepository\Cable-NN\NN_Archive\20261022NNFirstTry\FieldNN_Hybrid_20261022'
$Data = 'E:\CableNNDataSet\DeepSeek_CMDC\FieldNN_Hybrid_Long_20261022'

& $Python -X utf8 -B "$Exp\prepare_hybrid.py" --output $Data --workers 7 --variants-train 20 --variants-val 4 --variants-test 4
& $Python -X utf8 -B "$Exp\train.py" --data $Data --config "$Exp\config_long.json" --model unet --output "$Exp\output\hybrid_long" --epochs 30 --batch-size 64 --workers 0 --init-checkpoint "$Exp\output\hybrid\unet\best.pt"
& $Python -X utf8 -B "$Exp\calibrate.py" --data $Data --checkpoint "$Exp\output\hybrid_long\unet\best.pt" --output "$Exp\output\hybrid_long\unet"
& $Python -X utf8 -B "$Exp\evaluate.py" --data $Data --checkpoint "$Exp\output\hybrid_long\unet\best.pt" --thresholds "$Exp\output\hybrid_long\unet\thresholds.json" --output "$Exp\output\hybrid_long\unet\evaluation"
```

## 50 epoch 合成预训练曲线（2026-10-23 凌晨）

- `train_pretrain.py`：9,600 合成训练、1,200 验证；U-Net 从头 50 epoch（7,500 step）、
  batch 64、lr 3e-4、cosine、EMA 0.999。epoch 50 train/val loss .445/.543，
  train/val event F1@5m .815/.712，val terminal F1 .971；best val F1 .715。
- 曲线：`Image/20261022_FieldNN_Hybrid_pretrain50_curves.png`（step 横轴，train/val loss + F1）。
- 这个 scratch 模型的 F1 低于部署用的 Iter5 合成 backbone（.84–.87）；用它的 best 做
  10 epoch 现场微调，best 只在 epoch 2（field non-wet .345 / 3.56 候选），没有超过
  long U-Net 0.5。因此它保留为**长训练曲线/对照实验**，不是最终现场模型。
- 冻结 encoder + mixup 的 field 长训练：best epoch 6，0.5 测试 field non-wet .432 /
  4.19 候选、P_ann .456、strict .515、wet .818；也没有超过 long U-Net 0.5
  （.426 / 3.61、P_ann .523、strict .525、wet .818）。曲线见
  `Image/20261022_FieldNN_Hybrid_stable_curves.png`。
- 最终部署仍选：**Ensemble 0.6**（精度/综合）或 **U-Net 0.5**（现场召回）；
  中间精度选项 Transformer 0.6。
- 四个现场案例：`Image/20261022_FieldNN_Hybrid_case1.png`（1100 m 湿接头）、
  `case2.png`（900 m 湿接头）、`case3.png`（驻马店 4/4 接头）、
  `case4.png`（消防#3 3/3 接头）。

```powershell
& $Python -X utf8 -B "$Exp\train_pretrain.py" --output "$Exp\output\pretrain50" --epochs 50 --batch-size 64 --lr 3e-4 --workers 0
& $Python -X utf8 -B "$Exp\plot_pretrain_curves.py"
```

### 50 → 100 轮续训

- 50 轮结果保存在 `output/pretrain50`，并完整备份到 `output/pretrain50_saved`；
  100 轮从 50 轮 `last.pt` 续训到新目录 `output/pretrain100`，不改 50 轮目录。
- 续训用新 cosine 周期（lr 5e-5 → 1e-6），epoch 51–100。
- 合成 test：50 轮 event F1@5m .766 / interval .643 / terminal .973；
  100 轮 **.782 / .714 / .971**。100 轮 F1 更高，但 train/val loss 从 .445/.543
  变成 .255/.584，gap 明显扩大，后段已过拟合。
- 100 best 的 field 微调 best 在 epoch 1（field non-wet .297 / 2.97 候选），
  没有超过 long U-Net 0.5；部署仍选 long U-Net 0.5 / Ensemble 0.6。
- 曲线：`Image/20261022_FieldNN_Hybrid_pretrain100_curves.png`（0–15,000 step）。
- 150 轮未继续（100 轮 gap 已 .329）；如需要可从 `output/pretrain100/unet/last.pt`
  再开新 cosine 周期续到 150。

```powershell
& $Python -X utf8 -B "$Exp\train_pretrain.py" --output "$Exp\output\pretrain100" --epochs 100 --batch-size 64 --lr 5e-5 --workers 0 --resume --restart-scheduler
& $Python -X utf8 -B "$Exp\evaluate_pretrain.py" --checkpoint "$Exp\output\pretrain100\unet\best.pt" --output "$Exp\output\pretrain100\unet\evaluation_test.json"
& $Python -X utf8 -B "$Exp\plot_pretrain_curves.py" --run "$Exp\output\pretrain100\unet" --output "D:\GitRepository\Cable-NN\Image\20261022_FieldNN_Hybrid_pretrain100_curves.png"
```

## 边界

- 用户选择 case-level 随机划分，同一电缆不同日期可能跨 split；strict 子集只覆盖 5 个
  整电缆组 + 1500 m 湿接头，不能把它称为全新盲测。
- 工程师位置是弱正例（10–30 m 量级不确定），不是逐点真值；未知位置不作负例。
- 注入事件是“带限高斯反射”，不是完整 RLGC 接头模型；它提供的是真实域背景 + 精确位置，
  不是现场缺陷物理的完整复现。
- CalibrationData 是夹具短接、无被测电缆；只能当 d≥30 m 的事件/区间负例，不是健康电缆。
- 1500 m 湿接头只有一个实验组，不能证明跨电缆泛化。
