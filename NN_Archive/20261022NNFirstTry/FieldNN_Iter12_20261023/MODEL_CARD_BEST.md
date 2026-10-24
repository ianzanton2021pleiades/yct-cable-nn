# FieldNN 最终候选模型卡

## 模型身份与用途

**模型：** Iter11-U-Net（U-Net）  
**现场 event 阈值：** 0.94  
**阈值来源：** [field_thresholds.json](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter11_20261023/output/iter11_50ep/unet/field_thresholds.json>)

该模型用于在当前 FieldNN 现场特征上提出已知正位置附近的阻抗突变候选，不构成缺陷分类、完整现场判定或自动维修结论。

## 输入与结构

输入为 12 通道、8192 点，12→8 输入 stem 接现有 Iter3 主干；三个输出头分别预测 event、interval、terminal。工作网格为 0–3250 m、9 kHz–200 MHz；项目目标为不超过 2.5 km 的短电缆。现场解码在 nominal 距离轴固定抑制首端 30 m 的 event/interval 候选。U-Net 采用七尺度 1D 编码/解码，特征宽度为 32/64/128/192/192/192/192；bottleneck 串联 dilation 1/2/4/8 残差块，并通过全局均值投影回注上下文；解码使用线性上采样和跳连。

12 通道特征输入覆盖 0–3250 m 的 8192 点距离网格；FDR 频率范围为 9 kHz–200 MHz。项目应用目标是 2.5 km 及以下短电缆；3250 m 网格保留边界与传播轴的计算余量。首端固定 30 m nominal 抑制会隐藏该区间内的真实响应，结果不代表近端可检出。

## 数据范围与分组

全场推理共有 213 条记录，其中带标签的 field split 有 65/31/42 条（train/val/test）；val 有 9 组，非湿 test 有 6 组，1500 m 湿接头 test 有 1 组。全场有 75 条曲线没有已知正位置，仍保留在全场推理；当前赢家保存了 213/213 条预测记录。

现场 split 和曲线覆盖沿用冻结的 Iter8 特征缓存。val/test 按电缆组隔离；1500 m 湿接头组单独报告。全场记录持续保留用于推理；没有已知正位置的曲线仍为 unknown，不作为负样本。

## 训练目标与选择

实验：`Iter11-U-Net`；模型配置：input_channels=12, width=32, depth=3, state_size=8；等效 epoch=50；更新步数=5000；batch size=96；主干学习率=0.0；stem 学习率=1e-05；现场弱监督权重=0.05；校准参考权重=0.5；Iter8 初始化=True；仅训练输入 stem=True；验证选择 checkpoint step=1100、field event threshold=0.94。

合成批次使用 Iter8 三头目标：event/terminal Gaussian focal，interval 为全网格加权 BCE 与 Dice。每隔配置指定的训练步加入同定义的正位置跨 sweep bag MIL 与短接/无电缆校准损失；MIL 每个已知位置通过 bag 内 max pooling 产生一个正例，unknown 不作为负例。校准参考只对 30 m 以外的 event/interval mask 施加负约束，不监督 terminal，也不是健康电缆。train/val 使用相同正例 MIL loss；现场曲线另记录固定 0.94 event 阈值下按组平均的 hit@20 m。

三轮实际设置：Iter10-U-Net: λMIL=0.1, λcal=0.5, field/cal 每 2 个训练步；Iter11-U-Net: λMIL=0.05, λcal=0.5, field/cal 每 2 个训练步；Iter12-Transformer: λMIL=0.2, λcal=0.5, field/cal 每 2 个训练步。

bestU：从至少 50 epoch 的完整新轨迹中按现场指标选择；历史权重仅作参照。 bestT：本轮更新权重未超过验证参考，run 内 best.pt 仍选择历史 reference 或初始权重。 当前最终模型为 Iter11-U-Net。比较优胜者按非湿保留组 group macro@20 m （保留 12 位小数）、命中数、候选数、validation 候选数和 validation MIL loss 排序。

Iter11 相对 U5：hit@20 13/40→17/40，group macro 0.3993→0.5094，候选 94→94。这是重复使用保留组上的已知位置覆盖比较，不是完整准确率。

Iter11 相对 U8 保持相同 hit@20 与 group macro，候选数变化 +1；候选仍未核验，因此把它解释为长轨迹下保持覆盖与稳定适应，不宣称准确率提升。

Iter10、Iter11、Iter12 都必须完成至少 50 个等效 epoch 才能成为 bestU/bestT；若 `best.pt` 最终选中 `iter8_reference`，该轮只作为独立探索曲线展示，不作为新模型候选。若 best.pt 选中 step 0 的 T5 初始化 checkpoint，文档会明确标出初始化权重来源和步数，不称为训练后的改进模型。组间选择先比较非湿保留组 group macro hit@20 m（round 到 12 位），再比较命中数、候选负担、验证集候选数和验证 MIL loss。若新轨迹相对 U5 少至少 4 个已知 hit@20 位置，或现场候选数超过 U5 的 1.25 倍，则回退到对应 Iter8 参考并写明原因。回退权重不会被称作 50 epoch 训练结果。

## 三轮训练记录与曲线

以下每轮都保留自己的真实记录点、选中 source 和曲线；不同实验轨迹不拼接。field loss 图记录固定定义的正例 MIL loss，不是合成+MIL+calibration 混合训练总 loss。

每轮训练曲线单独展示；不同实验轨迹不拼接。赢家曲线只对应赢家自身训练过程。
### Iter10-U-Net（独立训练轨迹）

等效 epoch=50.0；成功 optimizer updates=7500；训练 attempt=7503，AMP skip=3；真实 history 点=76；最佳 source=`iter10_dense`，step=400；总参数=3888195，可训练参数=3888195；batch=64，λMIL=0.1，stem LR=0.0001，init_iter8=False，stem_only=False。
训练前验证基线：train/val MIL loss=1.8229/1.4700，val group macro@20=0.4111，阈值=0.6800，候选=62。
初始化 checkpoint：[初始化 checkpoint](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter5_20261022/output/iter5/unet/best.pt>)。配置与产物：[config.json](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter10_20261023/output/iter10_50ep/unet/config.json>)、[run_info.json](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter10_20261023/output/iter10_50ep/unet/run_info.json>)、[initial_validation.json](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter10_20261023/output/iter10_50ep/unet/initial_validation.json>)、[best.pt](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter10_20261023/output/iter10_50ep/unet/best.pt>)、[history_steps.csv](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter10_20261023/output/iter10_50ep/unet/history_steps.csv>)、[optimizer_steps.csv](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter10_20261023/output/iter10_50ep/unet/optimizer_steps.csv>)。

![Iter10-U-Net independent training trajectory: field loss](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter10_20261023/output/iter10_50ep/unet/figures/field_loss.png>)

![Iter10-U-Net independent training trajectory: fixed-.94 group hit@20](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter10_20261023/output/iter10_50ep/unet/figures/field_hit20.png>)

![Iter10-U-Net independent training trajectory: validation operating point](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter10_20261023/output/iter10_50ep/unet/figures/validation_operating_point.png>)

### Iter11-U-Net（本轮赢家训练轨迹）

等效 epoch=50.0；成功 optimizer updates=5000；训练 attempt=5002，AMP skip=2；真实 history 点=51；最佳 source=`iter10_dense`，step=1100；总参数=3888195，可训练参数=96；batch=96，λMIL=0.05，stem LR=1e-05，init_iter8=True，stem_only=True。
训练前验证基线：train/val MIL loss=0.1999/0.6495，val group macro@20=0.7167，阈值=0.9400，候选=74。
初始化 checkpoint：[初始化 checkpoint](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter8_20261022/output/iter8/unet/best.pt>)。配置与产物：[config.json](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter11_20261023/output/iter11_50ep/unet/config.json>)、[run_info.json](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter11_20261023/output/iter11_50ep/unet/run_info.json>)、[initial_validation.json](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter11_20261023/output/iter11_50ep/unet/initial_validation.json>)、[best.pt](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter11_20261023/output/iter11_50ep/unet/best.pt>)、[history_steps.csv](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter11_20261023/output/iter11_50ep/unet/history_steps.csv>)、[optimizer_steps.csv](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter11_20261023/output/iter11_50ep/unet/optimizer_steps.csv>)。

![Iter11-U-Net training trajectory: field loss](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter11_20261023/output/iter11_50ep/unet/figures/field_loss.png>)

![Iter11-U-Net training trajectory: fixed-.94 group hit@20](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter11_20261023/output/iter11_50ep/unet/figures/field_hit20.png>)

![Iter11-U-Net training trajectory: validation operating point](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter11_20261023/output/iter11_50ep/unet/figures/validation_operating_point.png>)

### Iter12-Transformer（独立训练轨迹；best.pt 为历史 reference）

等效 epoch=50.0；成功 optimizer updates=5000；训练 attempt=5001，AMP skip=1；真实 history 点=51；最佳 source=`iter8_reference`，step=no update step (validation reference)；总参数=4298115，可训练参数=96；batch=96，λMIL=0.2，stem LR=3e-05，init_iter8=False，stem_only=True。
训练前验证基线：train/val MIL loss=1.2664/1.1718，val group macro@20=0.5222，阈值=0.7300，候选=50。
初始化 checkpoint：[初始化 checkpoint](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter5_20261022/output/iter5/transformer/best.pt>)。配置与产物：[config.json](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter12_20261023/output/iter12_50ep/transformer/config.json>)、[run_info.json](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter12_20261023/output/iter12_50ep/transformer/run_info.json>)、[initial_validation.json](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter12_20261023/output/iter12_50ep/transformer/initial_validation.json>)、[best.pt (historical reference)](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter12_20261023/output/iter12_50ep/transformer/best.pt>)、[history_steps.csv](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter12_20261023/output/iter12_50ep/transformer/history_steps.csv>)、[optimizer_steps.csv](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter12_20261023/output/iter12_50ep/transformer/optimizer_steps.csv>)。

![Iter12-Transformer exploration curve (best.pt is reference): field loss](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter12_20261023/output/iter12_50ep/transformer/figures/field_loss.png>)

![Iter12-Transformer exploration curve (best.pt is reference): fixed-.94 group hit@20](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter12_20261023/output/iter12_50ep/transformer/figures/field_hit20.png>)

![Iter12-Transformer exploration curve (best.pt is reference): validation operating point](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter12_20261023/output/iter12_50ep/transformer/figures/validation_operating_point.png>)

### Iter12 T8-stem pilot（独立试跑；不纳入正式 50 epoch 结果与候选排名）

该 pilot 冻结 Iter8 Transformer 主干，仅训练输入 stem；记录 10.0 个等效 epoch、1000 updates、11 个真实 history 点。初始来源：[Iter8 Transformer checkpoint](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter8_20261022/output/iter8/transformer/best.pt>)。当时 train/val field MIL loss=0.002079/0.737823；正例梯度趋于饱和，validation 未改善。该 pilot 未评分 test。
正式 I12 是从 Iter5 Transformer 主干重新启动的独立完整 50 epoch；config 中 `init_iter8=false`，T8 仅保留为 validation reference。pilot 曲线不与正式轨迹拼接。
保留文件：[config.json](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter12_20261023/output/pilot_from_t8_stem/transformer/config.json>)、[run_info.json](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter12_20261023/output/pilot_from_t8_stem/transformer/run_info.json>)、[selection.json](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter12_20261023/output/pilot_from_t8_stem/transformer/selection.json>)、[history_steps.csv](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter12_20261023/output/pilot_from_t8_stem/transformer/history_steps.csv>)、[optimizer_steps.csv](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter12_20261023/output/pilot_from_t8_stem/transformer/optimizer_steps.csv>)。


### 当前训练结果的解释

I11 从已有 I8 权重开始，仅更新 96 个输入层参数。其 train MIL loss 首末为 0.1999→0.1994，val 为 0.6495→0.6548；训练稳定，但约 0.45 的 train/val 差距仍然存在。固定 0.60 的高命中曲线不代表该差距已解决，也不表示从零训练获得了收敛提升。50 轮是完整实验时长，最终按验证规则选择 step 1100 的权重；推理包原始 epoch=10 使用从 0 开始的编号，对应第 11 轮。

## 现场评估结果

**最终候选非湿 test：** 6 组、40 个已知位置；hit@10/20/30 m = 10/40/17/40/19/40；group macro@20 m = 0.5094；候选 94 个 / 31 条 sweep；matched mean error@20 m = 7.99 m

**1500 m 湿接头 test（单独报告）：** 1 组、11 个已知位置；hit@10/20/30 m = 7/11/7/11/7/11；group macro@20 m = 0.6364；候选 13 个 / 11 条 sweep；matched mean error@20 m = 4.16 m

| 比较模型 | event 阈值 | 非湿 hit@10/20/30 m | group macro@20 m | 候选 / sweep | matched mean error@20 m | 湿组 hit@20 m | 湿组候选 |
|---|---:|---:|---:|---:|---:|---:|---:|
| [Iter5-U-Net](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter5_20261022/output/iter5/unet/best.pt>) | 0.6 | 9/40/13/40/14/40 | 0.3993 | 94/31 | 7.42 | 7/11 | 35 |
| [Iter11-U-Net (bestU)](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter11_20261023/output/iter11_50ep/unet/best.pt>) | 0.94 | 10/40/17/40/19/40 | 0.5094 | 94/31 | 7.99 | 7/11 | 13 |
| [Iter8-Transformer (bestT, historical fallback)](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter8_20261022/output/iter8/transformer/best.pt>) | 0.98 | 8/40/10/40/10/40 | 0.3368 | 49/31 | 7.45 | 3/11 | 13 |

### I11 固定 0.60 覆盖工作点（同一模型的补充 operating point）

阈值 `0.60` 沿用 pre-existing Iter5 U-Net locked preset; not selected or adjusted from test results，没有依据 test 调整；它是同一 I11 checkpoint 的第二工作点，不是第 4 个模型；不替换主 .94 阈值，也不改变 bestU/bestT 权重选择。

| I11 工作点 | 评估范围 | 非湿 hit@20 m | group macro@20 | 候选 / sweep | 湿组 hit@20 m | 湿组候选 / sweep |
|---|---|---:|---:|---:|---:|---:|
| .94 主点 | 全场 213 sweep CUDA FP16 推理；已标注 test 组评分 | 17/40 | 0.5094 | 94/31=3.032 | 7/11 | 13/11 |
| .60 固定覆盖点 | CPU FP32 batch 1；138 条已标注 cache 曲线 | 22/40 | 0.6667 | 203/31=6.548 | 8/11 | 57/11 |

固定 .60 曲线包含 51 个真实点（update 0–5000）；固定 .60 group hit@20 的 train/val 末点为 0.86938/0.85556。I11 的验证选中 step=1100；末点只是训练轨迹记录，不是 checkpoint 选择点。这是已知位置覆盖指标，不是完整 Accuracy。
I11 .60 CPU cache 有 203 个非湿候选；U8 既有 .60 结果为 205 个候选，但 U8 数来自全 213 sweep 推理，I11 .60 只覆盖 138 条标注缓存曲线。两者范围不同，不能把表面上的 2 个差值称作改进；所有未匹配候选仍未核验。
表格中 .60 行的参数量与 NN-only latency（若已测）沿用同一个 I11 checkpoint benchmark；CPU FP32 batch-1 是现场覆盖预测的计算路径，不是 GPU latency 的计算路径。
主 `performance.csv`/`performance_table.png` 中已添加同一 I11 checkpoint 的 .60 次工作点行。

审计资料：[coverage_workpoint_summary.json](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter11_20261023/output/coverage_workpoint/coverage_workpoint_summary.json>)、[coverage workpoint performance.csv](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter11_20261023/output/coverage_workpoint/cpu_cache_export_060/performance.csv>)、[coverage workpoint performance.md](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter11_20261023/output/coverage_workpoint/cpu_cache_export_060/performance.md>)、[fixed_060_hit_curve.png](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter11_20261023/output/coverage_workpoint/fixed_060_hit_curve.png>)、[selected_cases.csv](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter11_20261023/output/coverage_workpoint/cpu_cache_export_060/selected_cases.csv>)。

同一 I11 推理包的可选 .60 阈值配置：[thresholds_coverage_060.json](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter12_20261023/output/final_delivery/best_model/thresholds_coverage_060.json>)。默认 `thresholds.json`/`field_thresholds.json` 保持 .94，不复制第二份权重。

![I11 fixed .60 case 1](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter11_20261023/output/coverage_workpoint/cpu_cache_export_060/assets/case_unconfirmed_case_2026_5_30_12_13_22深圳110kV坂田变电站10kV四季线F65坂田南学校_1公用智能柜.png>)
![I11 fixed .60 case 2](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter11_20261023/output/coverage_workpoint/cpu_cache_export_060/assets/case_unconfirmed_case_2026_7_19_17_00_31深圳110kV樟树布变电站10kV湖中线F23龙山工业区_5三遥柜.png>)
![I11 fixed .60 case 3](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter11_20261023/output/coverage_workpoint/cpu_cache_export_060/assets/case_linked_cable_nanling_f55_june_2026.png>)

完整历史 U5/U8/T8/I10/I11/I12 指标见 [comparison.md](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter12_20261023/output/final_delivery/comparison.md>)、[performance.csv](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter12_20261023/output/final_delivery/performance.csv>)、[performance_table.png](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter12_20261023/output/final_delivery/performance_table.png>)、[historical_results.csv](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter12_20261023/output/final_delivery/historical_results.csv>)。

| 候选 | checkpoint | event 阈值 | 阈值来源 | tie-break 验证资料 |
|---|---|---:|---|---|
| Iter5-U-Net | [checkpoint](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter5_20261022/output/iter5/unet/best.pt>) | 0.6 | Iter10 comparison baseline: field event threshold 0.60 | [saved predictions](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter5_20261022/output/iter5/unet/field_e06/predictions.jsonl>) |
| Iter8-U-Net | [checkpoint](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter8_20261022/output/iter8/unet/best.pt>) | 0.94 | Iter10 comparison baseline: field validation event threshold 0.94 | [reference_validation.json](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter11_20261023/output/iter11_50ep/unet/reference_validation.json>) |
| Iter8-Transformer | [checkpoint](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter8_20261022/output/iter8/transformer/best.pt>) | 0.98 | Iter10 comparison baseline: field validation event threshold 0.98 | [reference_validation.json](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter12_20261023/output/iter12_50ep/transformer/reference_validation.json>) |
| Iter10-U-Net | [checkpoint](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter10_20261023/output/iter10_50ep/unet/best.pt>) | 0.83 | [selection.json](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter10_20261023/output/iter10_50ep/unet/selection.json>) | [selection.json](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter10_20261023/output/iter10_50ep/unet/selection.json>) |
| Iter11-U-Net | [checkpoint](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter11_20261023/output/iter11_50ep/unet/best.pt>) | 0.94 | [field_thresholds.json](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter11_20261023/output/iter11_50ep/unet/field_thresholds.json>) | [selection.json](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter11_20261023/output/iter11_50ep/unet/selection.json>) |
| Iter12-Transformer (best.pt is Iter8 reference) | [checkpoint](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter12_20261023/output/iter12_50ep/transformer/best.pt>) | 0.98 | [field_thresholds.json](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter12_20261023/output/iter12_50ep/transformer/field_thresholds.json>) | [selection.json](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter12_20261023/output/iter12_50ep/transformer/selection.json>) |

所有现场位置指标复用 Iter6 `evaluate_field.py` 正例匹配规则。未命中位置不代表不存在其他缺陷；未匹配候选仍未核验，因此不计算完整现场 Accuracy、precision、recall 或 F1。保留组在多轮开发中已反复使用，结果不是新盲测。Iter11 与 Iter12 同时改变了架构、stem 学习率和现场弱监督权重，不能据此单独归因架构差异。

## NN-only 参数与推理计时

参数量 3,888,195；batch=1，input shape=[1, 12, 8192]，autocast=float16；预热 20 次、同步测量 100 次；NN forward median/p95=10.083/11.189 ms。范围：NN forward only; input preloaded and resident on device before timing; excludes H2D, FDR read, FFT/Algorithm 1, preprocessing and postprocessing.

只有实际 benchmark JSON 存在时才填入 warmup、重复次数与时延；该测量仅覆盖输入已预载入后的神经网络 forward，不包括 FDR 读取、FFT、特征计算、数据搬运和后处理。

## 现场案例

案例图只展示已知位置与候选的空间对应关系，不构成额外的样本级性能指标。

![现场案例 1](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter12_20261023/output/final_delivery/best_exporter/assets/case_unconfirmed_case_2026_5_30_12_13_22深圳110kV坂田变电站10kV四季线F65坂田南学校_1公用智能柜.png>)

![现场案例 2](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter12_20261023/output/final_delivery/best_exporter/assets/case_linked_cable_nanling_f55_june_2026.png>)

![现场案例 3](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter12_20261023/output/final_delivery/best_exporter/assets/case_unconfirmed_case_2026_7_19_17_00_31深圳110kV樟树布变电站10kV湖中线F23龙山工业区_5三遥柜.png>)

## 交付文件

- 精简推理包：[inference.pt](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter12_20261023/output/final_delivery/best_model/inference.pt>)（15603962 bytes），epoch=10；由 [D:\GitRepository\Cable-NN\NN_Archive\20261022NNFirstTry\FieldNN_Iter11_20261023\output\iter11_50ep\unet\inference.pt](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter11_20261023/output/iter11_50ep/unet/inference.pt>) 在 CPU 上读取，仅保留 `model_name`、`config`、`model`、`epoch`、`selection`。
- 推理阈值配置：[field_thresholds.json](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter12_20261023/output/final_delivery/best_model/field_thresholds.json>)、[thresholds.json](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter12_20261023/output/final_delivery/best_model/thresholds.json>)；两个文件都包含 event、interval=0.99、terminal=0.4。
- 最终索引：[model_index.json](</D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter12_20261023/output/final_delivery/model_index.json>)
- 其余比较与案例文件均使用绝对路径链接；案例来源与曲线分别列于上文。

## 适用边界

现场标注只覆盖已知正位置；unknown 位置保持 unknown，未匹配候选需要人工核验。该模型尚不构成 GUI 集成或完整现场自动诊断能力。
