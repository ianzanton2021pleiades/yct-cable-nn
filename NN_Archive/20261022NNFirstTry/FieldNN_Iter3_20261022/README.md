> 已归档，现存权重、结果与复现缺口以 [归档索引](../README.md) 为准；下面保留实验原说明。

# FieldNN Iter3 运行说明

本目录包含新一轮数据生成、U-Net/Transformer 训练、验证阈值校准和现场曲线覆盖检查脚本。正式生成和训练时间较长，由用户在 Windows PowerShell 7 中启动；本 README 说明运行方式和结果边界，不是阶段性能报告。

## 相比旧四通道方案

旧版的四通道输入没有显式距离坐标和局部背景尺度，U-Net 分支只在几个下采样尺度各自做局部卷积，较难同时表达首端干扰、长距离上下文和远端位置。旧训练损失对稀疏事件使用较高正类权重并叠加 Dice，且用合成末端真值把末端之后的所有输出头都从损失中屏蔽，容易让后端伪峰缺少负例约束。

Iter3 使用八通道特征，包含归一化距离坐标和局部脉冲背景信息；U-Net/Transformer 使用逐层编码器、跳连解码器及更长上下文；event/terminal 使用稀疏热图 focal loss，区间使用全域 BCE 与 Dice，末端之后仍参与负例损失。上述改变是模型与监督设计调整，尚不代表实际性能已经提升。

## 数据与标签含义

默认生成 12,000 条合成曲线，数据随机种子为 `2026102203`，与上一轮种子不同。该数据仍依赖探索性 RLGC、夹具、测量误差和噪声先验，不是现场参数标定结果；不同随机种子也不等于跨线路或跨设备的独立实测盲测集。

标签位置按配置传播速度、夹具传播及 Algorithm1 变换得到的 response axis 生成，真实几何位置另存在 metadata 中。事件是否作为正例，取决于单次传播核主瓣相对变换噪声的可观测性启发式；事件不可观测的局部窗口用 `target=-1` 忽略，不视作健康负例。默认事件/末端 Gaussian σ 为 5 m。该规则属于带限传播核弱监督，不是多重反射下的精确峰位，也不是现场金标准。测试集指标衡量模型对 response-axis 标签的拟合。

数据生成默认使用 7 个 worker，并为系统保留一个物理 CPU 核。默认数据目录为 `E:\CableNNDataSet\CodeX\FieldNN_Iter3_20261022`。若该目录已有本轮完整数据，可用 `-SkipGenerate` 直接复用。

## 正式运行

默认配置是 40 epochs、batch size 32、训练 DataLoader 6 workers，验证 DataLoader 使用主进程；U-Net 和 Transformer 在同一 GPU 上顺序运行。学习率为 `3e-4`，AdamW weight decay 为 `0.01`，前 3 个 epoch 线性 warmup 后按验证损失使用 ReduceLROnPlateau。

```powershell
$Experiment = 'D:\GitRepository\Cable-NN\NN_Archive\20261022NNFirstTry\FieldNN_Iter3_20261022'
pwsh -NoProfile -File "$Experiment\run_iteration.ps1" -Epochs 40 -BatchSize 32 -Workers 6
```

若使用其他参数，可以替换上述三个数值。首次运行不要加 `-Resume`。中断后在同一实验目录、同一数据目录下运行：

```powershell
pwsh -NoProfile -File "$Experiment\run_iteration.ps1" -Epochs 40 -BatchSize 32 -Workers 6 -Resume
```

`-Resume` 会让数据生成器检查并续用已完成分片；训练阶段只在对应模型已有 `last.pt` 时恢复该模型。尚未创建 checkpoint 的后续模型会从头开始训练。若需要从头重跑训练，请使用新的 `output` 目录，避免覆盖已有 run。

## 阈值与结果边界

验证阶段分别为 event、interval、terminal 选择阈值：优先满足 precision ≥ 0.85，并在满足条件的阈值中最大化 recall；若没有候选阈值达到目标，则按 F0.5 选择。随后测试集固定使用验证集阈值，不在 test 上调参。事件指标按 5 m 匹配容差校准；三头各自校准。

默认 `field-subset=all` 表示覆盖清单中的现场记录，不表示盲测，也不能把未标注位置当负例计算现场 precision/recall。程序附带的 1500 m 湿接头实验只有 11 个已知正例位置；可报告这些位置的命中情况和额外未核实候选数，不能据此计算完整 precision/recall，且该实验不是独立电缆测试集。

## 产物位置

结果留在项目目录，无需压缩或另行打包：

- `output/iteration3/iteration_summary.json`：两模型的测试汇总与现场输出路径。
- `output/iteration3/logs/`：每个生成、训练、验证、测试和现场检查阶段的命令及日志。
- `output/iteration3/unet/`、`output/iteration3/transformer/`：各自的 `history.csv`、`best.pt`、`last.pt`、配置、阈值、验证/测试评价与 `field/` 现场覆盖结果。
- `output/iteration3/pipeline_status.json`：已完成阶段及对应日志路径，便于中断后确认进度。

完成正式训练前，`output/iteration3/iteration_summary.json` 不存在或不完整都不构成性能结论。CPU smoke 仅确认运行链路，不是模型效果证据。

Batch 32 已做 2 次预热、3 次计时的 GPU 单步对比：两模型约 330 条/秒、分配显存约 2.4 GiB；Batch 64 约 324–332 条/秒、约 4.8 GiB，因此选 32。此为设备计算短测，不含 DataLoader，不能当作完整训练吞吐或完成时间预测。AMP 初始 scale 为 1024，短查无更新跳步。


## 本轮短检查记录

已完成20条生成检查、两个模型GPU前向/反向/保存与恢复、验证/测试入口以及每模型1条原始FDR推理。检查发现默认AMP初始scale 65536使U-Net首步溢出；改为1024后有效更新、恢复续训均无跳步。正式模型尚未训练，不能把 smoke 输出用于性能汇报。最终运行默认 batch 32。
