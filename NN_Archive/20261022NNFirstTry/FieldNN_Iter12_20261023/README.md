> 已归档，现存权重、结果与复现缺口以 [归档索引](../README.md) 为准；下面保留实验原说明。

# Iter12：Transformer 输入层受限适配

本实验复用 `NN_Archive/20261022NNFirstTry/FieldNN_Iter10_20261023/train_recorded.py`，通过主代理runner选择Transformer分支。正式50轮从Iter5 Transformer主干初始化，新增12→8恒等输入层（新4通道初始为0），冻结主干，只训练96个输入层参数。Iter8 Transformer仅作为独立验证参考，仍有资格被保留。

配置：batch size 96、50 个合成训练 epoch、stem 学习率 `3e-5`、现场正例 MIL 权重 `0.2`、短接校准权重 `0.5`。训练参数为：

```text
--model transformer --stem-only --weak-weight 0.2 --stem-lr 3e-5 --epochs 50 --batch-size 96
```

现场候选预算沿用 Iter5 U-Net 验证基线，在两个架构之间保持同一预算；Transformer 不另用 Iter5 Transformer 的候选密度。模型选择只依据现场验证集，测试集不参与权重或阈值选择。训练产物由 runner 写入本目录的 `output/`，不纳入 Git。

在Iter11早期固定验证曲线基本保持初始化水平后，Iter12在启动前将现场正例权重和输入层学习率提高，以在96个参数的容量限制内探索更充分的适应。此时尚未评价Iter11测试结果。该组合与Iter11同时存在架构、学习率和监督权重差异，不用于单独归因架构优劣；候选预算和权重选择规则保持一致。

最初从T8初始化的试跑在约8轮内，train MIL约0.002而val MIL约0.738，正例监督已近乎饱和，冻结该主干无法消除其既有泛化差距。该pilot未评分test，输出完整保留在`output/pilot_from_t8_stem`。正式试验改从未使用现场MIL训练的T5主干出发，并重新执行完整50轮；不将pilot曲线拼接成正式曲线。
