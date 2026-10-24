> 已归档，现存权重、结果与复现缺口以 [归档索引](../README.md) 为准；下面保留实验原说明。

# Iter11：Iter8 权重上的输入层受限适配

本实验复用 `FieldNN_Iter10_20261023/train_recorded.py`，不复制训练器，也不改动 Iter8、CMDC 或现场数据。目标是只适配 Iter8 已学好的 12→8 输入 stem，冻结 U-Net 主干。

主代理 runner 应透传以下训练参数：

```text
--init-iter8 --stem-only --weak-weight 0.05 --stem-lr 1e-5 --epochs 50
```

训练从 Iter8 U-Net `best.pt` 完整权重初始化，仍使用 Iter8 合成回放和短接参考；MIL 权重为 0.05、短接权重为 0.5。只训练输入 stem 的 96 个参数。模型选择沿用 Iter10 验证规则，并保留 Iter8 权重作为验证集回退。测试组不参与权重或阈值选择。

训练结果由主代理runner保存到该实验目录的`output/`。2026-10-23夜间用户明确授权CodeX直接执行，当前由CodeX顺序运行并跟进训练、评估和成果导出。

## 正式夜间配置

run_iteration11.ps1默认B96、6workers、50完整epoch，对9600合成训练样本每轮100次更新，共5000更新、每100步加初始点共51个真实观测点。只训练96个输入层参数，仍保留完整3888195参数网络推理；stem LR1e-5、MIL .05、短接 .5。与I10同时改变初始化、可训练容量、学习率和batch，不称单因素消融。正式过程顺序使用GPU，不与I10并发。

