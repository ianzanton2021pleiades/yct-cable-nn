> 已归档，现存权重、结果与复现缺口以 [归档索引](../README.md) 为准；下面保留实验原说明。

# FieldNN PINN-v2（2026-10-23，WorkBuddy）

在 `FieldNN_PINN_20261023`（v1）基础上的第二轮改进。骨干、数据缓存、评价口径与 v1 完全一致，改动集中在三处：输入前端、物理残差、优化日程。

## 相对 v1 的改动

| 项目 | v1 | v2 |
|---|---|---|
| 输入通道 | 8（Iter3 原样） | 11：新增 8 m 锐化对比度、100 m 宽反射对比度、8 m 阶跃局部斜率 |
| 物理重构 | 单一 5 m 脉冲 | 3/8/20 m 三个脉冲宽度同时约束 |
| 阶跃约束 | 单一 30 m 去趋势 | 30 m 与 120 m 两个尺度 |
| 事件—反射一致性 | 全序列余弦 | 双向支撑重合 + 软峰位对齐 |
| 背景抑制 | 无（合成样本 `unknown` 全零，稀疏项不生效） | top-128 已知安静点直接惩罚 |
| 学习率日程 | 3e-5 → 1e-6，50 轮后人工重启 | 3 轮 warmup + 单段 cosine（3e-4 → 1e-6，100 轮） |
| 增强 | warp 0.012、mixup 0.25 | warp 0.02、mixup 0.30、噪声 0.35 |

## 文件

| 文件 | 作用 |
|---|---|
| `features_v2.py` | 11 通道前端；`extend_features` 由缓存的 8 通道反解原始迹线后扩展，两条路径逐点一致 |
| `data_v2.py` | 合成 shard / 注入现场 / 实测现场 / 短接四类数据集的 11 通道封装 |
| `losses_v2.py` | 与原版逐字一致的三项监督 + 背景 OHEM；多尺度物理残差 |
| `train_v2.py` | 100 轮单段 cosine 预训练，逐轮记录全部分项 |
| `eval_v2.py` | 合成 clean/mixed、注入现场、实测现场、短接四组评价 |
| `finetune_v2.py` | 实测域微调（与 v1 同配方） |
| `plot_v2.py` | 与 v1 对照的损失、物理残差、F1 曲线 |
| `smoke_v2.py` | 特征反解一致性与物理项梯度检查 |

## 运行

```powershell
$py = "E:\Anaconda\envs\gpushare_cu124\python.exe"
& $py smoke_v2.py
& $py train_v2.py --output output\v2_100 --epochs 100 --workers 4 --eval-f1-every 5
& $py eval_v2.py --checkpoint output\v2_100\unet\best.pt
& $py finetune_v2.py --checkpoint output\v2_100\unet\best.pt --output output\field_ft --epochs 12
& $py eval_v2.py --checkpoint output\field_ft\best.pt
& $py plot_v2.py
```

`output/`、`assets/`、`smoke_*` 已在 `.gitignore` 中。

## 结果（100 轮）

| 指标 | v1 | v2 |
|---|---|---|
| 验证监督损失（第 100 轮） | 0.486 | **0.449** |
| 验证监督损失最低点 | 0.448 @95 | **0.368 @63** |
| 事件项（同轮次预训练） | 0.661 | **0.517** |
| 合成 clean F1@5m | 0.795 | **0.842** |
| 合成 mixed F1@5m | 0.736 | **0.776** |
| 注入现场 hit@20（微调后） | 0.251 / 4.65 候选 | **0.421 / 3.34 候选** |
| 原始现场 hit@20（阈值 0.5） | 0.405 / 3.72 候选 | 0.342 / 2.98 候选 |
| 原始现场末端 50 m 内 | 0.571 | **0.684** |

第 17 轮因主机内存不足中断，`output/v2_100_phase1` 为第一段，`output/v2_100_cont` 为续训段，`history_stitched.csv` 为拼接结果。

分析报告：`AgentsDoc/2026-10-23_PINN2_多尺度前端与物理一致性.md`。
