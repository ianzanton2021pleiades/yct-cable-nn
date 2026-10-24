> 已归档，现存权重、结果与复现缺口以 [归档索引](../README.md) 为准；下面保留实验原说明。

# FieldNN PINN（2026-10-23，WorkBuddy）

对 CMDC `FieldNN_Hybrid_20261022` 合成预训练过拟合问题的改进实验。骨干网络与数据缓存完全沿用 CMDC/CodeX 的 Iter4/Iter5，唯一改动是加入 signed 反射系数头与 TDR 前向模型残差，并对训练分布做距离轴随机化与 mixup。

## 结构

| 文件 | 作用 |
|---|---|
| `models_pinn.py` | Iter5 U-Net / Transformer 拓扑 + 第 4 个 signed 反射系数头 |
| `losses_pinn.py` | 三个监督项（与 CMDC 逐字一致）+ `PhysicsResidual` 五个物理残差 |
| `data_pinn.py` | Iter5 `ShardDataset` 基础视图 + 距离轴 warp + mixup + 特征噪声 |
| `train_pinn.py` | 100 轮合成预训练，逐步/逐轮日志，epoch 50 快照 |
| `finetune_field.py` | 实测注入域微调（只读复用 `Deepseek_CMDC/FieldNN_Hybrid_Long_20261022`） |
| `eval_pinn.py` | 合成 test / hybrid test / 现场 test 评估，含 P_ann 与全命中曲线数 |
| `case_study_pinn.py` | 现场案例图，末端"必须在最后事件之后"的物理后处理 |
| `plot_curves.py` | 50/100 轮与基线对照曲线、物理残差曲线 |
| `compare_tables.py` | 多份 `evaluation.json` 汇总为 markdown 表 |

## 物理残差

观测脉冲是反射系数与测量脉冲的卷积，观测阶跃是其积分：

```
h(d) = a(d) * g(d),    s(d) = ∫ a(x) dx
```

`PhysicsResidual` 在 `d ≥ 30 m` 上计算：重构余弦残差、阶跃余弦残差、事件质量落在观测包络上的支撑残差、末端不早于最后反射的顺序残差、`|a|` 的稀疏残差、事件图与 `|a|` 的一致性残差。全部无量纲，因为特征归一化不可逆。

## 运行

```powershell
$py = "E:\Anaconda\envs\gpushare_cu124\python.exe"
& $py train_pinn.py --output output\pinn100 --epochs 100 --workers 4 --epoch-snapshots 50
& $py finetune_field.py --checkpoint output\pinn100\unet\best.pt --output output\field_ft --epochs 12
& $py eval_pinn.py --checkpoint output\field_ft\best.pt
& $py case_study_pinn.py --checkpoint output\field_ft\best.pt
& $py plot_curves.py --history output\pinn100\unet\history.csv --steps output\pinn100\unet\steps.csv --output assets\curves.png
```

长任务由用户执行；`output/` 已在 `.gitignore` 中。
