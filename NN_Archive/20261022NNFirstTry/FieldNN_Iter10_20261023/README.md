> 已归档，现存权重、结果与复现缺口以 [归档索引](../README.md) 为准；下面保留实验原说明。

# Iter10：受限微调与组会成果导出

当前正式结果仍以 Iter8 U-Net 为优先保留模型。Iter9 的局部提议虽覆盖测试已知位置42/51，但评分后U-Net仅13/51，未超过Iter8；本轮不继续扩大该路线。

## 一条命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File "D:\GitRepository\Cable-NN\NN_Archive\20261022NNFirstTry\FieldNN_Iter10_20261023\run_iteration10.ps1"
```

中断后在同一命令末尾加 `-Resume`。默认B64、6个加载进程、主线程1个，启动器沿用项目的CPU亲和设置，保留1个物理核心。不改变显卡配置。正式执行仍由用户排期，与DeepSeek任务错开。

最新夜间运行默认50个完整epoch，B64下为7500次实际参数更新，每100步加初始点共76个真实观测点。依次执行验证组选权重/阈值、213条原始FDR只读推理、原始曲线图、短接参考评价、独立合成域校准/评价及现场表图文档。训练复用Iter8的138条标注现场缓存和Iter4合成缓存；最终评价保留全部213条现场曲线。

## 本轮改动

Iter8原训练6轮、每轮150次更新；最优权重在第1轮。训练现场MIL loss随后接近0，验证loss约0.65，不能据此认定泛化良好。Iter10保留U-Net结构和12通道输入，仍从Iter5权重初始化适配层，主干学习率从3e-5降为1e-5、输入层从3e-4降为1e-4，现场MIL权重由0.5降为0.1，短接权重保留0.5。本轮属于联合调整，不将增益单独归因于某一项。

每100次成功optimizer更新评估一次固定train/val现场组，含step0共76点；AMP跳步不计作成功更新。固定阈值0.94的train/val hit曲线与同定义正例MIL loss均在eval模式计算。候选预算校准后的val hit另画，不混在固定阈值曲线中。保留全部记录，不平滑、不插值补点，不承诺曲线单调。

每个记录点和Iter8参考都按相同val规则比较：在Iter5验证组候选预算内，优先提高group-macro已知位置hit@20m，再比较候选数量和正例loss。旧Iter8权重仍有资格被保留。`best_dense.pt`对应本轮轨迹，`best.pt`对应最终验证选择，`selection.json`明确来源。如果保留Iter8，Iter10曲线只表示探索过程，不称为旧模型的训练历史。

## 结果位置

- `output/existing_iter8/`：本次CPU导出的既有Iter8综合表与现场案例。
- `MODEL_CARD_Iter8.md`：当前最佳模型的结构、训练、数据与评价口径。
- `output/iter10_50ep/iteration10_summary.json`：新一轮状态、各阶段日志与耗时。
- `output/iter10_50ep/unet/history_steps.csv`：全部真实记录点。
- `output/iter10_50ep/unet/figures/`：train/val loss、hit曲线及说明，PNG 200dpi和PDF。
- `output/iter10_50ep/unet/evaluation/field_report.json`：现场分组结果。
- `output/iter10_50ep/unet/nn_benchmark.json`：batch1、预载入输入的NN forward时延，20次预热、100次测量；不含FDR/FFT/特征/H2D/解码。
- `output/iter10_50ep/deliverables/`：最终选定权重的综合表、2–4现场案例、模型结果文档。

## 指标解释

现场仅有部分正例，未知位置不当负例。主要报告一对一已知位置hit@10/20/30m、独立组macro、匹配定位误差与候选数量；Accuracy/Macro sensitivity/Macro F1缺完整现场标签，不能计算。位置命中包含按袋选择最佳单扫频的口径，不能称作逐条扫频准确率。1500m浸水实验单列，不冒充独立多电缆现场证据。案例为明确选择的成功例，完整组级结果同时保留。

合成域F1和短接无电缆参考伪事件数分列；它们不等于现场精度。仅单次训练N=1，不报告未测std/p值。全部保留组已多轮用于研发，不能称全新盲测。结构与训练参数详见模型文档。

2026-10-23夜间用户明确授权CodeX直接执行训练和评估。首次初始化错误已修复；本轮50epoch已由CodeX正式启动并跟进。最终指标以实际生成的结果为准。
