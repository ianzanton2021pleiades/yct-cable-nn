# CommandCode FieldNN Iter5 Session（2026-10-22）

## Session 基本信息

- 日期：2026-10-22，Asia/Shanghai；Agent：CommandCode（CMDC）；工作目录 D:/GitRepository/Cable-NN。
- 目标：接续 CodeX Iter4，用 U-Net / Transformer 快速训练、推理、评估；明天组会；现场“不乱报”。
- 资源：`E:\Anaconda\envs\gpushare_cu124\python.exe`；NVIDIA CMP 90HX 10 GiB；7 训练 workers，保留 1 物理核；未改 GPU 配置。
- 约束：现场 213 条只做评价；不提交；原现场目录只读；长任务由本 Agent 直接跑（用户要求尽快）。

## 用户在本轮的决策（grill-me 第 1 轮）

1. 其他现场案例的工程师记录：让我去案例文件夹找。
2. 现场 213 条：只做评价，不参与训练。
3. 首端 20–24 m 候选：抑制（选项 2/3 都可以）。
4. 工作点：精度优先，U-Net 0.6 / Transformer 0.5（基于 Iter4 的建议值）。

## 起始状态（读取 Iter4）

- Iter4 已完成：`FieldNN_Iter4_20261022/output/adaptation4`；U-Net best epoch 21（1-based 22），Transformer best 15（16）。
- Iter4 mixed test：U-Net P/R/F1@5m .862/.768/.812，interval .865，terminal .9995，候选 4.56/条；Transformer .858/.762/.807，interval .875，terminal .998，候选 3.75/条。
- 1500 m 湿接头：U-Net 7/11、Transformer 6/11；额外候选 88/41。
- 关键诊断：900–1400 m 的湿接头脉冲峰值仅背景 1.1–2.3 倍，1100 m 以上阶跃跳变落到背景 95 分位；不是阈值问题。现场候选 49–56% 在 <100 m，20–24 m 在 36–42% 的不同电缆上重复出现。

## Iter5 实现

目录：`DifferentAgent/CMDC/FieldNN_Iter5_20261022`。数据复用 `E:\CableNNDataSet\CodeX\FieldNN_Iter4_20261022` 只读缓存，没有重新生成 S11。

1. `data.py`：增强样本的 event/terminal 目标按 `sigma=min(cap, sqrt(5²+(0.6·sigma_input)²))` 重渲染（cap 20/22 m），35% 干净回放保留 5 m 标签；bank 注入 1–6×。
2. `decode.py`：`NEAR_END_SUPPRESSION_M=30`，`event_prominence=0.10`；可由 `cfg.evaluation.near_exclusion_m` 覆盖。
3. `train.py`：从 Iter4 best weights-only 微调；AdamW 1e-4、B32、7 workers、AMP 1024、warmup 1、patience 6。
4. `evaluate.py`：val precision target 0.90，test 冻结；clean/mixed 由 `FIELDNN_EVAL_VIEW` 控制。
5. `field_eval5.py`：包装 Iter3 field_eval，先 import 自己的 decode 再 import field_eval（避免 Iter3 路径覆盖）；抑制 30 m 并写 summary。
6. `run_iteration5.py/.ps1`：训练→val→test mixed/clean→213 条现场；支持 `-Resume`、`-Models`、`-SkipField`。

踩坑记录：`field_eval5.py` 最初 `sys.path` 顺序错误，Iter3 的 decode 被先导入；另外 `if path not in sys.path` 会跳过把 HERE 重新插到最前。最终改为 `insert(0, ITER3); insert(0, HERE)`，且在 import field_eval 前 import 自己的 decode。

## 正式结果

- U-Net：16 轮、best epoch 10；mixed P/R/F1 .945/.703/.806，interval .906，terminal .998；clean F1 .882/.930/.999；现场 7/11，额外 28，候选 2.65/条。
- Transformer：13 轮、best epoch 7；mixed .923/.693/.792，interval .881，terminal .997；clean F1 .881/.921/1.0；现场 6/11，额外 21，候选 2.21/条。
- 阈值：两模型 event 0.6、interval 0.99、terminal 0.4。Transformer 0.5 的备选结果保留在 `evaluation_*_e05` 和 `field_e05`（mixed R .724、候选 3.54/注释曲线）。
- 结论：已知命中不变，现场候选 -41~42%，已知序列额外候选 -49~68%；每个候选的工程师注释命中率提高约 45–50%。

## 现场证据台账（新资产）

子代理只读挖掘 `E:\20261010截至的全部fdr文件`，生成 `field_evidence.md/json`：177 条记录、50/60 case group 有位置证据（49 个 engineer_note）、1500 m 11 个位置与旧常量完全一致。台账标注了坐标/相别冲突（§4），例如 2026-7-15 1715m = 2026-7-20 206m、2026-6-23 现场读数与工程师读数系统性错位。评价器 `field_evidence_eval.py` 做 case 级、跨 sweep 取最佳；不要用这些注释调参。

## 关键文件

- 结果：`DifferentAgent/CMDC/FieldNN_Iter5_20261022/output/iter5/iteration5_summary.json`
- 文档：同目录 `README.md`、`HANDOFF.md`；`AgentsDoc/2026-10-22_FieldNN_Iter5_现场调优结果.md`
- 图：`Image/20261022_FieldNN_Iter5_precision.png`
- 证据：`DifferentAgent/CMDC/FieldNN_Iter5_20261022/field_evidence.{md,json}`、`output/evidence_comparison.json`
- 模型：`output/iter5/{unet,transformer}/best.pt`（output/ 已 gitignore）

## 下一轮建议

1. 若现场更看重弱远端：优先改前端/带宽或做物理一致性后处理，不要降阈值硬凑 900–1400 m。
2. 30 m 首端抑制对 <300 m 电缆偏激进；可改 20 m 或按长度 2% 自适应。
3. 若要继续压误报：用现场 213 条做无标签一致性/熵正则（需用户批准），并保留 1500 m 与 engineer 台账作为独立评价。
4. 若要接 GUI：把 `models.py` + `features.py` + `decode.py` 打包成库，保持与 `yct-TDR-2-GUI-multisamples-FreqSel_causal_fixed_nn.pyw` 的神经网络下拉框一致；不要改 GUI 的非神经网络部分。
