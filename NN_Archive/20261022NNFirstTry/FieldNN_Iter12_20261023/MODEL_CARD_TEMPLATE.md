# FieldNN 最终候选模型卡

## 模型身份与用途

**模型：** {{MODEL_NAME}}（{{ARCHITECTURE}}）  
**现场 event 阈值：** {{EVENT_THRESHOLD}}  
**阈值来源：** {{THRESHOLD_SOURCE}}

该模型用于在当前 FieldNN 现场特征上提出已知正位置附近的阻抗突变候选，不构成缺陷分类、完整现场判定或自动维修结论。

## 输入与结构

{{ARCHITECTURE_DETAIL}}

12 通道特征输入覆盖 0–3250 m 的 8192 点距离网格；FDR 频率范围为 9 kHz–200 MHz。项目应用目标是 2.5 km 及以下短电缆；3250 m 网格保留边界与传播轴的计算余量。首端固定 30 m nominal 抑制会隐藏该区间内的真实响应，结果不代表近端可检出。

## 数据范围与分组

{{DATA_SUMMARY}}

现场 split 和曲线覆盖沿用冻结的 Iter8 特征缓存。val/test 按电缆组隔离；1500 m 湿接头组单独报告。全场记录持续保留用于推理；没有已知正位置的曲线仍为 unknown，不作为负样本。

## 训练目标与选择

{{TRAINING_DESCRIPTION}}

{{TRAINING_OBJECTIVE}}

{{SELECTION_DECISION}}

{{INTERPRETATION_NOTES}}

Iter10、Iter11、Iter12 都必须完成至少 50 个等效 epoch 才能成为 bestU/bestT；若 `best.pt` 最终选中 `iter8_reference`，该轮只作为独立探索曲线展示，不作为新模型候选。若 best.pt 选中 step 0 的 T5 初始化 checkpoint，文档会明确标出初始化权重来源和步数，不称为训练后的改进模型。组间选择先比较非湿保留组 group macro hit@20 m（round 到 12 位），再比较命中数、候选负担、验证集候选数和验证 MIL loss。若新轨迹相对 U5 少至少 4 个已知 hit@20 位置，或现场候选数超过 U5 的 1.25 倍，则回退到对应 Iter8 参考并写明原因。回退权重不会被称作 50 epoch 训练结果。

## 三轮训练记录与曲线

以下每轮都保留自己的真实记录点、选中 source 和曲线；不同实验轨迹不拼接。field loss 图记录固定定义的正例 MIL loss，不是合成+MIL+calibration 混合训练总 loss。

{{EXPERIMENTS}}

### 当前训练结果的解释

I11 从已有 I8 权重开始，仅更新 96 个输入层参数。其 train MIL loss 首末为 0.1999→0.1994，val 为 0.6495→0.6548；训练稳定，但约 0.45 的 train/val 差距仍然存在。固定 0.60 的高命中曲线不代表该差距已解决，也不表示从零训练获得了收敛提升。50 轮是完整实验时长，最终按验证规则选择 step 1100 的权重；推理包原始 epoch=10 使用从 0 开始的编号，对应第 11 轮。

## 现场评估结果

**最终候选非湿 test：** {{NONWET_SUMMARY}}

**1500 m 湿接头 test（单独报告）：** {{WET_SUMMARY}}

| 比较模型 | event 阈值 | 非湿 hit@10/20/30 m | group macro@20 m | 候选 / sweep | matched mean error@20 m | 湿组 hit@20 m | 湿组候选 |
|---|---:|---:|---:|---:|---:|---:|---:|
{{COMPARISON_TABLE}}

{{COVERAGE_POINT}}

完整历史 U5/U8/T8/I10/I11/I12 指标见 {{COMPARISON_ARTIFACTS}}。

| 候选 | checkpoint | event 阈值 | 阈值来源 | tie-break 验证资料 |
|---|---|---:|---|---|
{{MODEL_REFERENCES}}

所有现场位置指标复用 Iter6 `evaluate_field.py` 正例匹配规则。未命中位置不代表不存在其他缺陷；未匹配候选仍未核验，因此不计算完整现场 Accuracy、precision、recall 或 F1。保留组在多轮开发中已反复使用，结果不是新盲测。Iter11 与 Iter12 同时改变了架构、stem 学习率和现场弱监督权重，不能据此单独归因架构差异。

## NN-only 参数与推理计时

{{RUNTIME_SUMMARY}}

只有实际 benchmark JSON 存在时才填入 warmup、重复次数与时延；该测量仅覆盖输入已预载入后的神经网络 forward，不包括 FDR 读取、FFT、特征计算、数据搬运和后处理。

## 现场案例

案例图只展示已知位置与候选的空间对应关系，不构成额外的样本级性能指标。

{{CASE_IMAGES}}

## 交付文件

- 精简推理包：{{PACKAGE_PATH}}（{{PACKAGE_BYTES}} bytes），epoch={{PACKED_EPOCH}}；由 {{PACKED_SOURCE}} 在 CPU 上读取，仅保留 `model_name`、`config`、`model`、`epoch`、`selection`。
- 推理阈值配置：{{THRESHOLDS_PATH}}；两个文件都包含 event、interval=0.99、terminal=0.4。
- 最终索引：{{DELIVERY_INDEX}}
- 其余比较与案例文件均使用绝对路径链接；案例来源与曲线分别列于上文。

## 适用边界

现场标注只覆盖已知正位置；unknown 位置保持 unknown，未匹配候选需要人工核验。该模型尚不构成 GUI 集成或完整现场自动诊断能力。
