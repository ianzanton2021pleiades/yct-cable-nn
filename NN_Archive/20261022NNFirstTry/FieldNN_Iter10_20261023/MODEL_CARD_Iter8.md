# FieldNN Iter8 U-Net 模型卡

## 模型身份与用途

Iter8 U-Net 是面向短电缆 FDR 的研究模型，用于在 0–3250 m 固定距离轴上输出事件、缺陷区间和末端三类响应。当前结果支持“在已知正位置附近提出候选”，不支持现场完整缺陷分类或自动判定。模型没有接入 GUI。

它是在 Iter5 U-Net 上继续训练的 12 通道输入版本：输入层把 12 通道映射到原 8 通道主干，前 8 个输入权重初始化为恒等映射，新增 4 个通道初始权重为零。基础通道数为32，采用七个尺度的编码/解码结构，参数量3,888,195。

主干来自 Iter3 七尺度 1D U-Net：编码分辨率为 8192→4096→2048→1024→512→256→128，通道数 32/64/128/192/192/192/192；各级使用残差卷积、GroupNorm 和 GELU。U-Net bottleneck 串联 dilation 1/2/4/8 的残差块，并把全局均值投影回局部特征。解码使用线性上采样和跳连；最终由三个 1×1 卷积头输出 event、interval、terminal logits。Iter8 在主干前保留 12→8 的输入 stem。

## 输入与训练

- 原始输入是 S11 的 Algorithm 1 距离域响应，名义网格 0–3250 m、8192 点；频段 9 kHz–200 MHz，名义速度 172 Mm/s。现场推理按记录中的 `source_scale_ratio` 转到 FDR source/report 轴；`Wave_Velocity` 是报告轴设置，不是独立测得的材料传播速度。
- 12 个通道包含归一化脉冲、归一化阶跃、全局 RMS、30 m 脉冲局部 RMS 比、30 m 阶跃去趋势、局部 RMS、距离坐标，以及 90/300 m 两种阶跃去趋势的全局/局部归一化特征。
- 训练使用 9,600 条合成训练曲线、21 个现场正例训练袋和 39 条用户确认的短接/无测试电缆参考。现场监督是跨 sweep 正例袋 MIL；未标注位置不作为负例。短接参考只监督 30 m 外的事件/区间负例，末端不监督，也不是健康电缆。
- 配置上限 18 epoch、batch 64、AdamW 学习率 3e-5、6 个加载进程、早停 patience 5、AMP。实际记录 6 个 epoch；被选 checkpoint 对应 history 第 1 轮（零基 epoch 0）。训练加载 Iter5 U-Net 初始权重。CMP 90HX 与 PyTorch 2.5.1+cu124 仅为历史训练记录，本次导出未调用 GPU。

## 阈值与评估口径

合成验证的阈值为 event 0.6、interval 0.975、terminal 0.6。现场候选密度工作点由 field validation 选择：U-Net event threshold 0.94，目标是在不超过对应 Iter5 U-Net 锁定阈值 0.6 的验证候选密度下提高 group macro hit@20。Transformer 的 0.98 工作点使用其对应 Iter5 Transformer 验证预算，因此不能把两种架构说成严格相同候选预算。

Field hit 是已知正位置与候选事件的一对一匹配；各 bag 先选单条最佳 sweep，再按组等权宏平均。未命中位置不等于全部现场缺陷；未命中候选仍未核验。既有 213 条现场曲线已在多轮开发中反复使用，test 是现有数据集内的组隔离结果，不是新采集的盲测。

## 结果

| 数据与工作点 | 范围 | 主要结果 |
|---|---|---|
| Synthetic test, mixed | 1,200 曲线 | event P/R/F1@5m = 0.916/0.732/0.814；interval F1@IoU≥0.5 = 0.896；terminal F1@10m = 0.998 |
| Synthetic test, clean | 1,200 曲线 | event P/R/F1@5m = 0.912/0.858/0.884；interval F1 = 0.927；terminal F1@10m = 0.999 |
| 非湿现场 test, threshold 0.94 | 31 sweeps、6 组、40 个已知位置 | hit@10/20/30m = 10/40、17/40、19/40；micro = 0.250/0.425/0.475；group macro = 0.318/0.509/0.554；matched mean error@20m = 7.99m；候选 93 个，3.00/曲线 |
| 1500 m 湿接头 test, threshold 0.94 | 11 sweeps、1 组、11 个已知位置 | hit@10/20/30m = 7/11、7/11、7/11；候选 13 个，1.18/曲线。此组单独报告，不与 6 个非湿组混称 |
| 全部现场预测，threshold 0.94 | 213 条预测记录 | 213/213 有预测；565 个候选，2.65/曲线。完整现场未知位置仍是 unknown，不能据候选数计算误报率 |
| CalibrationData 短接参考 | train 39、val 12、test 12 | threshold 0.94 离线筛选后 event false events 为 0/39、0/12、0/12；这是夹具短接参考，不是健康电缆误报率 |

非湿 test 组之间差异明显：F65 组 4/4 个已知位置均命中；Nanling F55 group 6/8，单 sweep 示例 3/3；Huzhong F23 group 2/3。下面三图各展示一条 sweep（每例 N=1），仅说明已知位置与候选的对应关系，不构成样本级 Accuracy/F1。

![F65 已知位置案例](/D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter10_20261023/output/existing_iter8/assets/case_unconfirmed_case_2026_5_30_12_13_22深圳110kV坂田变电站10kV四季线F65坂田南学校_1公用智能柜.png)

![Nanling F55 已知位置案例](/D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter10_20261023/output/existing_iter8/assets/case_linked_cable_nanling_f55_june_2026.png)

![Huzhong F23 已知位置案例](/D:/GitRepository/Cable-NN/NN_Archive/20261022NNFirstTry/FieldNN_Iter10_20261023/output/existing_iter8/assets/case_unconfirmed_case_2026_7_19_17_00_31深圳110kV樟树布变电站10kV湖中线F23龙山工业区_5三遥柜.png)

U-Net 在非湿组的 0.94 工作点相对 Iter5 U-Net 0.6：hit@20 从 13/40 增至 17/40，group macro 从 0.399 增至 0.509，候选数从 94 降至 93。作为背景，Iter8 U-Net 固定阈值 0.6 的 hit@20 为 22/40，group macro 0.667，但候选数增至 205；候选更多不能解释为 precision 更高。Iter8 Transformer 的 0.98 工作点为 10/40、group macro 0.337、49 个候选。

## 资源、限制与适用边界

- `best.pt` 模型 state tensor 共 15,552,780 bytes（14.83 MiB）；参数 FP32 估值相同。整个 checkpoint 为 46,818,320 bytes（44.65 MiB），包含训练状态。
- 没有找到该版本可引用的 NN-only median/p95 推理延迟；此卡不报告推理速度。
- 30 m 近端继续屏蔽；不声称首端可检出。末端、远端弱反射和未知位置不能由正例袋指标代替完整验证。
- Field 完整 Accuracy、precision、recall、F1 不可计算，因为现场没有全位置真值；unknown 不当负例。合成指标只表示几何真值下的合成 test 表现，不能外推现场准确率。

## 可复现导出

`export_existing.py` 从 Iter8 现有缓存与 predictions JSONL 读取结果，默认以 threshold 0.94 离线筛选已保存的 0.6 预测，不重新推理。综合表和案例图位于 `output/existing_iter8/`；动态 checkpoint 结果文档为同目录 `MODEL_RESULTS.md`。
