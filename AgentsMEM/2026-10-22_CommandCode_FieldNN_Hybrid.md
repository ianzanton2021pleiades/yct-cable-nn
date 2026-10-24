# 2026-10-22 CommandCode FieldNN Hybrid Session

## Session 基本信息

- 日期：2026-10-22，Asia/Shanghai；Agent：CommandCode（CMDC）；工作目录 `D:/GitRepository/Cable-NN`。
- 用户目标：跳出“纯仿真”圈，用**实测 S11 + 注入已知反射**构造现场域数据集；解决 train/val loss 差距和现场过检；组会临近，需要效果好的 Transformer/U-Net（或集成/变体）。
- 资源：`E:\Anaconda\envs\gpushare_cu124\python.exe`；NVIDIA CMP 90HX 10 GiB；未改 GPU 配置。
- 授权：新实验 `DifferentAgent/CMDC/FieldNN_Hybrid_20261022`；新数据 `E:\CableNNDataSet\DeepSeek_CMDC\FieldNN_Hybrid_20261022`；现场目录只读；不提交 Git；长任务由 Agent 直接执行（用户本轮明确选择）。

## 用户 grill-me 决策（第 1 轮）

1. 现场实测数据：实测 S11 做底 + 注入已知反射，专家注释做弱正例，整组留出（推荐项）。
2. 数据划分：按 case 随机划分，同一条电缆不同日期可分到训练和测试（用户明确选择，不是整组隔离）。
3. 第三个模型：可以 U-Net+Transformer，也可以尝试其他模型/组合/结构/变体，核心是效果好。
4. 执行方式：Iter8 跑完后由 Agent 直接跑完整实验并给结果。

补充：我保留 1500 m 湿接头和 5 个整电缆 strict holdout 强制 test，并单独报告 strict 子集，避免组会被问“同电缆泄漏”时无话可说。

## 起始诊断

- Iter8 `history.csv`：`train_field_positive` 从 0.62 降到 0.002，`val_field_positive` 0.72–0.78 不降；合成域 train/val 仅 0.30 vs 0.40。结论：模型在背 21 个现场正例袋，不是学现场特征。
- Iter8 U-Net 固定 .6/.99/.4 测试：6 非湿组 macro hit@20 .433→.675，但候选/组 3.72→8.06；wet 7/11→8/11，候选 35→57。命中上升主要来自候选膨胀，不能称 precision 提升。
- 现有 `field_evidence.json`：177 条记录、49 个 case 有 engineer_note 位置、45 个 cable_group 有位置证据；1500 m 湿接头 11 个已知位置；CalibrationData 63 份可用短接（用户已确认无被测电缆）。

## 本轮实现

目录：`DifferentAgent/CMDC/FieldNN_Hybrid_20261022`。

1. `prepare_hybrid.py`
   - 从 `.fdr` SQLite `MeasureData_<rowid>` 读复数 S11，重采样到 9 kHz–200 MHz。
   - 末端估计：高置信工程师全长×0.85；否则 REF `detect_endpoint_from_ifft`×0.85；否则 1000 m；上限 2400 m。
   - 注入 `Γ(f)=A·exp(-2π²f²σ_t²)·exp(-j4πfd/v)`，σ_t=2σ_d/v；点事件/区间（两个反号反射）；σ_d=3–35 m；A 按局部背景 RMS 定标，目标峰值/背景 1.3–15；避开已知工程师位置 40 m、事件间距 60 m。
   - 扰动：平滑复增益/相位纹波、±2 m 时移、复高斯噪声。
   - 标签：注入事件权重 1.0、工程师位置权重 0.5、其他 -1 未知；terminal 不监督；CalibrationData d≥30 m 事件/区间负例。
   - 未知区 top-64 事件概率稀疏惩罚 0.02。
   - 保存 injected features/targets/records + 原始 field features/records（评价用）+ calibration 数组/records + split/summary。
2. `models.py`：Iter3 同拓扑 + bottleneck dropout 0.1；无新增参数，Iter5 checkpoint strict 加载。
3. `losses.py`：[B,5,L] 目标（event/interval/terminal/weight/unknown）；加权 focal、加权 interval BCE+Dice、top-k unknown sparsity。
4. `train.py`：50% 合成 replay、45% 注入实测、5% 短接负例；WeightedRandomSampler；AdamW 3e-5、wd 0.02、AMP1024、batch64、6 workers；EMA 0.999；每 epoch raw/EMA 都评估；选择分数 `syn+0.5hyb+0.1cal+0.5(1-hit20)+0.05cand`；早停 6。
5. `calibrate.py`：event 阈值在合成 val F1@5m 与注入 val hit@20 间平衡，限制候选 ≤3/curve；interval/terminal 合成 precision-first。
6. `evaluate.py`：合成 clean/mixed、注入 hybrid test、原始现场 test、1500 m wet、CalibrationData test；strict 子集单独输出。
7. `run_hybrid.py/.ps1`：准备→训练→校准→评价→集成（U-Net+Transformer 概率平均）。
8. `plot_results.py`：输出对比表和图到 `Image/20261022_FieldNN_Hybrid_metrics.png`。

## 已完成验证

- 注入符号/位置验证：单条实测曲线注入 300/800/1500 m，Algorithm1 峰精确落在 299.9/800.0/1500.0 m；σ_d=5/15/30、A=0.005–0.08 的峰值比例符合预期。
- `prepare_hybrid.py --smoke`：40 条实测、80 条注入样本，0 error；train 26 / val 12 / test 42；注入事件 1 个 52 条、2 个 28 条；target ratio 1.31–14.9；end estimate 246–1909 m。
- `train.py` U-Net 2 batch：AMP 0 跳步，grad norm ~27–33，峰值显存 0.68 GiB（batch 8 冷启动）；`val_syn .119`、`val_hyb 2.62`、`hyb_hit20 .40`、`hyb_cand 3.75`、`cal_false 0`；仅链路验证，不是效果。
- `calibrate.py`：阈值 event .7 / interval .95 / terminal .7；合成 val F1 .774、注入 val hit20 .267、候选 2.92；仅 smoke。
- `evaluate.py`：合成 test clean F1 .852 / mixed .773、interval .809、terminal .976；注入 test hit20 .333、候选 2.57；现场 test hit20 .246、候选 2.45；wet 1500 hit20 .636、候选 2.91；short false 0/15。均为 2-batch smoke，不代表正式效果。

## 配方迭代与 pilot 结果

### 第一版（50% 合成/45% 注入/5% 短接，注入目标比 1.3–15，稀疏 0.02）

- 2 epoch U-Net：`val_hyb` 1.102→1.084，`hyb_hit20` .559→.593，`hyb_cand` 4.0→5.0，`val_syn` .412→.501；best epoch 1。
- 测试：合成 clean F1 .853 / mixed .768；现场非湿 43 条 hit20 .253、候选 1.65；wet 1500 hit20 .727、候选 0.82；strict（含 wet）hit20 .355；短接 0/15。
- 诊断：模型偏保守，现场非湿命中低；注入幅度偏强，真实工程师弱正例权重 0.5 偏低；选择分数只看合成/注入。

### 第二版（45% 合成/45% 注入/5% 原始现场弱正例/5% 短接，注入目标比 1.2–8，工程师权重 0.75，稀疏 0.01，现场 val 参与选择/阈值）

- 数据重新生成：213 条现场、0 错误；train 1,268 注入 + 84 原始现场弱正例、val 108、test 236。
- 3 epoch U-Net：epoch1 `val_syn` .414、`val_hyb` 1.139、`hyb_hit20` .421、`hyb_cand` 4.13、`field_hit20` .389、`field_cand` 3.78、score 1.768；epoch2 field_hit .444 但 candidates 5.56；epoch3 field_hit .472 但 candidates 5.89；best epoch 1。合成 val 从 .414 升到 .467，已出现过拟合趋势。
- 阈值：event .6 / interval .95 / terminal .5；合成 val F1 .776、现场 val hit20 .278、候选 2.22。
- 测试：合成 clean F1 .852 / mixed .771、interval .803、terminal .971；注入 hybrid hit20 .361、候选 2.59；注入 strict（5 整电缆）hit20 .158、候选 2.69；现场非湿 hit20 .299、候选 1.81；现场 strict（5 整电缆）hit20 .323、候选 2.14；wet 1500 hit20 .727、候选 1.27；短接 0/15。
- 结论：新配方把现场非湿 hit20 从 .253 提到 .299，候选 1.65→1.81；wet 保持 .727；strict 现场 .323。模型仍偏保守，远端弱反射和 strict 注入事件命中不足。

## 正式结果与收尾

- 首次正式运行发现 `run_hybrid.py` 的 Iter5 路径写错（`DifferentAgent/CodeX/...`，实际在 `DifferentAgent/CMDC/...`），U-Net 从随机初始化开始，val_syn 2.17；已停止并把该次输出移到 `output/hybrid_scratch_init`。修正为 `HERE.parent / "FieldNN_Iter5_20261022"` 后重跑。
- 修正后 U-Net/Transformer 的 best 都在 epoch 1；U-Net 到 epoch7 早停，后续 val_syn .414→.488、val_hyb 1.14→1.96、field_cand 3.89→7.44，注入过拟合被早停挡掉。Transformer 同样在 epoch7 早停，best epoch 1。
- 0.6 精度点：U-Net 现场 non-wet hit20 .274/候选1.93、strict .323/2.10、wet .727/1.27；Transformer .342/2.05、.384/2.43、.636/1.55；集成 .274/1.67、.313/1.95、.636/1.55；短接均0/15。
- 0.5 平衡点：U-Net .416/3.51、.525/4.10、.818/3.09；Transformer .390/3.35、.434/3.57、.636/2.64；集成 .390/2.93、.455/3.33、.727/1.82；短接均0/15。合成分数：clean .836–.859、mixed .757–.781、interval .783–.818、terminal .968–.971。
- 结论：U-Net 0.5 现场召回最高；集成 0.5 候选最低、合成最好；0.6 是精度优先点。现场指标仍是已知正例命中率+候选密度，不是 precision/recall；case-level 划分和 5 条 strict 电缆 + wet 的边界未变。
- 产物：`output/hybrid/run_summary.json`、各模型 `evaluation/evaluation.json`（0.6）与 `evaluation_e05/evaluation.json`（0.5）；图 `Image/20261022_FieldNN_Hybrid_metrics_e05.png`、`Image/20261022_FieldNN_Hybrid_training.png`；技术说明 `AgentsDoc/2026-10-22_FieldNN_Hybrid_实测注入数据集.md`；Milestone 已追加 22:45 条目。
- 未修改旧 Iter3/Iter5/Iter8 代码、原始 FDR、GPU 配置；未提交 Git。

## 长训练最终结果（20 变体，2026-10-22 晚）

- 用户要求：好看的 train/val loss 曲线、precision/sensitivity/F1 综合表（含延迟/模型大小/p 值）、2 个现场 FDR 案例，不要 RG58。
- 新数据 `E:\CableNNDataSet\DeepSeek_CMDC\FieldNN_Hybrid_Long_20261022`：train 2,530 注入（127 trace × 20）、val 108、test 236，0 错误。
- `config_long.json`：lr 8e-6、cosine、30 epoch 上限、dropout 0.15、wd 0.03、域内 batch（45/45/5/5）；从 v10 best 微调。
- U-Net best epoch2、Transformer best epoch1；均在 epoch9–10 早停；后续 val_syn/val_hyb/候选密度上升的 epoch 被丢弃。
- 0.6 精度点：U-Net field non-wet .2737 / P_ann .6190 / cand1.953 / strict .3232 / wet .7273；Transformer .3474/.7174/2.140/.3939/.6364，p=.0400；Ensemble .3053/.7160/1.884/.3636/.6364，p=.0328。合成 clean F1 .851–.867、interval .799–.825、terminal .969–.972。
- 0.5 平衡点：U-Net .4263/.5226/3.605/.5253/.8182；Transformer .4053/.4695/3.814/.4545/.6364；Ensemble .4158/.5563/3.302/.4949/.7273。Ensemble Macro F1 .8841，合成 clean F1 .8584。
- 延迟（CMP 90HX, batch1）：U-Net ~7.5 ms、Transformer ~7.5 ms、Ensemble ~15.5–16.0 ms；参数量 14.83/16.40/31.23 MB。p 为 12 个非湿 case 的 per-case hit@20 配对 bootstrap。
- 损失曲线：`Image/20261022_FieldNN_Hybrid_loss_curves.png`（Iter5 预训练 + long 微调 synthetic/hybrid + field hit/候选）。
- 案例：`Image/20261022_FieldNN_Hybrid_case1.png` = 1500m 湿接头 1100m sweep，1/1、3 候选；`case2.png` = 2025-8-20 山东驻马店 B 相，4/4 接头（577/1177/1758/1885m）、6 候选。
- 综合表：`output/hybrid_long/metrics_table_e05.md`、`metrics_table_e06.md`；技术说明 `AgentsDoc/2026-10-22_FieldNN_Hybrid_实测注入数据集.md` §8；README/Milestone 已更新。
- 边界：P_ann 是下界；现场敏感度是已知正例命中率，不是 precision/recall；case-level split/5 整电缆 strict/1500m 单实验的边界不变；未提交 Git。

## 50 epoch 长曲线与最终选型（2026-10-23 凌晨）

- 用户要求：50–100 step/epoch 的好看 train/val loss + accuracy 曲线、综合表、2–4 个现场案例；GPU 若被 ChatGPT 占用则排队。检查时 GPU 0 MiB，无占用。
- 新增 `train_pretrain.py`：合成域 9,600/1,200，U-Net 从头 50 epoch（7,500 step）、batch64、lr3e-4、cosine、EMA。epoch50 train/val loss .445/.543，train/val F1@5m .815/.712，terminal .971，best F1 .715。曲线 `Image/20261022_FieldNN_Hybrid_pretrain50_curves.png`。
- scratch best 的 10 epoch 现场微调：best epoch2，field non-wet .345/3.56，未超过 long U-Net 0.5（.426/3.61、P_ann .523）。scratch 模型 F1 .715 < Iter5 backbone .84–.87，故只作长曲线/对照，不作部署。
- `config_stable.json` 冻结 encoder + mixup 的 50 epoch field 训练：best epoch6；0.5 测试 field non-wet .432/4.19、P_ann .456、strict .515/4.48、wet .818/3.73；0.7 精度点 .194/0.89；未超过 long U-Net 0.5。曲线 `Image/20261022_FieldNN_Hybrid_stable_curves.png`。
- 最终部署仍选 long U-Net 0.5（召回）或 Ensemble 0.6（精度/综合）；Transformer 0.6 中间选项。综合表沿用 `output/hybrid_long/metrics_table_e05.md`/`e06.md`。
- 4 个现场案例（long U-Net 0.5）：case1 湿接头 1100m 1/1/3 候选；case2 湿接头 900m 1/1/4；case3 驻马店 B 相 4/4/6；case4 消防#3 C 相 3/3/5。图 `Image/20261022_FieldNN_Hybrid_case1..4.png`。
- 边界不变：field 只有弱正例；50 epoch scratch/field 长训练都证明继续堆 field epoch 会过拟合；最终模型是早停/强 backbone 版本。

## 50 → 100 轮续训（2026-10-23 凌晨）

- 用户要求保留 50 轮结果，并在其基础上续到 100/150，保存到新目录、不要覆盖 50。
- 50 轮目录 `output/pretrain50` 不动，完整备份 `output/pretrain50_saved`；100 轮 `output/pretrain100` 从 50 `last.pt` 复制后 `--resume --restart-scheduler`，新 cosine lr 5e-5→1e-6，epoch51–100，再 7,500 step。
- 合成 test：50 轮 checkpoint epoch38，P/R/F1@5m .7786/.7543/.7663，interval .6428，terminal .9731；100 轮 epoch81，P/R/F1 .7865/.7770/.7817，interval .7137，terminal .9708。100 轮 F1 +.015、interval +.071。
- 但 100 轮 train/val loss .255/.584（gap .329），50 轮 .445/.543（gap .098）；100 轮 best 在 epoch81，后段过拟合，不是“无过拟合”模型。
- 100 best 的 10 epoch field 微调 best 只在 epoch1：field non-wet .297/2.97，未超过 long U-Net .426/3.61；epoch4 field .531 但候选 6.09，按统一分数不保留。部署仍 long U-Net 0.5 / Ensemble 0.6。
- 150 未继续：100 gap 已 .329；如需要可从 `output/pretrain100/unet/last.pt` 再开新 cosine 续 150。
- 曲线：`Image/20261022_FieldNN_Hybrid_pretrain50_curves.png`（0–7,500 step）、`pretrain100_curves.png`（0–15,000 step）；技术说明 §8.5。

## 关键文件

- 实验：`DifferentAgent/CMDC/FieldNN_Hybrid_20261022/{prepare_hybrid.py,hybrid_data.py,models.py,losses.py,train.py,calibrate.py,evaluate.py,eval_utils.py,run_hybrid.py,run_hybrid.ps1,plot_results.py,README.md}`
- 数据：`E:\CableNNDataSet\DeepSeek_CMDC\FieldNN_Hybrid_20261022`
- 结果：`DifferentAgent/CMDC/FieldNN_Hybrid_20261022/output/hybrid`
- 图：`Image/20261022_FieldNN_Hybrid_metrics.png`

## 注意事项 / 风险

- 用户选择 case-level 随机划分；同一电缆不同日期可跨 split，现场指标可能偏乐观。strict 子集（5 整电缆 + wet）单独报告，但仍不是全新盲测。
- 工程师位置是弱正例；未知位置不是负例；现场只报告已知正例命中率和候选密度，不报告 precision/recall。
- 注入是带限高斯反射，不是完整 RLGC 接头模型；真实域背景 + 精确位置，但不是现场缺陷物理的完整复现。
- 1500 m 湿接头只有一个实验组；不能证明跨电缆泛化。
- CalibrationData 是夹具短接、无被测电缆；只能做 d≥30 m 负例，不是健康电缆。
- Windows DataLoader 之前有 6 workers 死锁历史；本轮正式训练用 6 workers，若卡住改 `--workers 0` 续训。
