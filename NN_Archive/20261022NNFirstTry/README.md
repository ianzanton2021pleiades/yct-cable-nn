# 20261022NNFirstTry：首轮 FieldNN / Hybrid / PINN 归档

归档日期：2026-10-24；整理：Codex。该路线暂停，归档用于了解已有尝试、查看结果和按需复用，不作为后续默认训练路线。目录按实验名平铺，不压缩、不提交。

## 先读什么

- 本索引：实验顺序、结论、现存权重和复现缺口。
- `Evidence/`：从现有正式输出摘取的汇总表和阶段摘要，纳入 Git，读结果不需要重新训练。
- 每个实验的 README、代码、配置：保留当前工作树版本。旧 README 中的训练命令是历史入口，不表示它提到的每个权重现在都存在；以本索引的清点为准。
- `output/`、`assets/`：本地完整结果和日志，沿用实验 .gitignore 不纳入 Git。没有删除正式结果。
- 项目 `AgentsDoc/` 的 Hybrid、Iter5、PINN、PINN2 报告及 `Image/` 图件保留原位置。

归档前15个实验共1462个文件、约415.06 MiB，其中9个PT约263.94 MiB。移动不会减小物理磁盘占用；文件变更数也不等于试验数量。

## 实验顺序与得到的经验

| 实验目录 | 原Agent | 主要尝试与归档结论 |
|---|---|---|
| FieldNN_20261022 | CodeX | 三头整曲网络与合成到现场基线；保留数据生成、模型、训练、推理和复数S11诊断。现存无PT，不应把旧README列出的产物当成现存权重。 |
| FieldNN_Iter2 | CMDC | V3八通道TDR链；现存4轮pilot结果（event F1@5m约0.435、interval约0.108、terminal hit约0.434），不是成熟模型；旧四通道/V2脚本前轮已删。无PT。 |
| FieldNN_Iter3_20261022 | CodeX | 响应轴弱标签与多尺度输入。合成event F1@5m U-Net 0.8736、Transformer 0.8879；现场湿接头须用修正后的源/报告轴：hit@10m为3/11、5/11。合成强不代表现场强。无PT。 |
| FieldNN_Iter4_20261022 | CodeX | 复用Iter3 S11，展宽、距离增益、夹具扰动与clean replay；建立现场化缓存。保留adaptation_summary、历史、日志，当前无PT。 |
| FieldNN_Iter5_20261022 | CMDC | 现场工作点与候选负担调节；非湿保留组U-Net hit@20m 13/40、94候选/31曲线；湿组7/11。保留两模型PT。 |
| FieldNN_Hybrid_20261022 | CMDC | 实测背景注入+合成预训练/微调。原评估口径非湿hit@20约0.426、3.605候选/曲线；湿组约0.818。划分与后期Iter8/11不同，不直接排名。保留两模型PT。 |
| FieldNN_Iter6_20261022 | CodeX | 正例袋MIL、冻结教师、分组现场证据；未标注区域保持unknown。现存正式摘要与现场预测，无独立PT。 |
| FieldNN_Iter7_20261022 | CodeX | 继续现场适配；固定阈值候选膨胀，限制候选后远端湿接头没有稳定改善。保留结果，无PT。 |
| FieldNN_Iter8_20261022 | CodeX | 12通道宽尺度阶跃+真实夹具短接监督；U-Net相近预算hit@20由13/40升至17/40，93候选/31曲线，湿组仍7/11。保留两模型PT。 |
| FieldNN_Iter9_20261022 | CodeX | 物理候选+局部网络+弱正例袋PU，未超过Iter8。前轮已删除训练/评价链；只保留候选生成/审计与已有结果，不能称可训练复现。 |
| FieldNN_Iter10_20261023 | CodeX | 逐更新记录的长轨迹适配；正式50轮U-Net hit@20为14/40、89候选/31曲线，未超过Iter8。训练日志和历史保留，无独立PT。 |
| FieldNN_Iter11_20261023 | CodeX | 冻结U-Net主干，仅适配96个stem参数；50轮后按验证选step1100。主阈值0.94：17/40、94候选/31曲线，未超过Iter8；固定0.60：22/40但203候选，不能称同预算改进。现存推理权重在Iter12导出包。 |
| FieldNN_Iter12_20261023 | CodeX | Transformer受限stem适配，最后仍回退历史Iter8 Transformer；同时汇总I11 U-Net推理包与全场结果。不是新增更优Transformer。保留唯一最终inference.pt。 |
| FieldNN_PINN_20261023 | Workbuddy | 物理残差+反射系数头。150轮合成clean F1约0.796；原现场非湿约0.416/4.30候选，与Hybrid约0.426/3.60大致打平，湿组下降。只剩field_ft_150权重，预训练PT缺失。 |
| FieldNN_PINN2_20261023 | Workbuddy | 多尺度前端、OHEM与物理一致性联合修改；clean F1 0.842、mixed 0.776，注入现场hit@20 0.421/3.34候选。原现场0.5阈值为0.342/2.98候选；相近少候选预算未超过v1。只剩field_ft权重，预训练PT缺失。 |

指标来自本地既有报告/输出，本轮未重新计算。不同划分、阈值、容差和监督定义不能混成统一排行榜。现场只有部分已知正位置，额外候选未核验；以上不是完整现场precision/recall/F1，也不是全新盲测。PINN2多项联合改动没有单项消融，不能据此证明物理loss单独有效。

## 现存9个权重：全部保留

| 相对本目录路径 | 用途 |
|---|---|
| FieldNN_Hybrid_20261022/output/hybrid_long/unet/best.pt | Hybrid U-Net |
| FieldNN_Hybrid_20261022/output/hybrid_long/transformer/best.pt | Hybrid Transformer |
| FieldNN_Iter5_20261022/output/iter5/unet/best.pt | Iter5 U-Net，后续多实验初始化依赖 |
| FieldNN_Iter5_20261022/output/iter5/transformer/best.pt | Iter5 Transformer |
| FieldNN_Iter8_20261022/output/iter8/unet/best.pt | Iter8 U-Net，Iter11初始化依赖 |
| FieldNN_Iter8_20261022/output/iter8/transformer/best.pt | 历史Transformer回退 |
| FieldNN_Iter12_20261023/output/final_delivery/best_model/inference.pt | Iter11-U-Net最终推理包；默认阈值0.94，可选0.60阈值文件同目录；不是训练恢复checkpoint |
| FieldNN_PINN_20261023/output/field_ft_150/best.pt | PINN v1最终微调模型 |
| FieldNN_PINN2_20261023/output/field_ft/best.pt | PINN v2最终微调模型 |

根 .gitignore 忽略PT/PTH/CKPT，因此这9个权重只在本地归档，普通commit不会备份它们。训练日志、逐步CSV、完整预测和图同样留在各实验本地output；Git保留源码、配置、报告及Evidence精选摘要。若要异机恢复模型，需单独复制本归档的权重和相关output。

## 复现条件和边界

1. Python：`E:\Anaconda\envs\gpushare_cu124\python.exe`；源码原有numpy/scipy/torch/matplotlib等依赖，本轮未变更环境。
2. 多实验共同依赖项目原位 `DifferentAgent/CodeX/Core16_Field` 与 `DG_Update/DG_V3`，这两部分不是失败NN模型归档，未移动或删除。
3. 仓库DataSet为空；真实训练缓存位于 `E:\CableNNDataSet` 的CodeX、Deepseek_CMDC/DeepSeek_CMDC、WorkBuddy子树，原始FDR在 `E:\20261010截至的全部fdr文件`。均未移动、删除或改写，不能只复制代码就声称包含训练数据。
4. 本轮修正执行代码中的迁移路径，保留模型算法和超参数；output内历史JSON/日志记录的原路径保留作为当时运行记录。阅读旧路径时按本索引映射到实验同名目录。
5. 许多历史权重在本轮开始时已经缺失。可以直接查看已存指标/预测/日志；但无法重放缺失checkpoint的原始推理，也无法保证只用最终推理包断点续训。禁止把保存源码等同于全部历史实验完全可复现。
6. 本轮不训练、不推理、不启动GUI，不修改GUI神经网络以外部分，不操作GPU配置，不自动提交。

## 清理状态

用户已执行候选清理，删除建议文档CLEANUP.md已移除。2026-10-24核对：61个候选目录均已不存在，CMDC/CodeX/Workbuddy三个空容器已由用户移除；15个实验目录和9个PT仍在。保留 `cleanup_candidates.json` 和根 `FieldNN_实验瘦身.ps1` 供回溯；脚本默认仅预览，已不存在的目标会跳过。正式模型、正式训练历史/日志、全部正式评价结果与E盘缓存保留。

## 本轮检查

已静态读取142个Python脚本、15个实验根层JSON和14个PowerShell脚本，语法解析通过；删除脚本仅预览，确认61目录/390文件/9.121 MiB。未执行模型导入、训练、推理、实验冒烟或GUI验证。语法与路径检查不等于完整训练链已验证。
