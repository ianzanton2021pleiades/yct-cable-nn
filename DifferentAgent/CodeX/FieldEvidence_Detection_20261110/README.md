# 现场证据重建与检测实验

## 最近一次开发试运行：2026-11-11

`output/development_20261111_145901`三折训练与评价已完成，均9次更新、最佳第1轮。留出A02时训练区间缺背景负例，留出A03时缺完整区间正例，留出A04时缺异常点正例；当前三折不能用于网络选型或单/多相收益比较，不继续重复该划分。

分析见[开发试运行结果](review/20261111_开发试运行结果.md)。下一步先补齐不同物理案例的任务正负监督，再确定新的训练/验证划分。后文一键入口保留为本次运行的复现记录。

启动日期：2026-11-10。主导：GPT-6.1 Sol（CodeX）；资料整理与实现：GPT-6 Luna Max。

## 目标与执行顺序

1. 从已有FDR配套文字整理首批代表案例，关联测量、相位、物理案例与证据原文。
2. 修正证据到标签的转换，保留点、范围、单侧边界、未知与来源可靠性。
3. 在相同现场案例划分下，比较从头训练与合成预训练后的现场训练。
4. 按现场分项表现确定DG、输入或物理反演的下一项工作。

当前已完成24个候选案例的原文整理，并交付准备数据、训练和评价接口。measurement index关联31个FDR与120条Sweeps记录，其中115条标记实测；10条实测测次同时缺少厂商脉冲和阶跃表。当前accepted_labels只含A02、A03、A04三个物理案例；B06区间保持unknown，未作负标签。正式train/val划分仍为空，尚未形成训练集或验证集。

## 文件入口

- `evidence/`：首批来源摘录、测量关联和案例关系。
- `review/`：需裁定的问题、标签管线追踪与阶段交接。
- `evidence/batch_a.md`、`evidence/batch_b.md`：24个案例的简表；同名JSON保存逐条原文与来源。
- `review/标签草案_A.json`、`review/标签草案_B.json`：按原文精度整理的候选位置与范围，尚未作为训练输入。
- `review/comparison_groups.json`：4个已明确比较组；物理案例case_group与同条件输入comparison_group分开。
- `review/accepted_labels.json`：当前3个案例的逐测次监督；B06保持未知且未写成负样本。
- `review/标签与训练接入方案.md`：四任务标签、独立审核范围、测次ID、案例分组与现场训练对照契约草案。
- `scripts/prepare_field.py`：按比较组准备五通道现场NPZ与manifest，默认写入`output/field_cache/`。
- `dataset.py`、`model.py`、`losses.py`：首轮数据、共享相位模型和masked loss接口。
- `train.py`、`evaluate.py`、`config/first_round.json`、`config/split_template.json`：用户执行的训练与评价入口及空白正式拆分模板。
- `run_prepare.ps1`、`run_field_training.ps1`：PowerShell 7命令包装器，默认环境`gpushare_cu124`。
- `output/review_packet/index.html`：完整离线审阅包，含原文、真实测次和厂商曲线。
- `output/review_packet/focus_review.html`：A02、A03、A04、B06四例局部审阅页，原文候选位置单独标出。
- `output/review_packet/measurement_index.json`：实际读取的测次、频段、校准表与响应表覆盖。
- `scripts/build_review_packet.py`：完整审阅包生成脚本，复用现有FDR读取器。
- `scripts/build_focus_review.py`：四例局部审阅页生成脚本。
- 后续代码、配置放在本实验目录；可再生成输出放在忽略目录。

原始现场目录保持只读。原有实验与缓存保持原状。未获得本轮明确检查指令时，不自动运行测试、构建、lint或推理；长时间生成和训练交付脚本由用户执行。

## 协作与决策

最多3个GPT-6 Luna Max并发。主agent负责证据强度、任务与评价定义、跨任务集成。

用户问答最多4轮，每轮最多4问。第1轮3问已确认：

- 首轮验收FDR中可辨的结构与异常，现场物理事实另存。
- 位置窗口、单侧边界、不确定区间按原证据精度保留。
- 内部接头与异常为重点，末端单独验收。

采用stop-that-shit技能控制工作范围；不把完整标注平台、新DG或额外测试体系作为前置条件。

## 当前交接

原文证据、测次索引、标签草案和已接受标签分别保存。第2步已提供同次同条件三相成组接口，保留各相绝对响应与相间差异；单相组以P=1运行同一准备和模型接口，不复制成虚假三相。缺相与缺响应表分别mask。

用户已完成4组现场cache准备及A臂三案例开发试运行。该试运行存在前述任务覆盖缺失，正式划分仍需重定；`config/split_template.json`保持空白。当前不要重复运行旧三折入口。训练输出默认到已忽略的`output/field_training/`，准备数据默认到`output/field_cache/`。B臂需要兼容的共享encoder_state_dict，通过`-EncoderInit`提供；现有合成数据接入还需解决响应截断带来的长度泄漏，见`review/合成编码器预训练接入.md`。A/B后续采用相同seed、数据顺序、相采样和验证选模/早停规则，并记录actual_optimizer_updates与epochs_completed。`best.pt`按最低masked validation loss选择。长时间准备与GPU任务由用户执行。

阅读版处理分析见`../FDR_Reader_Analysis_20261110/README.md`；平滑实现位于该分析目录的`scripts/reader_preprocess.py`。

## 单相比较组

单相数据使用一个明确成员，manifest生成P=1；不要把同FDR的其他测次复制成B/C相：

~~~json
{
  "comparison_group_id": "comparison-single-example",
  "case_id": "example",
  "case_group_id": "physical-case-example",
  "source_path": "E:\\field\\sample.fdr",
  "members": {
    "single": "E:\\field\\sample.fdr#Sweeps.rowid=7"
  },
  "phase_names": {
    "single": "A相"
  },
  "member_roles": {
    "single": "labelled"
  },
  "coordinate_alignment_basis": "厂商原始距离轴"
}
~~~

## 准备、训练与评价命令

在PowerShell 7中执行：

~~~powershell
.\run_prepare.ps1
.\run_field_training.ps1 -Arm A
conda run -n gpushare_cu124 python .\evaluate.py --checkpoint .\output\field_training\A\best.pt --data-dir .\output\field_cache
~~~

B臂只预训练共享encoder，其余部分与A臂使用相同seed初始化。准备好兼容的encoder checkpoint后再执行：

~~~powershell
.\run_field_training.ps1 -Arm B -EncoderInit 'D:\path\to\encoder.pt'
~~~

B臂权重文件应为包含`encoder_state_dict`键的PyTorch checkpoint，该键值为共享encoder的state_dict；不尝试旧权重格式回退。

旧DG_V5缓存的valid_len按1.3×标注实长+10m生成且已截短，不能直接作为公平B臂输入。正式比较还需用S11重建固定显示窗口，并复核事件轴和可观测范围；这项数据准备不阻塞A臂代码交付。

评价分别输出完整相位上下文和各单相视图。required_measurement_ids不完整的标签计入not_scoreable；未知候选数和显式负范围内误报分开列出。审核范围不完整时不报告precision或F1。joint/terminal当前无已接受正标签，评价标为not_scored。

本轮旧版单相coverage分母包含未输入的其他相事件，单相coverage偏低。更新后的评价只统计所选measurement_id标签，所需上下文缺失仍记为not_scoreable；本轮已报告的0召回与0 IoU结论不变，修正从用户下次运行evaluate.py起生效。

## 邦润C相派生响应

原件C相（Sweeps.rowid=3）保存完整复S11，缺厂商脉冲/阶跃表。阅读版自动打开文件不会补表；其手动“重新计算”调用厂商原生算法并写回当前FDR。

已复制原件到 `output/derived_fdr/邦润线_C相待重算.fdr`。后续仅在阅读版打开这个副本，选中明确名为“C相”的测次，按Ctrl+R执行重新计算，保留其原有设置；完成后由数据准备流程读取派生表。不对“新的测试_4”猜相位，不把副本视为独立物理案例。当前副本尚未重算。

本实验不创建记忆文件，不自动提交Git。

## 三案例开发性试运行

准备数据后，双击 `开始开发试运行.cmd`，或在PowerShell 7中运行 `pwsh -ExecutionPolicy Bypass -File .\run_development.ps1`。脚本依次训练并评价A02、A03、A04三个已确认监督案例的leave-one-case-out划分，默认40个epoch；可用 `-Epochs` 覆盖。每轮输出到新的 `output/development_<时间戳>/`，保留控制台日志、各折配置与拆分、checkpoint、history、evaluation及汇总文件。

该运行属于开发性试运行，当前案例覆盖不足以作为正式泛化验收；仅运行A臂，B06仍为未知，B臂未运行。
