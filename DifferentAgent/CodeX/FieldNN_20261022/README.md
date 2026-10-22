# FieldNN 首轮试验

**最新决定：用户明确先冻结生成器和既有10000条数据，授权直接运行Transformer/U-Net各5epoch，Mamba搁置。该基线保留[S11域差检查](S11_REVIEW.md)的局限。当前运行目录为output/tu_baseline_20261022，进度和精确续接命令见[HANDOFF_TRANSFORMER_UNET.md](HANDOFF_TRANSFORMER_UNET.md)。下文pilot_v1/all为先前通用入口，不用于此次两模型运行。**

统一输入是距离域四通道：归一化脉冲响应、归一化阶跃响应、脉冲响应 logRMS、阶跃响应 logRMS。输出是三个 logits 通道：event 点、异常区间、terminal 点。当前比较纯 PyTorch Mamba、Transformer 和 1D U-Net。

当前频率配置为 32,768 点、9 kHz–200 MHz 对数网格。它不精确复现每台现场仪器的原生网格（例如 50,000 点或覆盖至 1 GHz）；插值只映射已有采样，不增加测量信息。生成标签代表合成参数变化区间及几何边界，不证明这些变化在实测频带和噪声下可检测。现场未知标签仍是未知，不能当正常样本，也不能据此报告现场准确率。

先单独运行 Generate，再运行 Audit 并审阅 JSON 和 PNG；通过完整性审计不代表生成数据已具备现场真实性。目前领域一致性尚未建立，实测频带对照也只是不同统计粒度下的描述，建议审阅后再决定是否训练。第一轮 5 epoch 只看损失趋势和流程，不代表最终质量。先完成单相评估，之后再决定是否做三相融合。

方法参考：[Duarte 等，2026，局放到达时间论文](../../../REF/论文/Duarte 等 - 2026 - A deep learning approach for end-to-end time-of-arrival estimation and its application to partial di.pdf)。可参考其软化逐点监督、峰解码、验证集选阈值和容差匹配；该论文处理的是有标注的局放时域波形，不是 S11 生成或无标签自监督，PD 到达时刻点也不能直接等同 FDR 异常区间。

## 启动阶段

PowerShell 7 中一次启动一个阶段，没有自动串行运行全部阶段的选项。Python 固定使用 E:\Anaconda\envs\gpushare_cu124\python.exe；默认数据目录为 E:\CableNNDataSet\CodeX\FieldNN_20261022。

```powershell
$Experiment = 'D:\GitRepository\Cable-NN\DifferentAgent\CodeX\FieldNN_20261022'
pwsh -NoProfile -File "$Experiment\run_pilot.ps1" -Stage Generate
pwsh -NoProfile -File "$Experiment\run_pilot.ps1" -Stage Audit
# 审阅完成后再继续以下阶段
pwsh -NoProfile -File "$Experiment\run_pilot.ps1" -Stage Train -Model all -Epochs 5
pwsh -NoProfile -File "$Experiment\run_pilot.ps1" -Stage Evaluate -Model all
pwsh -NoProfile -File "$Experiment\run_pilot.ps1" -Stage Field -Model all
```

- Generate 用当前配置生成或续补分片，并带 --resume。
- Audit 最多抽查 128 条，均匀覆盖 train/val/test，报告写入 output\audit。
- Train 支持 all、mamba、transformer、unet。模型、checkpoint 和训练历史写入 output\pilot_v1\<模型名>。
- Evaluate 对每个模型先用 val 校准阈值，再评估 test。
- Field 先生成 real manifest，再只对 development 子集运行推理。未确认集合不用于调参。

训练续跑须显式加 -Resume，并把 -Epochs 提高到目标总 epoch 数。例如：

    pwsh -NoProfile -File "$Experiment\run_pilot.ps1" -Stage Train -Model mamba -Epochs 20 -Resume

训练会恢复 ReduceLROnPlateau 调度器和 checkpoint 状态；Epochs 表示总目标，不是续跑的新增轮数。要延长训练，应先审阅 Audit 和已有损失趋势。

## 本轮交付边界

正式配置为10,000条，train/val/test为8,000/1,000/1,000，长度100–2500m；预计含原始S11、输入、标签及truth约5.2GB。首版以开路端接为主，未覆盖接设备和短路端接。固定输入域0–3250m不代表把研究范围扩大到3250m。

已完成20条最终对数网格生成与三模型微型训练—验证—测试检查、U-Net恢复训练和一条现场FDR从S11起始推理。正式数据和长训练尚未启动，微型checkpoint都在ignored的output/smoke_*下，仅供检查。原GUI和原始实测文件未修改。

Mamba包含输入相关选择性状态空间、卷积与门控，使用纯PyTorch扫描，不是官方融合CUDA内核。batch=4、AMP下2步预热加3步短测，Mamba约17条/s，Transformer/U-Net约217–221条/s；这是微型训练吞吐，不能据此保证整轮耗时。

后续调优、三相融合和优秀模型GUI接入以正式单相结果为前提。续接计划见本目录PLAN.md；生成图的对称对数显示只为展示强近端/弱远端，模型输入保持原线性响应。
