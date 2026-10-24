# CodeX 组会成果交付：压缩接续摘要

- 日期：2026-10-22 Asia/Shanghai；项目 D:/GitRepository/Cable-NN；本文件是当前根会话压缩锚点。历史完整细节见 AgentsMEM/2026-10-22_CodeX_FieldNN_Iter6.md。不要依赖超长对话逐轮重读。
- 用户最新目标：Iter9已完成，组会迫近。希望真实现场FDR（禁用RG58）train/val accuracy及loss曲线，横轴optimizer steps，足够50–100记录点或epochs（待确认）；合并参考两表的综合指标表；2–4个良好现场案例识别图；选择最佳模型并写中文模型文档。参考图片只是样式，不能抄其99–100%、均值±std或p值。
- 用户希望先压缩上下文。当前工具清单无compact/compress接口；仅保存摘要不等于系统压缩，不能虚称已执行系统压缩。尚未启动新一轮训练或GPU。
- 用户再次授权GPT-6 Luna Max子代理，<=4并发（当前系统槽位含root最多4，即可用3个worker），root只做关键决策。默认沿用给命令用户执行GPU/长任务，与DeepSeek并行排队；不要停进程或动CMDC。新实验DifferentAgent/CodeX，数据E:/CableNNDataSet/CodeX；原FDR E:/20261010截至的全部fdr文件只读。
- AGENTS：pwsh UTF8、gpushare_cu124解释器E:/Anaconda/envs/gpushare_cu124/python.exe；不改GPU配置/非NN GUI、不提交、不写C盘；新增实验.gitignore标准output/assets/s11_output/__pycache__/*.pyc/smoke_*/。当前请求未明示自动检查/测试，修改后不能擅跑测试、构建、smoke/GPU；数据分析/已有结果读取和交付图生成可做。

## 当前方法与数据

- <=2.5km现场事件定位，不做缺陷分类；U-Net+Transformer最多加1其他模型。Field full213 sweeps/67 FDR/51连接组，train122、val35、test56曲线；仅138条有可用局部位置证据。字段和台账在Iter8 E盘cache内；原connected split始终保持，现场数据已多轮开发，不能称全新盲测。
- 有位置的测试是42sweeps：非wet31条、6连接组、40个bag-position claims；wet是单一1500m电缆实验11个已知位置条件。标注只有局部正例，未知不等于负例；没有完整现场TN/FP真值。Accuracy/MacroSensitivity/MacroF1不能直接计算成全现场指标。2–4好案例可选保留组成功例，但应记录选择规则并保留整体指标，不冒充代表全库。
- 用户已确认CalibrationData=夹具短接、无被测电缆。有效63源，39/12/12 split；三个10000行全9000Hz单频表和一个空表已剔除，不删现场sweeps。其30m外可以真负监督，terminal未知；不是健康电缆。I5与I8在校准test均0伪事件，不能拿这当真实电缆精度改善。
- 用户上一轮已明确接受weak-bag PU：pi=.1只是训练统计假设，P以known-position袋max近似，有选择偏差，不宣称标准nnPU无偏风险；真实电缆未知候选不写硬负标签。

## 最佳既有对照

- Iter8 U-Net：val选theta=.94，非wet6组macro known-hit@20 .509371、17/40，3.00候选/curve；I5 U .399328、13/40、3.032候选/curve。wet7/11。T8 .336828、10/40、1.58cand，wet3/11。
- 正式结果DifferentAgent/CodeX/FieldNN_Iter8_20261022/output/iter8/iteration8_summary.json及density_analysis.json。
- 当前成果图Image/20261022_FieldNN_Iter8_progress.png，生成脚本Iter8/plot_iteration_progress.py；对比I5/6/7/8同val预算，不是field precision/recall。
- I8 U+T空间候选融合已做CPU val锁参分析，test退化，未采用；Iter8/output/fusion_analysis有结果。

## Iter9 已完成，待完整评估

- 目录DifferentAgent/CodeX/FieldNN_Iter9_20261022；数据E:/CableNNDataSet/CodeX/FieldNN_Iter9_20261022。
- physical四map各top20/find_peaks，合并10m，nominal30..3240；双视野+-50/200m、20x256patch，去绝对d，局部U-Net369569参数/Transformer79553参数，从头训练。
- synthetic58700左右PN候选＋field122训练曲/5930 U候选/80有proposal正例袋/6缺proposal袋＋2147校准负patch。synBCE+.5MIL+weaknnPU+.25calBCE。默认B512，6workers，AdamW3e-4，AMP1024，clip1，earlystop6。
- 阈值与模型权重按field VAL macrohit20、统一U5 val候选预算2.548/curve、calFP惩罚.05选择；synthetic BCE tie-break。候选缺失标签留作miss。阈值网格.05.. .99加1.000001显式reject-all。
- output/iter9/iteration9_summary.json statuscomplete/full，U训练14epoch、best8：val score/macro .705556、theta.99、2.548cand、cal0、synValBCE .25154；T训练19、best13：val .694444、theta.975、1.774cand、cal0、synValBCE .27818。两模型AMP skip均0。
- U末轮14因grid在>.99概率无法继续细分而选rejectall，score0；这可能包含置信度饱和/阈值网格问题，不应未经分析断言全是定位退化。后续可评估raw logit/quantile校准（仅val），不能test调参。
- 当前每epoch仅history一行。Ns58685/B512=115steps/epoch（默认drop_last=False），约U1610/T2185steps。不能凭空恢复历史逐step accuracy；可用真实epoch点横轴换成累计step，或新跑记录50–100个真实step检查点。train total_loss包含MIL/PU/cal，val_synthetic_bce不能直接配成同一种train/val loss曲线。
- field/evaluation/field_report.json、proposal_coverage.json、calibration_report.json、synthetic_patch_report.json和field_comparison.json需要下一阶段读取，不能将synthetic候选分类Accuracy伪装现场Accuracy。

## 待用户确认的三项口径

1. 继续现场定位，以known-positive hit/candidate burden/loss为主，对不可定义的Accuracy等注明缺完整标签；或用户能补建正负完整标注现场候选集，才做Accuracy/Sensitivity/MacroF1。
2. 50–100真实日志点/optimizer steps即可，仍依val保留best；还是希望固定50–100完整epochs，需保留真实退化曲线且不能保证无过拟合。
3. 组会单次实测结果N=1（无训练std/p值）；还是3seed重复训练，接受时间成本再做可支持的统计。

下一阶段应先收齐回答，读取Iter9现场指标，冻结validation选定模型与报告口径。可派3个Luna：结果/病例审计、曲线日志/校准改进、表/模型文档与交付脚本。不要继续盲目开新模型；主要交付真实曲线、综合表、2–4现场案例、模型文档与可复现命令。用户允许大改，但不能为好看更改标签/泄漏测试/删差点/编造100%。

## 用户确认（最新）

三项已回答：1)沿用现场定位任务，曲线用known-positive hit及可比loss，分类指标缺失注明；2)需要50–100个真实记录点，横轴optimizer steps，不强制50–100epoch，按val选best；3)单次实测N=1，未支持std/p值注明未测。后续无需重复问这些问题。继续交付完整图表/2–4现场案例/中文模型文档。当前工具无法手动系统压缩，只保存此摘要；可基于本文件直接续作，不虚称压缩已发生。
