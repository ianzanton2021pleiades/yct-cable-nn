# CodeX FieldNN 现场迭代 Session

- 日期：2026-10-22，Asia/Shanghai；Agent：CodeX；目录 D:/GitRepository/Cable-NN。
- 用户目标：接续午间 DeepSeek/CMDC，优先 U-Net 与 Transformer 的现场可用性和减少过检；明日组会，不编写汇报文件。
- 授权：新实验 DifferentAgent/CodeX，新数据 E:/CableNNDataSet/CodeX；现场原数据只读；不改显卡配置、不提交。允许 GPT-6 Luna Max 子代理，当前3个只读审计。
- 本轮已读 AGENTS.md、stop-that-shit、officecli、今日 CodeX/CommandCode 项目记忆与交接。保留既有工作区修改。
- 当前无本项目训练进程；DeepSeek最新 pilot_v3 仅见 U-Net checkpoint。
- 确认 DeepSeek v3 数据是原10k的raw impulse/step派生，并未重新生成S11；8通道增益/去趋势/局部归一化，width48/depth3。
- 关键待决：旧loss将末端+10m以后全mask，推理却在0–3250m全域解码，后半区域缺少负样本惩罚；需量化。Loss尺度变更不能据数值判定优化器失效。
- 已向用户询问现场precision/recall取舍、长任务直接执行方式，等答；独立审计继续。

## 本轮执行决策与修正

- 用户回答：精度优先；正式一轮脚本由用户自己执行，CodeX只做必要短检查。当前3个Luna子代理均有明确文件责任，不开启用户侧新chat。
- 新实验：DifferentAgent/CodeX/FieldNN_Iter3_20261022；新数据拟写 E:/CableNNDataSet/CodeX/FieldNN_Iter3_20261022；旧CodeX/CMDC实验和原数据不改。
- 已定位：CMDC配置10通道但features_v3仅8；旧U-Net约43m局部上下文；旧loss全mask真实末端后；材料179.325Mm/s与距离轴172Mm/s导致2.5km约102m系统偏移；原10k单例高损耗anchor泛化不足。
- 已实现：7尺度真实层级编码器，U-Net空洞/全局瓶颈、Transformer128token瓶颈；保留符号和幅值尺度的8通道；全域Gaussian focal/BCE+Dice，记录逐头损失、梯度与AMP跳步；精度目标.85的val校准(达不到时F0.5)，健康误报与末端后误报统计；现场不以真长裁剪、不设top-k；213条只作覆盖，1500m实验只计已知正例命中与额外未验证候选。
- 新生成器在修正为基于分段传播核的带限响应位置；几何真值单独保留，可观测性属于合成弱监督而非现场gold；不采用固定100MHz群时延冒充宽带峰位。
- CPU模型forward/backward和单batch loss已由子代理检查；正式端到端短检查尚待新生成器完毕，不能将旧smoke_train_cpu checkpoint作为最终loss结果。
- 运行器已短查 Windows核心拓扑：8物理核/16逻辑核，排除最后1物理核后mask0x3fff；子进程继承，生成7worker/训练6worker+主线程1，val0worker，单GPU顺序模型，不改显卡设置。

## 最终短链路进展

- generate/config 已冻结：seed2026102203；最终20条在smoke_data_v2，16/2/2，耗时5.3s、2workers；6个healthy，42/63事件可观测、17/20末端可观测。ignored event/interval/terminal=1.137%/1.229%/0.185%；alpha100最大残差1.39e-17dB/m；physical max|S11|=.98395。只证明微型计算一致性，不代表实测拟合。
- 旧smoke_data和E:/CableNNDataSet/CodeX/FieldNN_Iter3_20261022_smoke为fixture换算前旧输出，保留未删，不用于正式训练。
- run_iteration.ps1 -Smoke -SkipGenerate B16全链路完成：两模型训练、val阈值、test、每模型1条原始FDR推理、JSON/图写入正常，产物smoke_output；这些不是效果证据。
- GPU短链路抓到真实优化数值问题：U-Net在默认GradScaler65536下第一步grad=inf、AMP跳步1；将train.py初始scale改1024后只重查两模型1训练batch+1val，grad finite(227.08/153.77)、跳步均0，val损失有限。新检查在output/smoke_scale1024，旧smoke_output checkpoint早于scale修正，仅作接口链路证据。
- B16峰值显存1.215/1.235GiB，包含进程冷启动的5–6samples/s不能作稳定吞吐；已委派Luna做2warm+3timed训练step的16/32/64 device-only对照，随后据结果选择正式batch。

## 本轮交付状态

- batch探测完成：2warm+3timed，B32 U-Net333.30/Transformer330.42 samples/s、峰值allocated2.41/2.46GiB；B64为332.29/323.79、4.77/4.84GiB，全部AMP skipped0。选择B32，已同步config/train/runner/PowerShell/README默认。设备短测不含DataLoader/H2D，不当完整吞吐；B16 U-Net首配置冷态读数不用于对比宣传。
- 最终额外核心检查：从scale1024 checkpoint用--resume恢复至epoch2，每模型只1train+1val batch（smoke train仅16样本，CLI B32实际余数16）。U-Net/Transformer grad norm74.87/66.50、AMP skipped均0，lr如期从1e-4恢复warmup至2e-4。checkpoint optimizer/scaler/scheduler/RNG恢复路径已实际走过。
- 正式启动：pwsh -NoProfile -ExecutionPolicy Bypass -File D:/GitRepository/Cable-NN/DifferentAgent/CodeX/FieldNN_Iter3_20261022/run_iteration.ps1；默认40epoch/B32/6训练workers，生成7workers，保留1物理核，顺序训练两模型，再val/test/213条现场(含去重11正例)评价。
- 正式数据尚未生成、正式模型尚未训练；不宣称现场性能提升。用户执行后CodeX直接读output/iteration3日志/iteration_summary.json继续判断。
- 输出ignore已用git check-ignore确认；E盘可用44.5GiB，12k数据约7.5GB量级足够。没有提交、更改GPU配置或修改旧CodeX/CMDC核心源码；Milestone仅追加本轮，AgentsMEM只维护本session文件。
- 全部子代理已结束，无本轮训练进程需继续后台运行。

## 用户正式运行完成后的评估（2026-10-22，下一轮继续）

- 用户已跑完Iteration3，继续授权模型/必要数据迭代；精度优先及正式长任务用户执行约定继续，不写组会报告。
- 完整12000样本，test1200；两模型213/213实测处理。U-Net best checkpoint epoch14(1-based15)、实际27epoch；Transformer best23(1-based24)、实际36epoch，早停。
- Test event P/R/F1@5m：U-Net .8949/.8533/.8736，Transformer .8814/.8945/.8879；interval F1 .8799/.9002，terminal F1 .9976/.9990（仅1034/1200可观测terminal真值，弱监督窗口忽略约.9%event/.9%interval/.17%terminal）。不同标签/数据版本不能直接与旧test作提升差值。
- 健康374曲线每条event误报.01337/.00802；末端后event0/2。训练最后train loss .00518/.00165而val .29398/.53060，已有明显泛化gap，不靠延长原样训练解决现场。
- 现场wet1500十一已知正例hit@10m=1/0，@20m=1/1，@30m=2/2；候选少不等于现场检测成功。
- 本轮委派3个Luna Max：training收敛/阈值审计；field同31开发对比和阳性定位；physics真实/合成带宽、幅值、可观测性弱标审计。只读CPU小样，未启动新正式训练。

## Iter4决策与评估轴修正

- 用户选择先增强和夹具适应，保留实测电缆整组评价；仅在这条路线失败后才允许少量现场弱标签训练。本轮不使用真实线缆波形的标签训练。
- 用户要求先估计长度，按约1.2L范围重算/绘图；同时要求参考更多现场.fdr，不能只评1500m。新增独立REF纯算法末端视图，不裁模型输入/候选，不把启发式长度当实际真值。
- 发现上一轮field评价轴错误：wet源FDR Wave_Velocity=200m/us，模型172m/us，原直接比较导致近端检出被误记漏检。field_eval现保留prediction_nominal并按source/report波速映射prediction；source setting不是独立真实材料波速。
- 两模型all213已按source/report轴重推理至output/iteration3/{model}/field_source_axis，均213/213；wet已知正例hit@10m为U-Net3/11、Transformer5/11，@20/30m分别4/11、5/11。只是评价修正，不是模型增益。
- 同31开发曲线候选旧→Iter3：UNet events534→90(中位12→3)、interval410→68(中位12→2)；Transformer events668→145(中位20→4)、interval235→67(中位7→2)。unknown无完整truth，不把候选数当FP。绝大多数候选集中在前500m；远端检测仍弱。
- 真实宽度证据：白塔寺原始Algorithm1/172轴末端pulse FWHM56.3m，阶跃10–90宽79m、上升.210；同长合成kernel4.36m、step宽25.8m/升.813；Iter3正标上限event20m/terminal40m排除了很多宽主瓣。全量35,740物理事件中10,099被忽略，其中5,615宽度超限；不可观测terminal1,512几乎全因40m cap。
- 新Iter4：复用12k S11转raw response缓存，35%clean回放，余下在线同源展宽/动态范围/真实calibration-short近端扰动；平滑sigma<=min(40m,可见中心最小间距/3)，避免把近邻两事件合并后仍强监督。先展宽缆响应再加sharp近端扰动，符合尖首端+宽弱远端。
- Iter4训练从各best.pt weights-only warmstart，重置optimizer/scaler/scheduler/RNG，lr1e-4/最多24epoch/B32/6workers，不改架构。val/test确定性mixed增强与clean对照，基线同增强评价，test阈值冻结。
- 真实short bank6个来源仅development_only CalibrationData，无现场电缆标签；v2先减稳态step平台、taper局部step再差分恢复impulse，防止同步taper造成imp/step不一致。正式新E盘缓存尚未启动。
- 长度模块复用REF AST纯函数并Algorithm1全域/1.2L重算；初版缩Dmax改变FFT步长造成DC漂移，正在固定重算使用full fft_step_hz；旧field_source_axis的模型候选/命中有效，初版重算蓝线图不作为最终交付。

用户额度将尽要求尽快脚本和交接，已停止扩展。Iter4完整60条缓存短链路已退出0，两个模型1batch更新无AMP跳步，mixed/clean评价与FDR入口连通；正式默认24轮/B32/6worker，启动run_adaptation.ps1。HANDOFF.md已写，旧现场蓝线重算FFT问题代码已固定。
