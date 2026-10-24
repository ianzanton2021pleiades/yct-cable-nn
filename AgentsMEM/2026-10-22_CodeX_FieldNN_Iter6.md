# 2026-10-22 CodeX Iter6 session

- User asks to audit CMDC Iter5 then improve U-Net/Transformer (optional third model), targeting actual field cables <=2.5km; no presentation/report deliverable.
- Authorized writes: DifferentAgent/CodeX new Iter6, E:/CableNNDataSet/CodeX datasets, AgentsMEM, milestone; raw E:/20261010截至的全部fdr文件 read-only. No GPU configuration changes, no git commit.
- Read AGENTS.md, stop-that-shit and officecli. Existing dirty working tree preserved.
- Two GPT-6 Luna Max agents auditing Iter5 and field evidence, read-only.
- Iter5 docs: wet-known hits remain 7/11 and 6/11, lower candidate density but engineer-note hit@20 falls to .421/.426; not demonstrated recall improvement.
- Asked user whether weak field-label adaptation is allowed with cable/case group holdout, and whether formal long training should be agent-run or user-run (prior session preferred user-run). Pending.

- User confirmed weak-label adaptation with cable/case holdout; formal long workflow still user-run. Agent only performs necessary short checks. Iter6 dir created with local ignores; 1500m wet group held out entirely. No third model unless evidence warrants it.

- Audits found no evidence for capacity/bandwidth/grid bottleneck; keep two architectures and change supervision. New train.py uses positive-only cross-sweep MIL (10m source/report-axis uncertainty), frozen Iter5 teacher soft preservation on unknown bins, synthetic replay every step, field every 2 steps, lr3e-5. Initial weights remain best eligible; validation uses only synthetic val + group-isolated field val. Formal defaults18epoch/B64/6workers; no full training started.

## Iter6交付与短检查

- 全流程脚本 run_iteration6.ps1 -Smoke 已实际退出0；smoke_output/iteration6_summary.json status=complete。缓存17条现场特征，train/val/test各2 bags。每模型仅1训练batch(B64)+1合成验证batch+两现场验证bags，随后2曲线val校准/test mixed/clean、1条原FDR推理及匹配评价，绝非效果评估。
- U-Net/Transformer grad_norm_last=20.477/17.055，AMP skipped均0，峰值allocated4.756/4.819GiB。冷启动含6worker创建的3.64/3.68 samples/s不是稳定吞吐。Transformer短步没有改善组合val，best保持初始权重(epoch=-1)，该路径也已走通校准/推理。
- CPU拓扑保留1物理核，实际mask0x3fff，7核参与，6loader workers+主线程1。未修改GPU功耗或驱动。
- 证据58条候选，39case/34原evidence cable groups/41源FDR；不是39个确认的独立真值案例。完整213manifest连接成51组，train31/val10/test10（122/35/56曲线），开发/夹具bank已接触组归train，wet1500强制test。全部已多轮开发，不能叫全盲。
- 发现并修复两个真实评价/数据风险：按source+适用sweep集合构袋，避免相别证据互套；group评分先逐bag一对一匹配再聚合，不把不同起点或湿位置条件合成一袋。窗口按证据engineer20m/TDR40m/wet10m，映射source velocity；未知区域不设negative。
- Git ignore已实际确认smoke_field_cache、smoke_output、smoke_eval_wet生成产物均忽略。README为技术使用说明，没有汇报文件；Milestone只追加本轮。未提交。
- 正式命令：pwsh -NoProfile -ExecutionPolicy Bypass -File "D:\GitRepository\Cable-NN\DifferentAgent\CodeX\FieldNN_Iter6_20261022\run_iteration6.ps1"。默认E:/CableNNDataSet/CodeX/FieldNN_Iter6_20261022现场缓存、旧Iter4合成数据只读、18epoch/B64/6workers，两模型顺序训练和评价；中断加-Resume。
- 正式结果主入口DifferentAgent/CodeX/FieldNN_Iter6_20261022/output/iter6/iteration6_summary.json。用户执行后读paired/field_comparison_{model}.json按固定.6/.99/.4比较保留组命中及候选密度，再决定Iter7；本轮尚未正式训练，不宣称现场增益。

- 修正后的现场evaluator已用正式Iter5 baseline真实11条wet记录CPU复评：两模型coverage11/11，@10/20/30命中分别7/11和6/11，候选总35/27、额外28/21，与旧结果一致。无不同湿位置条件合袋；该检查不是Iter6模型效果。

## 用户完成Iter6后的接续
- output/iter6 status=complete/full；U-Net训练6轮best第1轮，Transformer7轮best第2轮；AMP0跳步，稳定吞吐约293/286 synthetic样本每秒。mixed test事件F1 .8060->.8163 / .792->.8034。
- 固定.6/.99/.4现场测试组（42有证据sweeps，7有标签连接组）group macro@20 .4332->.6550 / .3773->.5907；候选密度明显增加。wet@10维持7/11、6/11，UNet额外28->49。尚不能据命中上升声称precision提升。
- 3个Luna分别审查现场结果、训练过拟合及val候选密度匹配阈值，第三者只CPU筛已有JSON置信度、不重推GPU。
- Iter7拟单因素改event未知区教师loss：全网格均值换为max(student,teacher)高响应top128 bins KL；不换模型/初值/数据/正例MIL，正式长训仍用户执行。

## Iter6候选密度匹配与Iter7决策

- CPU分析脚本Iter6/calibrate_field_density.py，仅按val选择threshold>=.6，事件候选数/适用曲线不高于Iter5 val密度，最大化val group macro hit@20；两模型阈值均.93。只过滤既有JSON的候选confidence，未修改正式thresholds或GPU重推。
- 非wet测试6组/40位置claim/31sweeps：U-Net组宏.399->.420、13/40->14/40，候选/条3.03->2.94；Transformer .349->.413、10/40->12/40，候选/条2.35->1.81。wet U7/11维持、T6/11->4/11。大部分固定.6命中增益随候选数量减少而消失，不能称可靠精度突破。
- 新Iter7单因素高响应教师约束已写train.py，从Iter5初始化以与Iter6对照；event未知区按max(student_p,teacher_p)选top128bins KL，非128独立事件，interval/terminal保持全网格软KL；系数/数据/结构/正例MIL/LR不变。复用Iter6缓存只读，不新增数据生成。
- Iter7 runner会自动完成与Iter5、Iter6固定阈值比较及val密度匹配分析。正式长任务仍用户执行；正在做必要短GPU检查（3 train batches，才能走到第二次field更新），不是正式训练。

## Iter7交付状态

- DifferentAgent/CodeX/FieldNN_Iter7_20261022/train.py、run_iteration7.py/.ps1及README已完成。数据只读复用Iter6现场缓存和Iter4合成数据；从Iter5 weights初始化进行单机制对照。不加第三模型，不复制/修改原始FDR，不改GPU配置，不提交。
- 两模型直接短训练均完成：B64，每模型3 train batches+1synthetic val batch+2field val bags，epochs1；未知区教师项实际非零，U-Net/Transformer mean KL .0010526/.0014510，最后grad norm21.505/20.601，AMP skipped均0，峰值4.7891/4.8641GiB。第一现场更新teacher为0，第三训练步的第二现场更新走通。没有正式Iter7长训，也不将该短查作为效果。
- 运行器--help通过，新smoke/output及Iter6密度分析产物已git check-ignore确认。完整runner复用Iter6已通的synthetic evaluator/field inference/evaluator，不重复全链路短查。
- 默认正式18epoch/B64/6workers、保留1物理核；每模型合成val校准和test mixed/clean、213现场推理固定点.6/.99/.4、与Iter5及Iter6三方比较，双模型完成后自动val密度匹配分析。单模型只输出固定点比较。
- 用户执行：pwsh -NoProfile -ExecutionPolicy Bypass -File "D:\GitRepository\Cable-NN\DifferentAgent\CodeX\FieldNN_Iter7_20261022\run_iteration7.ps1"；中断加-Resume。结果入口output/iter7/iteration7_summary.json与density_analysis.json。
- 后续优先读取density_analysis.json，不把固定.6候选增多当作模型成功；分开6个非wet保留组与wet1500。若候选预算相近仍无改善，应重新审视现场标签/特征与弱反射，而非继续增加同配方epoch。

## Iter7正式完成后的论文与Iter8接续

- 用户确认Iter7已运行完，特别要求分一个Luna Max通过本地HTTP API研究Zotero“线缆定位诊断-AI”“线缆诊断（无定位）-AI”PDF，以支持模型/数据/训练及可选第三模型。该调研已交audit_field，结果审计audit_iter5，输入多尺度CPU诊断audit_features。
- Iter7 output/iter7 status=complete/full。密度匹配阈值U=.87/T=.88（仅field val选）。非wet测试6组40claims：U16/40、macro .46183、候选3.258/条；T12/40、macro .390995、候选1.903/条。wet U7/11、T3/11。与Iter6相比并非两个模型同步提升，不能继续以延长此配方训练作为主方案。
- Zotero API本地127.0.0.1:23119可用；curl直连避免系统代理。集合主文献23+4=27，研究代理在读PDF原文，不仅题录。写缓存仅AgentsStorage/Zotero_FieldNN_20261022、简短技术依据AgentsDoc（不是组会汇报）。
- 当前考察多尺度局部对比/频段特征及真实校准短接作为无电缆事件样本；已向用户询问FDR CalibrationData是否确为夹具短接无被测电缆，未确认前不标成负例。

- 用户已明确确认FDR CalibrationData就是夹具短接、未接被测电缆。允许将其作为真实无被测电缆事件参考（不是健康电缆、不是远端terminal真值）。Iter8新增该监督，按源电缆connected group沿用Iter6划分，未标注电缆仍保持unknown。新目录FieldNN_Iter8_20261022已建.gitignore。

## Iter8交付与论文结论

- Zotero两集合23+4主条目，深入3篇PDF方法/实验，依据笔记AgentsDoc/2026-10-22_FieldNN_论文方法依据.md。缓存实际路径AgentsStorage/Zotero_FieldNN_Iter6_20261022/pdf_cache（先前指定不含Iter6目录未使用；不移动现缓存）。6FXGP6FN(CUTASAWU,10.1088/1361-6501/ae89d5)为1000频点复S11+f->100段Ωprofile，30cm微带/400k合成/FR4原型，不是现场电缆；B6L52JVD仅仿真DBN分段分类，TEV348DW是PD到时任务，不能挪用指标。无充分直接证据加第三模型，本轮仍两输入增强变体。
- Iter8保留原8特征，再加90/300m step detrend全局/局部RMS四通道；linear12->8 stem前8identity/新4zero，从Iter5加载backbone。stem lr3e-4，backbone3e-5。合成S11与raw cache不变。
- 新真实CalibrationData全轴raw，66来源(train39/val12/test15)，沿用Iter6连接组划分；wet1500源CalibrationData空，未补样本。用户已确认短接无被测电缆，d>=30 event/interval无测试电缆事件，<30及terminal未知，不冒充健康电缆。
- 正式目标数据E:/CableNNDataSet/CodeX/FieldNN_Iter8_20261022；新12ch field特征从原FDR只读重算；旧数据不改。训练每步synthetic，step0/2/4... field positive MIL .5 +8条calibration loss .5，取消冻结教师。cal loss top128有效event softplus +.25 interval mean；selection syn/base+.5weak/base+.5rawcal。
- 短检查 run_iteration8.ps1 -Smoke 已实际退出0，statuscomplete。17条12ch现场、6条校准小缓存；每模型3trainbatch B64、1synthetic valbatch、2fieldbags和1calvalbatch；2条synthetic val校准/test混合与clean、基线8ch/新12ch校准评价、各1条原FDR推理均通过。AMP跳步0，grad last22.241/20.428，峰值4.7824/4.8549GiB。7物理核mask0x3fff，保留1核。短测不证明效果，所选6条校准中baseline和新model均无伪事件，不声称已取得短接增益。
- 初版wrapper遇到旧prepfield硬编码8通道，改为本Iter8复制原冻结标签/split实现，仅shape/config/schema12与本地features预加载；已复查接口全链路，不改旧Iter6。所有新生成产物git check-ignore覆盖，未提交/改GPU/改GUI。
- 正式一键：pwsh -NoProfile -ExecutionPolicy Bypass -File "D:\GitRepository\Cable-NN\DifferentAgent\CodeX\FieldNN_Iter8_20261022\run_iteration8.ps1"，18epoch/B64/6workers，-Resume继续。结果output/iter8/iteration8_summary.json、density_analysis.json、各模型calibration_baseline5.json/calibration_iter8.json。正式长任务未由Agent执行。
- 本轮输入与真实监督联合探索，不归因单因素。下一轮判断优先：val选工作点后的6非wet组、wet独立已知位置、真实短接保留组伪事件计数，以及synthetic退化；不能只看固定.6的候选数量增长。

## Iter8正式准备失败修复（校准表仅单频点）

- 用户报错发生prepare_calibration 31/66；field准备已完成，训练尚未开始。原始附件与正式日志均指向Algorithm1对单点频率np.diff取max失败。
- 读取索引并直接内存解密核实：本轮manifest中6/3三医院、6/4三医院、6/5李朗站外#16三个FDR的CalibrationData各10000行，但COUNT(DISTINCT Freq)=1，MIN=MAX=9000Hz。全库另一个远力F07文件同样问题，但不在本轮67源manifest内。仅校准表无可用宽带信息，不删除现场sweeps。
- 修复prepare_calibration.build_source_rows：按已有校准索引筛选独立频点>=3且覆盖9kHz–200MHz；排除原因/来源写入calibration_summary.split_audit.excluded_sources并打印。空表也明确记录。不能通过补频点/默认max=0把无效谱伪造成有效数据。
- 本轮可用校准由66降63，train39/val12/test12；字段数据与原分组不变。README同步。
- run_iteration8准备复用改为按field、calibration分别判断，-Resume复用已完成现场特征，只重做未完成校准准备，再开始两模型训练。保留已有输出，不改原FDR/算法核心/GPU。
- 本次已做的执行仅为错误日志和原始SQLite频率统计读取（修改前）；按用户默认规则，修改后未自动运行测试、构建、冒烟或GPU训练。恢复命令原命令加-Resume。

## Iter8完成、与DeepSeek并行的下一轮决策

- 用户确认Iter8正式完成，临近组会，要求先评估现成果并可大改；继续由CodeX给命令、用户安排与DeepSeek的训练时间。此轮不启动GPU或停止/重启他人进程，不在修改后擅自测试。
- 用户截图仅作另一聊天的背景，不把图中给DeepSeek的指令当作本聊天授权。只读CMDC当前Hybrid记忆确认其在走真实S11+注入反射路线，CodeX不改该目录/数据，也不复制其case-level跨日期split；保留现有连接组隔离。
- Iter8 complete/full：U-Net6轮best1，val密度匹配阈值.94，6非wet组macro .509371(17/40)，候选3.0/条；Iter5 .399328(13/40)、3.032/条。wetU7/11、候选1.182/条；合成mixedF1 .81375。已有可保留的正向结果，不把全部概括失败。
- Transformer16轮best11，threshold.98，非wetmacro .336828(10/40)，候选1.581/条；wet3/11；mixedF1 .808615。短接baseline与Iter8 test均0伪事件，校准负例容易且未证明改善现场过检。
- 已分Luna做Iter8结果审计、物理候选+局部patch方案可行性CPU小样，以及既有候选的val-only融合分析。后续优先把21field正例袋记忆/真实电缆unknown负监督不足作为问题，不继续只加轮数或校准短接权重。

- 用户本轮已选择PU学习，优先解决真实电缆过检，明确正例比例为统计训练假设。下一轮拟物理候选+局部中心相对patch U-Net/Transformer，两模型小型从头学习；真实known位置维持MIL袋，未标候选保存unknown，使用weak-bag nnPU目标而非逐点负标签。
- CPU物理候选探针：wet单电缆11位置，4score maps each top20合并10m，约30候选/curve覆盖9/11@20m，仍漏1100/1300；只是proposal上限而非模型检出率。需报告coverage并且test不能强行插入标签位置。
- CPU候选融合val锁参后test@20退化，6非wetmacro .412 vsU8 .509，wet6/11 vsU8 7/11，未替换U8。原U8作为目前可保留模型；新路线保留相同group splits与candidate-density评估。

## Iter9交付准备

- 用户选PU后，已实现物理候选/局部双视野20x256patch、小U-Net/Transformer从头学习。synthetic pos/neg候选监督、field真实known位置MIL、未知电缆候选weak-bag nnPU(pi=.1假设)、真实短接负例BCE。P袋max近似有选择偏差，不能声称无偏风险保证；未知候选未写硬负标签。
- 直接用FIELD VAL已知位置macrohit20和真实短接伪事件选checkpoint/阈值，并以相同Iter5 U-Net val候选预算约束两模型；不再按单纯field-positive loss选权重。test不参与选择。候选缺失的labels继续留作miss，synthetic末端作为non-defect负候选，区间/末端本轮不训练。
- 生成Image/20261022_FieldNN_Iter8_progress.png，来自既有I5/6/7/8正式JSON，仅CPU作图不是训练/模型验证；图中U8+11.0pp并且候选3.03->3.00，不将partial known-positive当precision/recall。
- Iter9/proposals.py用find_peaks避免平坦零图造峰，跨map按各图相对强度排序；prepare整曲feature/proposal/patch任务按workers*4分块在worker执行，主线程仅落盘，避免9.6k未来缓存全曲积压。真实源只读，完整213原分组保留，Iter8已缓存138曲复用，余75重算。
- 只完成静态接口核对。没有运行Iter9脚本、数据准备、测试、构建、lint、GPU或smoke；用户当前未授权自动修改后检查，GPU还由其与DeepSeek安排。不停止他人任务，不改CMDC。新增可PrepareOnly只做CPU数据与proposal覆盖，再-Resume正式GPU。

## Iter9当前交付状态

- 已完成proposals.py/models.py/prepare.py/train.py/evaluate.py/audit_proposals.py/run_iteration9.py/.ps1及README/CONTRACT；仅静态阅读关键模型通道、数据索引/源坐标、checkpoint字段与命令参数。所有子代理确认未运行测试/GPU/cache。主代理也未运行Iter9任何脚本；本轮唯一运行的新脚本是已有Iter8结果CPU作图，非模型验证。
- 两phase用户命令：先run_iteration9.ps1 -PrepareOnly（CPU生成cache与proposal覆盖报告），显卡空闲后同脚本-Resume；直接无参数也可全链路。默认35ep/B512patch/6workers/pi.1，7物理核mask helper；max6stale早停。新数据E:/CableNNDataSet/CodeX/FieldNN_Iter9_20261022，结果output/iter9/iteration9_summary.json及每model/evaluation报告。
- 全字段契约：synthetic_{split}_patches/labels.npy，field/calibration_patches.npy，candidate JSONL(index,curve_index,distance_m SOURCE)，records连续curve index/原splitgroup，field_bags重新映射保留全部truth，positive_bags每position source±20m，缺proposal项保留。模型输入20x256输出B event logits，checkpoint包含config.model和validation.threshold。
- checkpoint/阈值只按val macrohit20在统一U5候选预算内选择，calFP每reference惩罚.05，syn BCE tie break；包含有限reject-all动作1.000001以使约束策略可行并明确标注弃权，不当概率。模型测试前锁阈值，测试报告分非wet/wet与unknown；synthetic patch分类不是旧整曲三头F1。
- 当前成果图Image/20261022_FieldNN_Iter8_progress.png已经生成并读取：U-Net known-position macro39.9->50.9%，候选3.03->3.00；T没有对应进步，不能把整体总结成完全失败或宣称现场可用。融合CPU验证-selected后测试退化，已拒绝替换U8。共用U5预算重新读T8依然选择.98，结论未变。
- 当前不可声称：Iter9可运行已验证、实际显存/吞吐、训练完成或效果改善；程序尚未运行。用户已明确模型运行时段自己安排；不得因旧授权而自行启动GPU/测试或干扰CMDC。
