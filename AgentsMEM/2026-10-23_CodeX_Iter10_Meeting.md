# CodeX Iter10：组会成果与真实曲线

- Session: root，2026-10-23 Asia/Shanghai；接续 2026-10-22_CodeX_Meeting_Delivery_Handoff.md，用户已手动压缩。
- 用户确认：现场用 known-positive hit 和同口径 loss；50–100真实记录点、optimizer steps横轴；单次N=1，std/p未测。GPU和长任务由用户执行。
- 当前I8 U-Net是已知最佳保留结果：非wet 17/40、6组macro .509371、3候选/curve，wet7/11。I9 U-Net12/40、wet1/11，Transformer更低，不升级。
- I8 U-Net原训练6epoch×150step=900，best epoch1。训练MIL .636→.003，而验证约.65，存在明显泛化差距；增加epoch不能承诺改善。下一轮优先细记录早期与约束更新，不盲开第三模型。
- 复用3个已授权Luna Max代理：audit_field只读审I9；audit_features导出I8综合表、2–4现场案例和模型卡；audit_iter5实现I10训练器。root负责方案、启动器与曲线/计时。
- 不启动GPU，不自动执行修改后的检查/测试，不动CMDC/原始FDR/旧输出，不提交Git。

## 方案冻结

- Iter9物理提议test上界42/51（nonwet31/40、wet11/11），U9评分后13/51，瓶颈是评分泛化；不扩候选。
- Iter10固定900成功更新、每10步eval+step0=91点；结构不变，从I5权重+identity输入层初始化。backbone1e-5/stem1e-4/MIL .1/cal .5。按统一I5 val候选预算选择权重/阈值，同规则重评I8参考；best_dense.pt属于新轨迹，best.pt属于最终选择。
- root创建run_iteration10.py/.ps1、postprocess.py、plot_curves.py、README；复用I8 cache不新生成数据；postprocess field inference batch与validation一致，NN-only batch1计时独立。Matplotlib cache写实验output，不写C盘。
- 训练密集记录只对field全量train/val评估，nonfield validation降为step0及数据epoch边界以避免91次全1200 synthetic重复处理。训练器由Luna实现，待最终静态接口核对；不自动运行GPU或修改后测试。

## 本轮交付完成

- 已CPU导出I8：FieldNN_Iter10_20261023/output/existing_iter8/performance.csv、performance.md、MODEL_RESULTS.md、selected_cases.csv及3个现场案例图；MODEL_CARD_Iter8.md含7尺度残差U-Net/空洞卷积/全局上下文及输入与评价说明。
- F65 B相4/4（MAE4.014m），Nanling F55 C相单扫3/3（MAE6.421m、组6/8），Huzhong F23 A相2/3（MAE13.668m）。图含完整pulse/step概览、全部候选与局部pulse放大。selected_cases已分别记录单扫/组hit率，避免3/3对应.75的语义错误。
- 训练器最终静态契约审阅无阻断问题：AMP skip不算optimizer updates；csv、save_last、checkpoint选择与runner/postprocess/plot相符。没有自动运行测试/冒烟/GPU。
- 用户运行命令：pwsh -NoProfile -ExecutionPolicy Bypass -File D:\GitRepository\Cable-NN\DifferentAgent\CodeX\FieldNN_Iter10_20261023\run_iteration10.ps1；中断加-Resume。输出output/iter10/iteration10_summary.json；待用户运行返回后评估，尚无I10训练/增益结论。
- 三个Luna任务已完成；未操作DeepSeek进程或CMDC文件，未提交Git。

## 夜间直接执行授权与报错修复（同一会话续作）

用户报initial_validation在新训练分支赋值前被读取（845行）；root承认交付初始化顺序错误，修复中。用户明确授权root直接GPU训练/评估、不再问问题，不动DeepSeek/CMDC文件；最新要求至少50完整epoch，覆盖先前仅91记录点的训练时长约定。GPU读取状态空闲0MiB/0%，未见其他训练Python进程；不更改显卡配置。
新正式输出独立output/iter10_50ep，默认50epoch×150=7500成功更新，每100更新+step0共76个真实点。旧失败日志保留。三个Luna任务：修初始化、提出有限正则方案、做综合表图/最终文档。正式运行由root跟进至完成。

运行状态：正式会话15079已启动，GPU训练step300/7500（2epoch）、AMP0skip。val-MIL1.470→.793，fixed.94valhit .256→.333，预算校准valmacro .411→.739；best源已从I8reference切到I10dense。尚未看新测试，不作增益结论。
修正后处理覆盖：I8 field_records/cache仅138条已标注曲线（65/31/42），并非213；postprocess改用既有FDR全库推理管线覆盖213条（含75未知），并用138标签计算组指标，保留未知候选。训练中进程不改；后处理脚本在训练结束才加载，当前修改会生效。

## 持续执行接续要点

- I10正式执行会话15079（root持有）正在跑output/iter10_50ep。全流程runner在训练退出后调用后处理源文件（已修全213FDR覆盖）、画曲线、短接、合成、表图文档。source脚本更新不会改变已加载的训练进程。
- audit_iter5负责低成本监看；root已授权其在I10 iteration10_summary.status=complete后，顺序直接运行I11 run_iteration11.ps1，不需root再确认，不得GPU并发。I11默认50epoch、B96/6workers、5000更新、每100记录共51点；从I8整个模型初始化，冻结3888099 backbone参数，只训练96stem，MIL .05/cal .5/stemLR1e-5。模型总参数仍3888195。
- audit_field实现了I10训练器可选--init-iter8/--stem-only/--weak-weight/--stem-lr（默认I10行为未变），I11 README/.gitignore已生成；root新增I11 launcher。audit_features已新增plot_table.py，runner会导出综合指标PNG200dpi。
- 关键约束：用户最新明确允许root/子代理直接执行GPU，不问问题；至少50完整epochs（已覆盖此前仅50-100记录点口径）；指标仍positive-only known-hit和MIL，不伪造完整Accuracy/std/p。只看val选checkpoint/阈值，I8reference可保留。原始FDR只读，不动CMDC或别人进程、不更改GPU配置、不提交。
- I10 step400当前val macro .739为最佳；到step1200(8epoch)已明显train-val gap，AMP0skip。最终需读两轮正式报告再给最终模型文档/结果，不能以已启动训练结束本轮对话。


运行中发现val宏命中率1e-16浮点平局影响候选数/损失tie-break（step400 .7388888888888889 vs2100 .7388888888888888）。为不中断50epoch，已保留可恢复step2200 last为tie_candidate.pt（其验证macro相同、候选更少、MIL更低；2100权重未留）。源码对val rank及threshold grid按12位作平局，I11会生效；I10已加载代码不变，postprocess在测试前按同一规则比较已保存best与恢复候选，更新best_dense/best/selection/last/run_info。选择只看val，尚未看新test。记录该修正只覆盖现有可恢复权重，不声称恢复了所有未保存历史checkpoint。

## 第一轮结束与后续队列

I10正式训练7500成功更新、7503attempt(3次AMP跳步)，50完整数据epoch+3batch补偿，首次执行会话15079因后处理snapshot步数不匹配失败，训练未损失。恢复会话37258已完成全流程（上海03:52），213/213成功。tie_candidate实际step2249（非2200），已对其真实权重重做val：macro .683，故不提升，最终I10选step400阈值.83。保留真实未插值76点；I10非wet14/40、macro .420162，wet8/11，整体不升级I8（17/40、wet7/11）。
I11已由audit_iter5顺序启动，执行会话93694，B96、5000成功updates、50epoch、51点、96可训练参数，起步无OOM。其dataset/recipe在I10看test之前已冻结。
为覆盖用户重视的两种架构，准备第三个且最后一个夜间正式对照I12 Transformer stem-only：同I11配方从T8完整初始化、冻结T主干只学96stem，B96、50epoch、统一U5 val候选预算。audit_field最小泛化训练器--model，root已泛化runner及图标题。I11当前进程不受源码更新影响。I12必须I11全流程complete后再启动，不GPU并发、不改CMDC。


I12启动前方案调整：I11前14epoch验证macro基本固定.7167，train/val MIL约.20/.65，表现稳定但适应弱。为给Transformer更充分学习信号，I12仍只学96stem、50epochsB96，但MIL改.2、stemLR3e-5（cal仍.5），参考T8始终有资格保留。该决策只看I11 train/val，尚未看I11新test，不据test调参；不作纯架构单因素因果比较。I12 launcher和README已同步，worker将读取最新launcher。


最新持久状态：I10恢复会话37258已正常exit0，不再运行；I11会话93694由audit_iter5管理，约step1400/5000（14/50epoch），96stem参数，valmacro .7167保持初始化、AMP skip1、LR已到2e-6，无OOM。T12最小模型名泛化已完成，root写好run_iteration12.ps1（最终weak.2/stemLR3e-5、B96、50epochs）。audit_iter5已获明确授权在I11 top summary complete后顺序启动I12并监看，I12是今晚最后一个训练配置。共享runner默认--modelunet保持原样，T12传--modeltransformer；postprocess加载saved.model_name支持两架构；I10一次性snapshot校正调用已随select_checkpoint(model_name)签名更新。最终还需统一benchmark、合并不超过3个主模型的综合表、挑2–4现场案例、最佳模型文档/打包权重及现场阈值。
注意现场工作点与synthetic阈值文件：postprocess已另存field_thresholds.json，而后续synthetic校准会覆写run/thresholds.json。最终分发目录必须用field_thresholds.json作为thresholds.json；不能把合成阈值当现场工作点。I10完成在新增field文件前，必要时从selection与checkpoint_epoch重建field阈值配置。


验证阈值分辨率CPU单次分析已结束：I8 val 31条曲线、10袋、9个group（注意不要从macro小数反推组数）。统一U5 cap79/31；U网格.94→unique值.9412549：macro .716667、74候选完全不变；T网格.98→.984992：macro .683333相同，仅48→46候选。无命中增益，决定不扩展后处理、不采用精扫，继续原.01阈值网格。脚本I10/audit_threshold_resolution.py，output/threshold_resolution_val.json；仅val评分无test选参。


## 05:15 接续（同一root会话）
确认I12正式T5-stem进程PID21868持续运行，18/50epoch、AMP0skip，不重启；此前一次进程列表因格式化未展示并非中断。audit_iter5恢复只读监看；audit_features复核finalizer接口与曲线归属；audit_field补充固定既有0.60覆盖优先工作点（复用I11预测/真实060记录列，不新forward），与原预算.94并列且必须报告候选负担。主权重/阈值不因此改动，不据test调参。I12为本夜最后训练配置，结束后完成最终表、2–4案例、模型文档与推理包。读取stop-that-shit技能，收束到交付。外部memory registry无Iter10/12命中，未依赖外部memory事实。

05:30：I12 T5-stem约27/50epoch、AMP0skip，验证仍选T8参考。已完成I11补充固定.60工作点真实CPU cache forward：138标注曲线，CPU单线程batch1/FP32，非湿22/40、macro.6667124542、203候选/31=6.548；湿8/11、57/11=5.182。原.94主点非湿17/40、macro.509371、94/31，推理全213、GPU FP16，两计算路径差异不可当提升。51点.60历史train/val末值.86938/.85556，best checkpoint仍step1100，末点非best。三案例F65 4/4(10候选)、F23 3/3(8)、F55 3/3(10)。有效目录I11/output/coverage_workpoint/cpu_cache_export_060。最初错误用.94过滤输出导出.60被root识别，已移stale_export_from_094_filtered_predictions并注明无效；勿引用。audit_features在最终卡/综合表补两工作点与范围，不改主阈值。I12 reference-selected不能冒充新50轮权重，finalizer已据selected_source修候选资格。

05:38：已启动root顺序交付等待进程session37360，pwsh每50秒等I12 top summary complete后自动执行finalize_delivery.py --run-benchmark；CPU affinity0x3fff/OMP1/MKL1，不与训练并发GPU。I12已38/50，仍参考T8。audit_features已把有效CPU .60工作点接入最终performance.csv/PNG（bestU=I11时同模型第二工作点；非I11则附表），卡片链接有效cpu_cache_export_060，最终I11包另存thresholds_coverage_060.json默认.94不变。root纠正plot_table脚注为“每配置单次训练N=1”而非案例N1。最终生成后尚需读取产物结果并查看综合表/模型卡、更新里程碑和记忆，才结束任务。

## 05:55 正式交付完成

- I12完整50epoch、5000成功updates、5001attempt、AMP1skip、51记录点；dense_best4100，val最终仍选iter8_reference(.98、macro.683333)。全213/213推理与9stage完整完成。不再启动训练。
- root顺序交付37360已完成统一CUDA benchmark（每模型batch1 FP16、20预热100次同步计时），随后finalizer末尾索引因cpu_prediction_scope字段不存在失败。root改scope为现有cpu_inference字典，复用benchmark JSON重导；修正training_objective漏列reference-selected I12及fallback文字“不产生checkpoint”不准确。最终96219 exit0；无重复benchmark或重训。
- 最终赢家I11-U-Net、bestT历史T8。I11 3,888,195参数，NN median/p95=10.083/11.189ms，权重包15,603,962bytes，raw epoch10(从0计，第11轮)、best step1100。完整训练时长50epoch不是best发生在第50轮。
- 最终根目录：DifferentAgent/CodeX/FieldNN_Iter12_20261023/output/final_delivery；performance.csv/PNG 4工作点行(3模型)，historical_results.csv、comparison.md、MODEL_CARD_BEST.md、model_index.json；best_model/inference.pt、thresholds.json/field_thresholds.json(.94)、thresholds_coverage_060.json(.6)。source保留I12根MODEL_CARD_BEST.md及生成器/模板，output由ignore排除。
- .94非湿17/40、macro.509371、94/31候选；对I5同94候选13/40和macro.399328有增益；对I8 17/40、93候选无新增整体提升。.6补充非湿22/40、macro.666712、203/31=6.548候选；湿8/11、57/11=5.182。不同CPU FP32缓存与GPU FP16路径，不把U8 205对203差异当改进。
- .6现场案例3例来自不同非RG58保留组：F65 B4/4(10候选)，F23 A3/3(8)，F55 C3/3(10)；有效资产见I11/output/coverage_workpoint/cpu_cache_export_060。主.94案例为4/4、3/3、2/3，均保留所有未核验候选。
- I11曲线51点0–5000；.6 group train/val末值.86938/.85556。MIL首末train.199904→.199439、val.649536→.654840，约.45泛化差距仍在，明确不是从零收敛或解决过拟合。模型卡已补该解释及raw epoch计数说明。
- root查看了最终综合表与.6真实曲线、F65现场图：修复表格长模型名越界为短标签多行，最终PNG无该越界。没有额外测试/构建/lint/冒烟。所有既有长任务均已结束，未改CMDC、原始FDR、GPU配置、GUI，未提交。
