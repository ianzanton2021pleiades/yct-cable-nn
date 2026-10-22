# FDR 数据生成—训练—现场评估 Session

- 日期：2026-10-22（Asia/Shanghai）；主导 Agent：CodeX；工作目录 D:/GitRepository/Cable-NN。
- 用户目标：以 DifferentAgent/CodeX/Core16_Field 为基础完成数据生成、Mamba/Transformer/第三种（优先讨论 U-Net）模型训练、推理与评估；识别现场接头、阻抗不连续区间及位置，可讨论三相对比。
- 授权：生成数据允许写 E:/CableNNDataSet/CodeX；E:/20261010截至的全部fdr文件只读；使用 gpushare_cu124 与 CMP 90HX，但不得修改用户设置的功率、风扇或驱动。新 GPU 授权适用于本项目任务。
- 并行：用户允许 GPT-6 Luna Max 子代理/线程以节省额度，并发最多4；实际环境含主代理共4槽，本轮派3个只读子代理，未创建用户侧线程。
- 提问预算：本session最多4轮 grilling，每轮不超过4问。第1轮已答，第2轮已发待答；剩余2轮。
- 执行边界：不自动提交；不修改已有GUI非神经网络部分；默认不自动测试。长时间生成/训练启动方式、标签定义及划分待用户决策后实施。
- 已读：项目 AGENTS.md、Core16_Field 旧session记录、现场证据与实现说明、field_config.json/field_responses.py/README；使用 officecli、stop-that-shit、grilling 及 PDF 技能，PDF 内容委派只读提取。
- 已有工作区变化（本轮之前）：AgentsMEM/2026-10-22_CodeX_Core16_Field.md 已修改；REF/论文/ 未跟踪。保留不动。
- 当前确认：Core16_Field 未冻结训练参数，2430m anchor 拟合内峰位仍偏约73–108m；固定接头比例、局部宽2m；measurement 噪声和误差增量为零。需避免固定几何捷径与拟合案例进入最终盲测。
- 环境仅静态目录读取：E:/Anaconda/envs/gpushare_cu124 有 torch-2.5.1+cu124、NumPy/SciPy/transformers/tqdm；未见 mamba/triton/einops/h5py/zarr 目录。尚未导入torch或启动GPU，包兼容性未验证。
- 本轮三个子任务：field_generator_review（接口/标签/随机化）、corpus_labels_review（已有统计/分组/三相/留出）、paper_methods_review（论文数据和防过拟合，页码证据）。

## 第1轮决策及追加授权

- 同意事件位置/置信度、异常区间、独立末端输出，不做16类成因分类。
- 用户选择“只用已有资料，不补充人工复核”；不能以用户将来补标为完成条件。已知标签不足时仅定性或证据范围内评价，不把未知当阴性。
- 单相基线后再加可选三相。
- 明确允许必要小规模CPU/GPU检查；正式生成/训练/批量评价由用户启动，需提供命令。
- 用户要求所有本实验模型及其他内容放 DifferentAgent/CodeX 下新目录，已建 FieldNN_20261022，包含.gitignore和PLAN.md。若额度不足，交付可继续的计划。
- 已执行一次微型环境检查：torch2.5.1+cu124，CUDA可用，NVIDIA CMP90HX、compute8.6、显存10GiB，32×16矩阵反向有限；无mamba_ssm/triton/einops/h5py/zarr。不更改GPU设置；无长任务。

## 资料核对结果

- Core16_Field路线 dg_v3_rlgc 是本模块频变连续核，不是DG_V3 effective_rlgc的逐字复用；field仅加近端网络，measurement误差未应用。现有truth宽分析窗不能直接当精确异常区间。
- 现场67个短线FDR中63个具有A/B/C命名，不等于63根独立电缆；相别和复测及跨日期同线必须合组。
- 开发排除组：无锡混合目录、7月20日两段关联测试、9月8/14污水厂同线处理前后、1500m浸水试验；参与拟合/先验构建，不得作为最终独立测试。
- 论文为监督PD ToA（1660条信号/5145标注时刻），非S11合成或无标签自监督；Gaussian热图/验证定阈值可迁移。论文按campaign/experiment分层但不能据此证明同电缆整体隔离；增强参数和逐样本峰值归一化不可直接照抄FDR。
- corpus_labels_review已接续实现本实验prepare_real_manifest.py与可追溯关联规则，只读取已有统计，不重复全库解包。

## 第2轮决策（已答）

1. 只以频变dg_v3_rlgc field为主训练，clean及另两路线少量对照；≤2500m。
2. 原始S11完整保留，三模型统一IFFT脉冲/累计阶跃输入、保留幅值尺度，不依赖真实长度裁剪。
3. 纯PyTorch真Mamba（input-dependent delta/B/C、卷积门控）优先，先测吞吐；或解决官方内核环境。
4. 首轮1万条、按底层拓扑8:1:1分组，共用划分，后续按学习曲线扩容。

用户全部接受；特别强调距离域首版仍必须关注生成S11合理性，效果不佳后再考虑复数S11输入。

## 实现与当前状态

- 新实验目录FieldNN_20261022：common/config、generator/generate_dataset/randomization_defaults、models/check_models、train/evaluate/metrics/infer、prepare_real_manifest/real_group_rules/real_evidence、audit_dataset、PLAN；README和run_pilot已由paper代理补齐。主代理补充绝对路径命令和Python -X utf8 -B，未执行启动器长阶段。
- 原Core16_Field和DG不修改；Core16物理anchor只读并复制进生成输出effective config以便复现；Milestone已追加。
- 主路线频变RLGC field，首轮10000条；100–2500m四个长度区间均衡，C00 25%，其余15机制均分，0–6接头、随机局部/长段/分段异常。材料/夹具抖动和测量误差/相关复噪声是先验而非实测识别值。
- 现场612/619为对数扫频，因此最终cfg.frequency.spacing=log，32768点9k–200MHz。common.frequency_grid供生成和评价复用；没有强行用插值增加信息。原smoke_v2是线性网格历史证据。
- 4输入通道：脉冲/RMS、step_rho/RMS、两路log10 RMS广播；固定0–3250m8192点，无真实长度裁剪。3输出为事件/参数异常区间/独立末端。区间是合成RLGC参数变化真值，不宣称现场全部可辨。
- 三模型参数量Mamba197499、Transformer293691、UNet216507；Mamba是真input-selective delta/B/C、A_log、卷积门控与双向scan，非官方融合CUDA核。
- 初次无预热单次基准不能作正式吞吐。补做batch4 AMP、2步预热+3步计时：训练吞吐Mamba17.05、Transformer220.77、UNet216.77条/s；峰值分配2.274/.148/.163GiB。这仅为微型短测。
- 训练保存history.csv/best.pt/last.pt、ReduceLROnPlateau与验证早停，支持--resume并允许从首轮5epoch继续扩展；last写临时文件后replace以支持长训恢复。首次实现Cosine已改为Plateau，避免短轮结束lr到0后扩展epoch问题。
- 阈值仅在val选，test冻结并检查model/epoch；事件按2/5/10m最大基数一对一匹配、区间IoU@.5和网格IoU、末端命中与误差。真实unknown不能调用合成完整truth指标。
- prepare_real_manifest CPU运行：213条短线，A/B/C/unknown=67/64/64/18；31条development_only、182条independent_identity_unconfirmed，所有机器标签unknown。候选证据另存real_evidence，不提升为gold。
- 最终对数smoke_log_data：20条、2workers、shard4，16/2/2 split，约3.1s；三模型各1训练batch(B2)+1valbatch，保存checkpoint；三模型均完成val选阈值和test；transformer对1个development现场原FDR完成S11起始推理（未算准确率）。审计20条无完整性问题。U-Net恢复训练单独验证1->2epoch通过。
- 历史smoke_data因缺物理anchor失败，未删除（ignored）；smoke_v2_data是成功线性网格20条，output/smoke_pipeline是其微型模型；最终以smoke_log_data/output/smoke_log_pipeline为准。所有生成输出被实验.gitignore覆盖，没有提交。
- 原始|S11|数组保留；物理网络被动检查与measurement后轻微超1分开。首次线性20条最高physical .9877，measurement1.00765，不泛化为正式全量结论。
- 每条约515KB（含S11/input/targets/truth），1万约5.2GB；还未写正式E盘数据或跑长训练。用户后续手动Generate/Audit，先看域差，再每模型5epoch趋势；结果再决定优化/扩容。
- 真实一致性尚未建立：先前anchor本来有73–108m残差；窗内脉冲最强峰可能贴窗边界，不能当真实末端；频带合成曲线与现场case权重不同、频率密度也影响中位数。audit pass仅指完整性，不得宣称已达到现场训练质量。
- 单相试验结果后再实施三相融合、真正模型迭代及GUI接入，不能说本轮已完成这三项。剩余2轮grilling保留给正式结果决策。

## 用户反馈后的复数S11检查

- 用户指出生成S11实部/相位不像现场，明确要求检查且参考DG_V2.7/V3，不能只比幅值。本轮定位原因，不自动修改生成参数或重跑正式生成/训练，无GPU。
- 当前E:/CableNNDataSet/CodeX/FieldNN_20261022已存在正式10000条（8000/1000/1000，157shards），由用户执行后产生，保持只读。
- 复用3条已有开发NPZ：无锡蓝安名义1100m（sweep1单次测试，非A相，9k200M线性50000）、637m A末端开路与2430m A（9k1GHz对数50000）。S11 NPZ=Real+jImage，无共轭变换；X_Axis_Offset只是显示偏移，未进入频域相位，不能直接当参考面校准。
- 现行Core16主体约alpha(100MHz)=.627dB/m，DG V3 Field alpha目标.003–.026dB/m，差24–209倍；tan_delta=.031676比V3先验.00025–.0035高9–127倍。当前单例等效拟合不能作为通用正常电缆材料。
- 父代理原生同频点C00对照初算：637m实测.1–1MHz展开phase约-6.98圈，当前C00约-.92圈；2430m5–200MHzcomplexRMSE约.093/circular phase RMS5.9deg；无锡同带complexRMSE约.899/phase106deg。不同文件原生线性/对数频率加权不同，不能拿3个RMSE作整体质量排名。
- 高频100–200MHz实测相位线性斜率等效延迟：无锡88.009ns，637m50.588ns，2430m48.765ns。当前24.4ns夹具单程延迟及±15%抖动，往返项约41.5–56.1ns，覆盖范围过窄；实际相位斜率不等于已唯一确定实物夹具长度。
- 三套传输线相位约定一致：Z'=R+jωL、Y'=G+jωC，往返exp(-2γL)。无直接相位反号证据；Core16的dg_v3_rlgc是自写同名分支，不是V3原核。
- 新FieldNN已经加复数测量误差/相关噪声，不能误称它仍是Core16原配置的零measurement；主要问题是材料/夹具/参考面先验泛化。
- V2.7主Field外层含模板慢幅值/实部快速能量约束、相位ripple、近端阻尼谐振，以及Client-IFFT逆变换事件/端末支路；外观接近现场有目标塑形贡献。默认主路径inject_joints=False，不能把core简化接头helper当作活跃路径；ensure_terminal_visibility也未见主路径调用。
- V3当前effective_rlgc按alpha100与tanδ扣损耗预算求R100，条件不自洽会拒绝；旧provisional_v1.yaml没有material配置，无当前兼容运行路径，dispersion_fraction当前被忽略。
- corpus代理已完成review_complex_s11.py（CPU约10秒）：3case原生频点对照、实虚/圆周相位/相位圈数、同夹具V3材料损耗预算受控对照；图/summary.json/复数NPZ在output/complex_s11_review，父代理已查看低频/高频两图。不做参数优化。
- 受控材料方案：同Z0=19.8/速度/长度/开路/夹具、均匀无接头；Core alpha(.1/1/100MHz)=.004657/.018803/.625553dB/m，V3预算(.02dB/m@100M,tanδ.001)=.000166/.000630/.020000。637m .1–1M Core均匀体-.912圈/2 Re零交叉，V3预算-6.700圈/14，实测-6.980圈/14。
- 重要反证：V3预算虽恢复绕圈/零交叉数量，该频带complexRMSE .755比Core均匀体.319更差，相位误差亦增大，1–5M还多出振荡。因此不能宣称直接换旧V3参数解决；需联合约束Zc、R/G损耗、速度/色散和近端。
- 已新增S11_REVIEW.md完整结论和图链接，PLAN/README标记当前先不继续正式Train，E盘10000条保留。核心源码/生成配置/数据未修改；没有GPU/训练。上轮复数形态验收不足已在报告明确。

## 用户最新决策：冻结数据先跑Transformer/U-Net

- 用户时间有限且额度仅余15%，明确不改生成器、复用现有数据，Mamba搁置；要求训练/推理/评价并提前留完整Markdown接续。
- 新请求给2轮grilling，每轮最多4问。第1轮已问启动方式和epoch，用户明确同意Agent直接启动Transformer、U-Net各5个完整epoch，验证最佳后评价和开发现场推理。此次授权更新此前长任务用户启动约定；GPU设置不得改。
- 已先创建FieldNN_20261022/HANDOFF_TRANSFORMER_UNET.md作为低额度接续入口，目标run为output/tu_baseline_20261022。未请求新增真值标注；现场unknown不作准确率。

- Transformer实际全epoch约50s（约175–192条/s），已5轮完成，best epoch5，val_loss .328837。初始瞬态约10batch/s曾被误当稳态，父代理建议B32又撤销；实际未用B32，全部有效epoch均batch4。中途Ctrl+C在epoch3未完成处，从epoch2的last恢复，history最终完整1..5。
- 完整val校准出现CPU重复计算瓶颈，授权只优化evaluate._calibrate_thresholds：15次全量解码/计分合为每候选一次共5次，三头指标独立；真实val前2条+确定性概率旧/新候选分数及阈值完全一致。保留完整1000条/原5候选/输出schema，不抽样。
- 当前允许Transformer CPU阈值计分与U-Net唯一训练进程重叠，避免GPU闲置，不并发两份训练；执行者field_generator_review管理session。summarize_tu.py已由paper代理写好，产物齐备后运行一次输出comparison.md/json/learning_curves.png。
