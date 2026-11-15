# 首轮代码接口

模型接收单相、两相或三相，使用共享单相编码器。多相上下文为可用相位特征的平均与自身特征之差；单相时差为零。各相绝对特征始终保留。缺相与缺响应表分开记录，不猜测相名。

## 准备数据

每个comparison_group保存一个NPZ：`x`为[P,5,L]、`valid`为[P,L]、`phase_mask`为[P]、`distance_m`为[L]；`meta`为JSON字符串，含comparison_group_id、case_group_id、measurement_ids、源文件/派生文件关系、events与negative_ranges、原始设置。P固定槽位A/B/C用于已有比较组；独立单相可P=1，phase名只作元数据。

公共距离网格0–3000m、间隔0.25m，不按真值末端裁输入。每相仅在原响应表实际覆盖内有效。保留原始复S11、校准与厂商表到独立源缓存，不把显示降采样用于训练。

五通道顺序：平滑脉冲的signed_log1p、原始阶跃、平滑阶跃、平滑阶跃对距离的导数、distance/3000。不逐相幅值归一。平滑在完整源表上执行，按配置的物理跨度（初值0.5m）和实际距离步长换成奇数窗，再重采样；阅读版K参数与训练物理跨度分别记录。直接复用居中移动平均算法，不写新滤波框架。

`manifest.json`格式：`{"groups":[{"comparison_group_id":...,"case_group_id":...,"file":"groups/<id>.npz","supervised":true,...}]}`。不自动冻结train/val划分。split文件显式给出`{"train":[case_group_id...],"val":[case_group_id...]}`，必须按物理案例归组。

`review/accepted_labels.json`格式为`{"measurements":[{"measurement_id":...,"case_id":...,"events":[{"task":...,"shape":...,"position_m":...或"start_m":...和"end_m":...,"required_measurement_ids":[...],"source_refs":[...]}],"negative_ranges":{"anomaly_point":[[lo,hi]],"anomaly_range":[[lo,hi]]}}]}`。缺省required_measurement_ids为自身。石杨C复测单侧起点需要原审阅比较组上下文，不伪造成完整占据区间。起点用区间头局部上升约束：以起点左右各2m为训练差分模板，优化softplus(-(右侧平均logit−左侧平均logit))。这是局部训练模板，不是新增已确认异常范围；不监督未知终点。

## Dataset与模型

dataset.py提供`FieldDataset(data_dir, case_groups, training=False, single_phase_probability=0.5)`及`collate_groups`。每样本返回x、valid、phase_mask、distance_m、meta；目标为target与supervision_mask，均[P,4,L]。任务顺序joint、terminal、anomaly_point、anomaly_range。

点标签使用固定训练核（默认sigma=2m）表达，不挪中心，不把核外未知当负例。区间仅监督已确认内部；负例来自对应任务明确范围。阶段标签默认等权。缺输入或缺所需比较上下文时，不使用该条监督；不凭三相标签要求缺上下文的单相输出同样结论。

训练按组随机保留全部可用相或只保留一个可用相，绝不移除全部相。损失和输入mask同步。评估分别提供完整上下文与每个单相视图，按标签所需上下文报告可评分覆盖，不把上下文不足直接计作漏检。

model.py提供`PhaseDetector(in_channels=5,width=16)`，forward(x[B,P,5,L],valid[B,P,L],phase_mask[B,P])返回logits[B,P,4,L]。归一化只沿同位置通道，不跨距离或batch。每尺度屏蔽无效位置。上下采样采用固定倍率加裁剪，避免padding总长改变坐标。

dataset另返回`boundary_constraints`：每项包含phase_index、center_index、left_index、right_index。collate补batch_index。模板需在原有效覆盖内，并满足required_measurement_ids；单相视图缺少石杨所需相间上下文时不使用此约束。losses.py提供`masked_detection_loss(logits,target,supervision_mask,boundary_constraints=None)`，返回loss和统计；按任务平衡有效正/负监督，不让未知区进入负项，局部起点约束单独记录。无有效监督和起点约束的batch不做优化器更新。

## 训练与结果

train.py只在用户执行时运行。A臂随机初始化；B臂通过`--encoder-init`加载共享单相编码器，融合和输出仍同A臂随机初始化。现有单条合成电缆不拼成假三相；本次不实现未经定义的DG标签转换。

记录进度、各头训练/验证损失、有效监督量、学习率与checkpoint。不能把正例loss最低等同现场F1最优。首版按同口径验证loss保存，评价脚本另报内部任务与末端、单相与多相；参数默认是起始配置，不声称已调优。

不自动训练、测试、构建、lint或运行推理。当前只有少量用户确认案例，不自动生成具有正式验收含义的数据划分。
