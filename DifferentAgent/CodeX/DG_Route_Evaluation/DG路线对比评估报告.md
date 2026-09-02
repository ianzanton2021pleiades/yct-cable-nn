# DG 路线对比评估报告

- 运行时间：182.61 s
- 统一频率网格：9 kHz–200 MHz，10000点
- S11为唯一权威信号；算法1/2结果均由同一S11重新计算。
- V2.7只作历史真实感参照，不参与路线获胜判定。

## 1. 候选与公平性

DG V3在九工况中使用其`effective_rlgc`传播核，并在100 MHz锚定CST每米R/L/G/C；四路共用CST接头、端点和频率网格。DG V3的R随平方根频率变化，而CST的R固定，因此差异属于模型路线本身。统一测量层给四路施加完全相同的延迟、损耗、VNA误差和相关噪声。

## 2. 工况方向检查

| 候选 | clean通过数/9 | common-measurement通过数/9 | clean最大|S11| |
|---|---:|---:|---:|
| DG V3 RLGC (100 MHz锚定) | 9/9 | 9/9 | 0.993746 |
| CST V0 0.4 m梯形 | 9/9 | 9/9 | 0.999899 |
| CST V1 0.1 m梯形 | 9/9 | 9/9 | 0.994556 |
| CST V1 连续固定RLGC | 9/9 | 9/9 | 0.993693 |

四路均为9/9，说明这组方向检查只能证明四种实现都能表达报告工况，不能区分哪条路线更适合作为DG。逐工况原始指标和布尔检查见`metrics.csv`与`metrics.json`。

## 3. 已确认的路线证据

1. CST V0的0.4 m梯形在约145 MHz出现离散截止，算法1接头后尾波/主峰约0.159；该纹理是离散伪影，不是真实制造不均匀。
2. CST V1 0.1 m梯形把截止推到约581 MHz、尾波降至约0.00362，但相对连续模型的全频S11和脉冲误差仍未达到既有0.05阈值，不能宣称已收敛。
3. CST固定并联G与DG介损模型频率依赖不同。报告中16 pF/200 kΩ在100 kHz实际对应tanδ约49.7%，不是0.5%；因此报告曲线只能作为工况趋势，不能作为介损数值真值。
4. 现有DG V3 RLGC实测验证在30/30条Core留出曲线上优于旧V3，但该验证没有测量链，不能单独证明完整实测域真实性。
5. V2.7的本体纹理和受潮形态提供了有用目标，但其距离域核反算、末端重塑和局部修形违反当前S11权威边界。

## 4. 实测证据状态

当前只读目录共发现405个CSV，其中IFFT目录129个、无校准S11 125个、RG58缺陷制造4个、1500 m浸水12个。

已在`real_data_calibration`重新生成当前聚合统计：纳入250条测量，其中RG58=139、Field=85、校正数据=14、1500 m浸水=12；重复组61、成对比较136。未标注现场文件只承担分布真实性，不承担缺陷类型和位置真值。

### DG V3原生分布硬门槛

使用`provisional_rlgc_v1`各生成24个RG58和24个Field目标样本，并将S11幅值分位数、幅值斜率和相位斜率与当前实测聚合比较。

| 类别 | 有效样本/尝试 | 失败次数 | 通过特征/5 | 类别门槛 |
|---|---:|---:|---:|:---:|
| RG58 | 24/24 | 0 | 3/5 | 未通过 |
| Field | 20/96 | 76 | 2/5 | 未通过 |

Field失败均来自`dielectric loss exceeds the requested 100 MHz attenuation`：抽到的介损/受潮参数已经消耗或超过配置中的总衰减预算，当前参数空间不自洽。即使只看成功样本，Field的S11幅值q95、幅值斜率和相位斜率仍不在实测q05–q95内。RG58的三项幅值分位数进入实测范围，但幅值斜率与相位斜率失败。

相位斜率受反射零点和低幅值区unwrap影响较大，因此它是高风险诊断项；去掉该项也不会改变结论：RG58幅值斜率仍失败，Field幅值上沿和幅值斜率仍失败。完整记录见`output/distribution_gate.json`。

![DG V3原生分布与实测聚合](assets/distribution_gate.png)

## 5. 路线结论

没有任何当前候选通过三道门槛：四路均通过被动性和九工况方向检查；CST三路没有RG58/Field随机拓扑及测量分布，无法通过实测域生成器门槛；DG V3具有该能力，但本轮原生Field生成失败率和分布偏差使其未通过实测域门槛。

当前证据不支持用CST梯形网络整体替换DG V3。V0存在明确带内离散伪影；V1 0.1 m虽然改善，但对2.5 km电缆意味着约25000个单元，计算成本高且仍未证明数值收敛；连续固定RLGC又缺少真实导体和介质的频变损耗。

建议路线是：以DG V3连续频变RLGC和S11权威协议作为下一版底座，但当前`provisional_rlgc_v1`不得用于正式训练集。先修正Field总衰减预算与介损/受潮参数的联合采样，再加入小幅、空间相关的Z0/epsr/导体损耗/tanδ随机场，使完好电缆自然产生可解释的本体纹理；把CST九工况固化为物理回归集；受潮继续通过平滑耦合的介电常数与介损变化生成，不移植V2.7距离域修形。

## 6. 输出图

### clean

![clean-baseline](assets/comparison_board_clean_baseline.png)

![clean-overall_C20pF](assets/comparison_board_clean_overall_C20pF.png)

![clean-overall_G20k](assets/comparison_board_clean_overall_G20k.png)

![clean-segmented_loss](assets/comparison_board_clean_segmented_loss.png)

![clean-local_C32pF_15m](assets/comparison_board_clean_local_C32pF_15m.png)

![clean-local_G2k_15m](assets/comparison_board_clean_local_G2k_15m.png)

![clean-local_C4pF_15m](assets/comparison_board_clean_local_C4pF_15m.png)

![clean-local_R10ohm_15m](assets/comparison_board_clean_local_R10ohm_15m.png)

![clean-local_R50ohm_15m](assets/comparison_board_clean_local_R50ohm_15m.png)

### common_measurement

![common_measurement-baseline](assets/comparison_board_common_measurement_baseline.png)

![common_measurement-overall_C20pF](assets/comparison_board_common_measurement_overall_C20pF.png)

![common_measurement-overall_G20k](assets/comparison_board_common_measurement_overall_G20k.png)

![common_measurement-segmented_loss](assets/comparison_board_common_measurement_segmented_loss.png)

![common_measurement-local_C32pF_15m](assets/comparison_board_common_measurement_local_C32pF_15m.png)

![common_measurement-local_G2k_15m](assets/comparison_board_common_measurement_local_G2k_15m.png)

![common_measurement-local_C4pF_15m](assets/comparison_board_common_measurement_local_C4pF_15m.png)

![common_measurement-local_R10ohm_15m](assets/comparison_board_common_measurement_local_R10ohm_15m.png)

![common_measurement-local_R50ohm_15m](assets/comparison_board_common_measurement_local_R50ohm_15m.png)
