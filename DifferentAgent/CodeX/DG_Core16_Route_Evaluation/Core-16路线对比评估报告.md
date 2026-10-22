# Core-16 DG路线对比评估报告

- 运行时间：249.87 s
- 频率网格：9 kHz–200 MHz，10000点
- 算法2：time_points=10000，chunk_size=256，距离分析上限=60.0 m
- 输出同时保留 clean kernel 和 measured 两层；measured 由三条路线共用同一测量链和按工况固定的随机种子产生。
- 本目录是独立扩展实验；原 `DG_Route_Evaluation` 目录不被脚本写入。

## 1. 路线与公平性

三条路线固定为：DG V3 频变连续 RLGC（100 MHz锚定）、CST V1 0.1 m固定RLGC梯形、CST V1连续固定RLGC参考。三条路线共同使用40 m主体、1 m/135 Ω前置无损线、10 m特殊接头、20/30 m的0.5 m/300 Ω无损线、末端开路及相同的FDR和测量链。

CST梯形每个0.1 m单元按 `scale=0.1/0.4=0.25` 缩放每0.4 m等效RLGC；没有把0.4 m数值直接复制到0.1 m单元。DG V3按现有 effective_rlgc 方法把每米RLGC在100 MHz锚定，并保留其频变R/固定G行为。

## 2. Core-16 工况

| ID | 工况 | 空间类别 | 作用 | 末端附加G (μS) | 末端附加C (pF) |
|---|---|---|---|---:|---:|
| C00 | 正常 | baseline | 完好同型号电缆基准 | 5 | 16 |
| C01 | R↑（局部） | local | 接头压接不良、局部接触电阻、导体损伤 | 5 | 16 |
| C02 | R↑↑（局部） | local | 严重高阻接触、断股/近断路型工况 | 5 | 16 |
| C03 | G↑（长区段） | long_section | 长区段介质损耗增加的主导分量 | 5 | 16 |
| C04 | G↑↑（局部） | local | 局部强漏导/高损耗 | 5 | 16 |
| C05 | G↑（分段） | segmented_loss | 多区段不同程度损耗 | 500 | 16 |
| C06 | C↑（长区段） | long_section | 长区段介电参数增加 | 5 | 16 |
| C07 | C↑↑（局部） | local | 局部强容性异常 | 5 | 16 |
| C08 | C↓（局部） | local | 局部低电容/低介电异常 | 5 | 16 |
| C09 | G↑+C↑（局部） | local | 局部受潮、水树/湿损型复合介质异常 | 5 | 16 |
| C10 | G↑+C↑（长区段） | long_section | 分布受潮、长区段水树/湿老化 | 5 | 16 |
| C11 | R↑+G↑（局部） | local | 接触异常形成热点并伴随绝缘损耗增加 | 5 | 16 |
| C12 | R↑+G↑+C↑（局部） | local | 热点长期发展后的导体+绝缘复合劣化 | 5 | 16 |
| C13 | R↓+L↑+G↓+C↓（整段） | replacement | 异型电缆拼接 A：高特性阻抗、低损耗型 | 5 | 16 |
| C14 | R↑+L↓+G↑+C↑（整段） | replacement | 异型电缆拼接 B：低特性阻抗、较高损耗型 | 5 | 16 |
| C15 | R↑+L↑+G↑+C↓（整段） | replacement | 异型/修复电缆拼接 C：另一组全 RLGC 差异 | 5 | 16 |

C05 的末端属于30–40 m第四段，因此末端附加支路固定为G=500 μS、C=16 pF；其他工况末端附加支路保持G=5 μS、C=16 pF。

## 3. 数值检查汇总

这里的 `numeric_gate_pass` 只检查响应有限性以及 clean 层 `|S11|≤1+1e-9`，不把未经本轮确认的物理方向写成硬门槛。方向、幅度和三路线差异使用 `metrics.csv` 与各工况对照图复核。

| 路线 | 层 | 数值通过数 | 总数 |
|---|---|---:|---:|
| DG V3 RLGC（100 MHz锚定） | clean | 16 | 16 |
| DG V3 RLGC（100 MHz锚定） | measured | 16 | 16 |
| CST V1 0.1 m梯形 | clean | 16 | 16 |
| CST V1 0.1 m梯形 | measured | 16 | 16 |
| CST V1 连续固定RLGC | clean | 16 | 16 |
| CST V1 连续固定RLGC | measured | 16 | 16 |

逐条原始指标见 `output/metrics.csv` 和 `output/metrics.json`。脚本没有用任何实测曲线或距离域模板替换S11。

## 4. 对照图

### clean

![clean-C00](assets/comparison_board_clean_C00.png)

![clean-C01](assets/comparison_board_clean_C01.png)

![clean-C02](assets/comparison_board_clean_C02.png)

![clean-C03](assets/comparison_board_clean_C03.png)

![clean-C04](assets/comparison_board_clean_C04.png)

![clean-C05](assets/comparison_board_clean_C05.png)

![clean-C06](assets/comparison_board_clean_C06.png)

![clean-C07](assets/comparison_board_clean_C07.png)

![clean-C08](assets/comparison_board_clean_C08.png)

![clean-C09](assets/comparison_board_clean_C09.png)

![clean-C10](assets/comparison_board_clean_C10.png)

![clean-C11](assets/comparison_board_clean_C11.png)

![clean-C12](assets/comparison_board_clean_C12.png)

![clean-C13](assets/comparison_board_clean_C13.png)

![clean-C14](assets/comparison_board_clean_C14.png)

![clean-C15](assets/comparison_board_clean_C15.png)

### measured

![measured-C00](assets/comparison_board_measured_C00.png)

![measured-C01](assets/comparison_board_measured_C01.png)

![measured-C02](assets/comparison_board_measured_C02.png)

![measured-C03](assets/comparison_board_measured_C03.png)

![measured-C04](assets/comparison_board_measured_C04.png)

![measured-C05](assets/comparison_board_measured_C05.png)

![measured-C06](assets/comparison_board_measured_C06.png)

![measured-C07](assets/comparison_board_measured_C07.png)

![measured-C08](assets/comparison_board_measured_C08.png)

![measured-C09](assets/comparison_board_measured_C09.png)

![measured-C10](assets/comparison_board_measured_C10.png)

![measured-C11](assets/comparison_board_measured_C11.png)

![measured-C12](assets/comparison_board_measured_C12.png)

![measured-C13](assets/comparison_board_measured_C13.png)

![measured-C14](assets/comparison_board_measured_C14.png)

![measured-C15](assets/comparison_board_measured_C15.png)
