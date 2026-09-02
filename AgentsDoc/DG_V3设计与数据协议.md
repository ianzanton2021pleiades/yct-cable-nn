# DG V3设计与数据协议

## 1. 权威数据边界

DG V3 的权威数据只有三项：版本化生成参数、双频复数S11、稀疏物理真值。脉冲响应、阶跃响应、固定距离网格、归一化通道和模型标签都属于下游派生结果。

因此改进 IFFT 时只需重建 `responses`，不需要重新生成物理拓扑和S11；未来模型也可以从同一份权威数据选择不同频段、网格或输入表示。

## 2. 生成机制

- RG58 与 Field 使用独立长度、阻抗、介电常数、衰减和材料分布。RG58 使用有效同轴 RLGC，Field 使用有效分布 RLGC。
- 每段先计算频率相关 R、L、G、C，再由 `Zc=sqrt(Z/Y)`、`gamma=sqrt(ZY)` 得到频变特性阻抗和传播常数，最后通过 ABCD 网络级联。
- 健康段的色散来自导体趋肤效应；介质损耗由 `omega*C*tan_delta` 表示。Debye 复介电常数仅用于老化和受潮区间。
- 接头使用局部串联阻抗与并联电容两端口建模，真值角色统一为 `joint`。
- `short` 为局部突变；`aging`、`moisture_local`、`moisture_distributed` 使用13至81段连续RLGC渐变，但每个物理区间只产生一条缺陷真值。
- 夹具只位于测量近端；VNA误差采用直接性、源匹配和反射跟踪模型。
- 1 GHz与200 MHz共享电缆、夹具和系统误差参数，仅测量噪声流独立。
- 不注入单条实测曲线，不复制实测相位或残差，不重构末端，不使用距离域目标反算。

## 3. 文件结构

```text
dataset.json
annotations/{train,val,test}.jsonl
frequency/{split}/{1ghz,200mhz}/{sample_id}.npz
responses/client_hann_v1/{split}/{band}/{sample_id}.npz
```

频域NPZ使用 `float64` 命名数组：`frequency_hz`、`s11_real`、`s11_imag`。空间域NPZ使用 `distance_m`、`impulse_real`、`impulse_imag`、`step`，并保存有效覆盖范围。

JSONL每条记录包含：样本与split信息、物理长度、参考epsr、终端、双频文件路径、完整生成参数和稀疏事件列表。事件角色为 `terminal | joint | defect`，同时保存物理起止位置、往返时延、严重度和电气变化。物理位置是权威真值；标量时延按100 MHz群时延定义，并由 `generation.truth_delay_reference_hz` 明确记录参考频率。

## 4. 空间响应

默认 `client_hann_v1` 使用Hann窗、Client DC外推和0.25 m重采样。每条样本使用自己的参考epsr，输出长度为 `min(1.2×物理长度, IFFT有效覆盖)`。覆盖范围之外禁止外推；若慢波缺陷使终端超出低频有效范围，使用 `terminal_observable=false` 明确记录。

## 5. 实测校准

实测S11只用于确定聚合参数范围和验收区间：频段幅值分位数、幅相斜率、重复测量噪声、频率相关尺度、校正残留和近端/末端关系。Core RG58用于材料核主验收，Shield只作辅助观察；实测信息不得作为单曲线模板回注。全量统计与多拓扑验证完成后再冻结正式 `empirical_v1.yaml`。当前 `provisional_rlgc_v1.yaml` 只能用于软件验证。

## 6. Train/Evaluate边界

本阶段的 `CableDataset` 返回NumPy频域数组、可选空间响应、稀疏真值和coverage，不做归一化、padding、Tensor转换或标签栅格化。未来训练代码必须在独立transform中决定输入通道和目标形式，不能修改或复制DG协议。
