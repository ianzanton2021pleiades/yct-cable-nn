# DG 2.5 km 数据生成器 V2

V2 的首要约束是：**CSV 中的 S11 是唯一信号源，脉冲响应和阶跃响应必须由同一套 `core.tdr_signal.s11_to_responses()` 直接重算得到。** 生成完成后不再允许任何距离域补峰、压峰、下沉或斜率修正。

## 主要入口

- 批量生成：`[V2]DG_dataset_max2.5km.py`
- 单样本分析 GUI：`DG_GUI_dataset_max2.5km.py`
- PyTorch 数据集：`core/dataset.py`
- 传输线/RLGC：`core/s11_generator.py`
- S11 → IFFT：`core/tdr_signal.py`
- 回归测试：`tests/test_v2_invariants.py`

## 安装依赖

```bash
pip install -r requirements.txt
```

Tkinter 通常随 Windows Python 安装，用于 GUI。

## 快速运行

```bash
python "[V2]DG_dataset_max2.5km.py" \
  --output_dir ./DG_dataset_v2 \
  --n_total 3000 \
  --workers 8 \
  --seed 20260705 \
  --profile mixed \
  --real_data_root "E:/FDR案例-csv"
```

启动 GUI：

```bash
python DG_GUI_dataset_max2.5km.py
```

GUI 可固定随机种子、开关实测模板并修改实测数据目录，便于对同一物理参数做可复现 A/B 对比。

运行测试：

```bash
pytest -q
```

生成固定回归图和指标：

```bash
python tools/validate_v2.py --output_dir validation_output
```

## 输出一致性

每个频带 CSV 包含：

- `Frequency`
- `S11_Real`
- `S11_Imaginary`
- `Distance`
- `ImpulseResponse`
- `StepResponse`

V2 使用 17 位有效数字保存浮点数，并在 YAML 中保存未截断的 `epsr`。用 CSV 的 S11、YAML 的 `epsr` 和相同窗函数重新执行 `s11_to_responses()`，结果应与 CSV 距离域列逐点完全相同。

## 模板使用原则

实测模板仍然保留，但只迁移：

1. 平滑的幅频包络；
2. 快速残差的局部功率谱统计；
3. 各频带能量尺度；
4. 与目标电缆长度相近的模板优先。

V2 **不复制模板复残差、不复制模板相位、不复制模板中的接头或未知缺陷位置**。残差噪声由新的独立随机过程生成。校正数据也不再作为复波形直接叠加，只用于估计夹具复杂程度。

## 物理模型变化

- 分布参数模型加入介质损耗角和 Debye 型介电色散；
- 老化与受潮通过平滑渐变的 `Z0 / epsr / tanδ / Debye` 参数进入 S11；
- 长缺陷边缘使用 raised-cosine 过渡，减少不真实的硬边界高频振铃；
- 末端开路改为真正的高阻负载；
- 必要的末端可见性增强通过因果、最小相位的 S11 核完成；
- RG58 首端使用局部化、密集、衰减的高频纹理，而不是少量离散反射点；
- Field 夹具、接头和末端使用因果低通或低 Q 模式，避免远端事件两侧出现非因果高频振荡。

## 训练加载器

`core.dataset.CableDefectDataset` 默认网格与 V2 标签一致：

- 最大距离：2500 m
- 步长：0.25 m
- 点数：10000

加载器按照 manifest 中的路径读取 1 GHz 或 200 MHz CSV，并从 S11 重新计算输入响应。距离平移增强会同时平移输入和标签，避免监督错位。

## 版本定位

V2 解决的是 V1 中已经明确的结构性错误，并建立可自动回归的信号一致性约束。它不是最终的电缆参数反演模型。真实电缆类型、VNA 夹具族、温度、接地回路和测试人员操作仍可进一步分层建模。
