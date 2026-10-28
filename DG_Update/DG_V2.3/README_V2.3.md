# DG 2.5 km 数据生成器 V2.3

V2.3 保持核心不变量：**CSV 中的 S11 是唯一信号源**。夹具、端口振荡、接头、本体微失配、受潮和末端响应全部先编码到 S11；脉冲响应与阶跃响应仅由 `core.tdr_signal.s11_to_responses()` 重构，不进行距离域结果篡改。

## 工程入口

- 批量生成：`[V2.3]DG_dataset_max2.5km.py`
- 单样本 GUI：`DG_GUI_dataset_max2.5km.py`
- 传输线模型：`core/s11_generator.py`
- IFFT：`core/tdr_signal.py`
- 数据集加载：`core/dataset.py`
- 回归测试：`tests/test_v23_invariants.py`
- 固定案例验证：`tools/validate_v23.py`

## V2.3 重点改进

### RG58

- 保留 V2.2 的首端和全长基线模型。
- 正常接头幅值相对本体基线小幅提高，并继续随传播距离、前置受潮而衰减。
- 正负接头峰的完整有效支撑严格限制在 1 m 内。

### Field 本体与 S11

- 新增全长相关微失配纹理：首端后连续存在、缓慢衰减、远端保留非零底值；受潮后平滑减弱，不发生突然截断。
- Field 夹具改为少数相干短时延主模态、弱邻近模态和低能量密集回波。S11 实部/虚部呈连续载波及平滑频率包络，不再是宽频随机游走。
- 长电缆本体返回作为夹具载波上的较小快速纹波，并保留随频率升高的能量衰减。

### 正常接头与受潮接头

- 正常 Field 接头采用“正峰更窄更高、负峰更宽更低”的非镜像结构；负峰宽度约为正峰的 1.85–2.25 倍，幅值比例随机化。
- Field 接头宽度同时依赖电缆总长和相对位置，越远越宽；完整支撑不超过总长的 1.5%。
- 前置受潮会削弱后续正常接头。
- 局部受潮接头采用负极性主峰、弱正向前兆和宽负向尾部；随距离增加幅值下降、宽度增加，并由独立 S11 湿区核产生持续向下的阶跃变化。

## 运行

```bash
pip install -r requirements.txt
python "[V2.3]DG_dataset_max2.5km.py" \
  --output_dir ./DG_dataset_v23 \
  --n_total 3000 \
  --workers 8 \
  --seed 20260705 \
  --profile mixed \
  --real_data_root "E:/FDR案例-csv"
```

GUI：

```bash
python DG_GUI_dataset_max2.5km.py
```

测试与固定案例：

```bash
pytest -q
python tools/validate_v23.py --output_dir validation_output
```

## 输出一致性

CSV 使用 17 位有效数字保存 `Frequency / S11_Real / S11_Imaginary / Distance / ImpulseResponse / StepResponse`。使用 CSV S11、YAML 中的 `epsr` 与同一窗函数重算时，距离、脉冲和阶跃应逐点一致。
