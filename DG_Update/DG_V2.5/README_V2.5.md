# DG 2.5 km 数据生成器 V2.5

V2.5 继续保持核心不变量：**CSV 中的复数 S11 是唯一信号源**。首端、夹具、电缆本体、基线微失配、接头、缺陷、受潮和末端均在频域中合成，脉冲响应与阶跃响应只由 `core.tdr_signal.s11_to_responses()` 统一重构，禁止对最终距离域结果做人工覆盖。

## 工程入口

- 批量生成：`[V2.5]DG_dataset_max2.5km.py`
- 单样本 GUI：`DG_GUI_dataset_max2.5km.py`
- 传输线模型：`core/s11_generator.py`
- 公共 IFFT：`core/tdr_signal.py`
- 回归测试：`tests/test_v25_invariants.py`
- 固定案例验证：`tools/validate_v25.py`

## V2.5 核心修订

### 1. Field S11 与 IFFT 解耦失败的根治

- 先在无末端的匹配负载骨架上完成夹具、传播、缺陷、受潮、接头和基线建模，再在所有传播效应之后加入**唯一末端支路**。
- Field 可见度传递函数保留连续载波、低频较强和高频渐降的现场式 S11 包络，同时保留足够高频分量，使首端、缺陷和末端在同一 IFFT 中可见。
- 不再通过后处理“修图”，因此 S11 实部、虚部与导出的脉冲/阶跃严格一致。

### 2. 单一非对称末端响应

- 开路与短路末端采用同一延时、同一极性的双时间尺度因果核：上升沿窄，下降沿宽。
- 短 Field 电缆末端峰值提高；长电缆及受潮后末端峰值按传播损耗降低，同时下降尾部自动展宽。
- 删除原始物理末端后再加入一个结构末端，从机制上消除“针状峰 + 宽尾峰”或相邻双峰。

### 3. Field 基线与受潮阶跃稳定性

- 基线先生成有界、缓慢变化的阶跃纹理，再求导写回 S11，避免微小脉冲偏置积分成巨大的阶跃漂移。
- 分布受潮在湿区内形成平滑持续下降，湿区后保持较低平台；出口不再强制恢复到原基线。
- 修正湿区锚点连续性，消除受潮起止位置之外的伪脉冲和完好区大幅波动。

### 4. 缺陷与接头

- 保留并加强 `short`、`aging`、`moisture_local`、`moisture_distributed`。
- 支持 `capacitance_high`、`capacitance_low`、`loss_local`、`resistance_high` 四类 RLGC 方向性缺陷。
- Field 正常接头继续采用窄高正峰和宽低负峰；RG58 保留亚米级接头、全长微弱基线和混合衰减包络。

## 运行

```bash
pip install -r requirements.txt
python "[V2.5]DG_dataset_max2.5km.py" --output_dir ./DG_dataset_v25 --n_total 3000 --workers 8 --seed 20260706 --profile mixed --real_data_root "E:/FDR案例-csv"
```

GUI：

```bash
python DG_GUI_dataset_max2.5km.py
```

测试与固定案例验证：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
python tools/validate_v25.py --output_dir ./validation_output
```

当前回归集共 25 项，覆盖 S11/IFFT 一致性、CSV 往返、Field S11 连续性、首端振荡、单末端峰、末端非对称宽度、受潮持续下降、完好区阶跃稳定、接头宽度及扩展缺陷方向。
