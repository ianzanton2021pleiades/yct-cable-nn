# DG 2.5 km 数据生成器 V2.3（Field 可见度修订版）

本版保持核心不变量：**CSV 中的 S11 是唯一信号源**。夹具、首端振荡、全长基线、接头、缺陷、受潮和末端反射均先编码到 S11，脉冲响应与阶跃响应只由 `core.tdr_signal.s11_to_responses()` 重构，不对最终距离域曲线做人工覆盖。

## 工程入口

- 批量生成：`[V2.3]DG_dataset_max2.5km.py`
- 单样本 GUI：`DG_GUI_dataset_max2.5km.py`
- 传输线模型：`core/s11_generator.py`
- IFFT：`core/tdr_signal.py`
- 回归测试：`tests/test_v23_invariants.py`
- 固定案例验证：`tools/validate_v23.py`

## 本次修订重点

### Field 可见度链路

- 修正原 V2.3 `Field cable visibility transfer` 对电缆本体回波抑制过强的问题；保留平滑、相干、低频占优的实测式 S11 形态，但不再牺牲首端、缺陷和末端。
- 首端使用连续载波、宽带直达项、0–18 m 密集交替回波及短过渡尾部，恢复现场常见的高频振荡簇，同时避免几十米范围内稀疏单极性杂峰。
- 末端采用基于首端稳健峰值、全长基线 RMS、长度、终端类型及前置损耗的因果校准；结构性末端不再被当作噪声预算压缩。
- Field 基线保留快慢两种相关尺度和非零远端底值；接头仍为非对称双极峰，完整宽度不超过总长 1.5%。

### 缺陷与受潮

- `short`、`aging` 增加独立的因果 S11 可见度核，幅值相对局部基线、首端及末端自适应标定，仍随距离和前置损耗衰减。
- 分布受潮增强 C/G 变化对应的持续下降与斜率变化；局部受潮继续采用负极性主峰、弱正向前兆、宽负向尾部及湿区后低阻抗残留。
- 新增 Field 缺陷：
  - `capacitance_high`：局部 C/介电常数增加，入口负极性、有效长度增加；
  - `capacitance_low`：气隙/脱层/干裂类 C 降低，入口正极性、有效长度缩短；
  - `loss_local`：局部 G/介损增加，缺陷后阶跃下移、后续事件衰减；
  - `resistance_high`：屏蔽腐蚀/断丝/压接异常类串联 R 增加，缺陷后阶跃上移、后续事件衰减。

GUI 已加入全部新缺陷类型及常用组合。

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

测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_v23_invariants.py
```

当前 24 项回归测试覆盖 S11/IFFT 唯一来源、CSV 往返一致性、首端密集振荡、Field S11 连续性、末端可见度、缺陷相对基线可见度、接头宽度、受潮持续下降及扩展 RLGC 缺陷方向。
