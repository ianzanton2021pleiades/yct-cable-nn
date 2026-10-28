# DG 2.5 km 数据生成器 V2.2

V2.2 继续保持核心不变量：**CSV 中的 S11 是唯一信号源**。夹具、端口振荡、接头、本体微小失配、受潮形态和末端响应均先编码到 S11，脉冲响应与阶跃响应只由 `core.tdr_signal.s11_to_responses()` 重构，不做距离域后处理。

## 工程入口

- 批量生成：`[V2.2]DG_dataset_max2.5km.py`
- 单样本 GUI：`DG_GUI_dataset_max2.5km.py`
- 传输线模型：`core/s11_generator.py`
- IFFT 与响应：`core/tdr_signal.py`
- 数据集加载：`core/dataset.py`
- 回归测试：`tests/test_v22_invariants.py`
- 固定案例验证：`tools/validate_v22.py`

## V2.2 关键修复

### RG58

- 0 m 端口高频簇增强，并在最终末端幅值确定后校准端口/末端关系；校准窗仅覆盖 0–2.2 m，不会把约 5 m 的接头误认为端口。
- 新增全长本体微失配纹理：幅值连续、缓慢衰减，末端保留非零底噪，不再在固定距离突然消失。
- 数值分段与物理接头彻底解耦。缺陷边界不再自动显示为接头。
- 随机接头数量随长度变化：100 m 及以下最多 4 个，100 m 以上最多 5 个；随机接头间距不少于 5 m，并避开缺陷邻域。

### RG58 / Field 接头

- 接头由“单个正峰或负峰”改为有限宽度的双极性响应：主正峰、负向恢复峰和极弱回弹。
- Field 接头随距离增大而衰减、展宽；前部接头更明显，后部接头更弱。
- 接头幅值按当前样本的首端和末端响应自适应标定，避免 Field 接头长期淹没在基线中。

### Field 首端

- 保留 0–约 15 m 的密集夹具振荡和短过渡段。
- 删除原先延伸至几十米的稀疏、单极性小回波，尤其避免 Field-short 场景在首端后方出现无意义孤立峰。

## 运行

```bash
pip install -r requirements.txt
python "[V2.2]DG_dataset_max2.5km.py" \
  --output_dir ./DG_dataset_v22 \
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
pytest -q
python tools/validate_v22.py --output_dir validation_output
```

## 输出一致性

每个频带 CSV 保存 `Frequency / S11_Real / S11_Imaginary / Distance / ImpulseResponse / StepResponse`，浮点数使用 17 位有效数字。使用 CSV 中的 S11、YAML 中的 `epsr` 和相同窗函数重算时，距离、脉冲和阶跃应逐点一致。
