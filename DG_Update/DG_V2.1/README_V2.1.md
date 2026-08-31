# DG 2.5 km 数据生成器 V2.1

V2.1 的硬约束是：**CSV 中的 S11 是唯一信号源，脉冲响应和阶跃响应必须由同一套 `core.tdr_signal.s11_to_responses()` 直接重算得到。** 所有夹具、背景起伏、受潮形态和末端增强都先编码到 S11，生成完成后不做距离域补峰、压峰或曲线改形。

## 入口

- 批量生成：`[V2.1]DG_dataset_max2.5km.py`
- 单样本 GUI：`DG_GUI_dataset_max2.5km.py`
- S11/IFFT：`core/tdr_signal.py`
- RLGC 传输线：`core/s11_generator.py`
- PyTorch 数据集：`core/dataset.py`
- 回归测试：`tests/test_v21_invariants.py`
- 固定案例验证：`tools/validate_v21.py`

## V2.1 重点变化

### RG58 首端

- 0 m 端口峰单独增强，随后连续衰减；
- 增强跨度采用“绝对距离 + 长度比例”混合约束：主要增益区约 4–9 m，完整纹理约 18–42 m；
- 避免短电缆按比例拉得过长，也避免长电缆固定 8.5 m 后突然截断；
- 接头和末端仍保持独立、因果，不会复制首端高频振荡。

### Field 现场电缆

- SMA—鳄鱼夹夹具改为直接在 S11 中生成的密集、非等间距、最小相位回波簇；
- 形成强 0–15 m 簇和弱 15–60 m 尾部，替代过于理想的低 Q 少峰模型；
- 引入几十至上百米相关长度的缓慢阶跃漂移，使健康电缆本体不再近似直线；
- 增加基于当前物理 S11 的幅值预算，前端增强不依赖最终 `|S11|=1.25` 硬截幅；
- 远端缺陷、接头与末端继续使用因果低通核，防止它们出现首端式高频振荡。

### 受潮

- 受潮段采用非对称渐变的 `epsr / tanδ / Debye / Z0` 参数，入口更明确、出口更弥散；
- 阶跃响应按三段规律约束：健康段缓变 → 受潮段下降更快 → 出水后保持低位并恢复为较小斜率；
- 自动测量并抵消湿区出口的非真实正向回弹，但不抹掉物理损耗与边界信息；
- 受潮导致的波速降低进入有效电气长度，末端抬升相对标称长度后移；
- 后方接头和末端反射因损耗增加而衰减。

## 运行

```bash
pip install -r requirements.txt
python "[V2.1]DG_dataset_max2.5km.py" \
  --output_dir ./DG_dataset_v21 \
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

运行测试与固定案例：

```bash
pytest -q
python tools/validate_v21.py --output_dir validation_output
```

## 输出与重构

每个频带 CSV 保存 `Frequency / S11_Real / S11_Imaginary / Distance / ImpulseResponse / StepResponse`。浮点数使用 17 位有效数字，YAML 保存完整 `epsr`。用 CSV 的 S11、YAML 的 `epsr`、相同窗函数重新执行共享 IFFT，距离、脉冲和阶跃应逐点完全一致。

## 实测模板原则

模板仅迁移平滑幅频包络、残差功率统计、频带能量尺度和夹具复杂度标量；不复制模板相位、复残差、接头位置或未知缺陷。模板按目标电缆长度加权选择，并经过点数、截幅、频谱病态和异常平坦度检查。
