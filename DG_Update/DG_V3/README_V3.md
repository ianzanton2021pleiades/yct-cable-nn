# DG V3

DG V3 是面向 2.5 km 以内 RG58 与现场电力电缆的断代数据生成器。它只从传输线、接头、夹具、VNA误差和随机测量噪声直接生成复数 S11，不读取曲线模板，不通过 IFFT 反算或修改 S11。

## 当前状态

默认参数档为 `configs/provisional_rlgc_v1.yaml`。它使用 RLGC 材料核，但参数仍只用于代码、物理和协议验证，不是最终训练参数。原 `provisional_v1.yaml` 保留为旧经验材料核的历史对照，不提供兼容执行路径。必须先完成多组实测验收、审查输出并冻结 `empirical_v1.yaml`，才能生成正式训练数据。

RG58 由目标 Z0、epsr 和 100 MHz 衰减反求有效同轴几何；Field 使用不依赖虚构同轴尺寸的有效分布 RLGC。两者均从一次参数统一计算复数特性阻抗与传播常数。健康段使用趋肤损耗和介质损耗角，Debye 色散仅用于老化和受潮区域。当前低频导体模型仍保持 DG 2.7 的趋肤外推形式，尚未加入 Rdc 过渡。

## 数据生成

```powershell
python DG_Update/DG_V3/generate_dataset.py `
  --n-total 3000 `
  --workers 8 `
  --seed 20260831 `
  --profile mixed `
  --config DG_Update/DG_V3/configs/empirical_v1.yaml `
  --output DataSet/DG_V3
```

输出两批同拓扑扫频：

- `1ghz`: 9 kHz–1 GHz，50000点；
- `200mhz`: 9 kHz–200 MHz，6250点。

每条样本的真值写入对应 split 的 JSONL，包含 terminal、joint、defect 的物理位置、传播时延和电气参数变化。

## 派生响应与检查

```powershell
python DG_Update/DG_V3/build_responses.py DataSet/DG_V3
python DG_Update/DG_V3/validate_dataset.py DataSet/DG_V3
python DG_Update/DG_V3/validate_fixed_cases.py --output AgentsStorage/DG_V3_fixed_validation
python DG_Update/DG_V3/validate_rlgc_real.py --input-root E:\FDR案例-csv --output AgentsStorage/DG_V3_RLGC_validation
python DG_Update/DG_V3/dg_gui.py
```

空间响应采用 `client_hann_v1`，是可以删除并从 S11 重建的缓存，不属于 DG 权威数据。DG 不使用实测模板、末端重构、强制可见度或 IFFT 反投影修改 S11。

## 实测统计校准

```powershell
python DG_Update/DG_V3/calibrate_real_data.py `
  --input-root E:\FDR案例-csv `
  --output-dir AgentsStorage/DG_V3_calibration `
  --format both
```

校准工具排除 `IFFT现场数据汇总` 和标称超过2500 m的Field案例，只输出聚合统计，不输出单曲线模板或残差。

## 测试

```powershell
python -B -m unittest discover -s DG_Update/DG_V3/tests -p "test_*.py" -v
```
