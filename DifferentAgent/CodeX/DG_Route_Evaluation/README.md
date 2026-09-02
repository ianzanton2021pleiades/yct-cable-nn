# DG Route Evaluation

本目录独立评估 DG V3、CST V0、CST V1 的电缆建模路线。现有生成器和 `CST_Reproduction` 均作为只读来源，不在本目录之外写入结果。

## 候选

- `dg_v3_rlgc`：DG V3 频变连续 RLGC 电缆体，接入与 CST 相同的受控接头网络。
- `cst_v0_ladder_0p4m`：报告原始 0.4 m 梯形单元。
- `cst_v1_ladder_0p1m`：V1 默认 0.1 m 梯形单元。
- `cst_v1_continuous`：V1 连续固定 RLGC 参考。
- `v27_reference`：只引用 V2.7 设计和已有验证结果，不参与胜负。

## 运行

```powershell
E:\Anaconda\envs\gpushare_cu124\python.exe -B .\compare_dg_routes.py
```

默认生成9个工况的清洁内核与统一测量链结果、两套FDR、指标CSV/JSON、工况对照板和中文报告。可用 `--smoke` 进行小规模流程检查；正式结果不要使用smoke指标。

重新生成实测聚合与DG V3原生分布门槛：

```powershell
E:\Anaconda\envs\gpushare_cu124\python.exe -B ..\..\..\DG_Update\DG_V3\calibrate_real_data.py `
  --input-root "E:\FDR案例-csv" --output-dir .\real_data_calibration --format both
E:\Anaconda\envs\gpushare_cu124\python.exe -B .\evaluate_real_distribution.py
```

## 输出

```text
output/metrics.csv
output/summary.json
output/s11/<layer>/<candidate>/s11_<case>.csv
output/responses/<layer>/<candidate>/...
output/distribution_gate.json
assets/comparison_board_<layer>_<case>.png
assets/distribution_gate.png
real_data_calibration/calibration_summary.json
DG路线对比评估报告.md
```
