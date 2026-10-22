# Core-16 DG路线扩展评估

本目录是 `DifferentAgent/CodeX/DG_Route_Evaluation` 的独立扩展实验，专门把原报告仿真章节的9工况扩展为 Core-16。脚本只读取原 CST V1、DG V3 和 FDR 实现；输出落在本目录的 `output/`、`assets/`，不会写入原评估目录。

## 固定协议

- 路线固定为 `dg_v3_rlgc`、`cst_v1_ladder_0p1m`、`cst_v1_continuous`。
- 40 m主体按0–10、10–20、20–30、30–40 m分段；附件、终端、测量链和两套FDR算法保持原评估协议。
- 频率网格为9 kHz–200 MHz、10000个等间隔点。
- Algorithm 2 使用10000时间点、`chunk_size=256`；60 m是当前V1对照图和指标的距离上限。Algorithm 2核心函数直接调用原V1实现，没有改算法参数。
- measured 层的测量链固定为 `seed=20260902`、`delay_ns=1.35`、`loss_db_at_100mhz=0.12`、`directivity=0.0022`、`source_match=0.009`、`tracking=0.985`、`tracking_phase_rad=0.026`、`noise_sigma=0.0014`、`noise_correlation=0.965`。每个工况使用 `seed + 工况序号`，三条路线对同一工况共用该测量噪声种子。

## 参数边界

`core16_cases.yaml` 是 Core-16 数值的唯一来源，表中 RLGC 全部表示每0.4 m等效单元值。

- CST V1 0.1 m梯形严格使用 `scale=0.1/0.4=0.25` 缩放 R/L/G/C。
- CST V1连续路线先换算成单位长度RLGC。
- DG V3使用现有 `effective_rlgc` 方式，在100 MHz锚定每米 R/L/G/C；不把DG改成CST固定模型。
- 前置1.0 m、135 Ω无损线，10 m `special_joint1`，20/30 m各0.5 m、300 Ω无损线，40 m开路及末端附加支路均由协议配置固定。
- C05的30–40 m区段为G=500 μS/0.4 m，且末端属于第四段，所以末端附加并联支路按原 `segmented_loss` 语义使用G=500 μS、C=16 pF。其他工况末端保持G=5 μS、C=16 pF。
- C01、C02、C04、C07、C08、C09、C11、C12的局部区间均为14.8–15.2 m；C03、C06、C10均为10–20 m；C13–C15均只在20–30 m替换主体电缆，20 m和30 m附件保留。

## 用户执行

正式运行请使用已配置的 Python 3.11 环境：

```powershell
E:\Anaconda\envs\gpushare_cu124\python.exe -B .\compare_core16_routes.py
```

正式结果包括：

```text
output/metrics.csv
output/metrics.json
output/summary.json
output/s11/<clean|measured>/<route>/s11_C00.csv ... s11_C15.csv
output/responses/<clean|measured>/<route>/response_C00.npz ... response_C15.npz
assets/comparison_board_<clean|measured>_C00.png ... C15.png
Core-16路线对比评估报告.md
```

如需先做短时流程检查，使用独立输出目录，避免把冒烟结果混入正式结果：

```powershell
E:\Anaconda\envs\gpushare_cu124\python.exe -B .\compare_core16_routes.py `
  --smoke --output .\smoke_output --assets .\smoke_assets `
  --report .\smoke_Core-16路线对比评估报告.md
```

脚本不自动把未经用户审阅的物理方向设为硬门槛；`numeric_gate_pass`只表示响应有限性和 clean 层被动性。具体方向、幅度和三条路线差异应结合 `metrics.csv`、S11 CSV及对照图判断。
