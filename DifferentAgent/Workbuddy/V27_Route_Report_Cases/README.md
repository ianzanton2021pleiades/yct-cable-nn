# V27_Route_Report_Cases

用 DG V2.7 的 RG58 / Field 两条生成路线还原 HumanDoc 报告仿真章节的 9 个工况
（基准 / 整体C↑ / 整体G↑ / 分段损耗 / 局部C↑ / 局部G↑ / 局部C↓ / 局部R→10Ω / 局部R→50Ω），
生成对应的 S 参数（S11 CSV）与算法1/算法2 分析对比图。

## 运行

```bash
"E:\Anaconda\envs\gpushare_cu124\python.exe" compare_v27_routes.py            # 正式
"E:\Anaconda\envs\gpushare_cu124\python.exe" compare_v27_routes.py --smoke    # 冒烟（600点）
```

## 结论速览

- **DG V2.7 是基于 RLGC 生成的**：`DG_V2.7/core/s11_generator.py::_compute_s11_for_cable`
  由 (Z0, epsr, alpha@100MHz) 反解同轴几何，按频率构建 R(趋肤,∝√f)、L、C(复数,含Debye)、
  G(σ介 + ωC·tanδ)，逐段电报方程级联递推输入阻抗得到 S11。
- 每条路线 = V2.7 RLGC 传播核 + 报告 CST 场景结构（TL1、10 m 特殊接头、20/30 m 后
  300Ω×0.5 m 线、末端 16 pF‖200 kΩ），组装方式与 CodeX/DG_Route_Evaluation 的
  dg_v3 候选同构；所有参考路径只读。

## 文件

| 文件 | 说明 |
|---|---|
| `compare_v27_routes.py` | 主脚本（工况映射、V2.7 段 ABCD、统一测量层、指标、画板） |
| `evaluation_cases.yaml` | 频率网格、测量层参数、两条路线的基准介质参数 |
| `output/s11/<layer>/<route>/` | 每工况 S11 CSV（Frequency_Hz, S11_Real, S11_Imag） |
| `output/responses/<layer>/<route>/` | 算法1/算法2 距离域响应 npz |
| `output/metrics.csv` / `metrics.json` | 逐工况定量指标与方向性布尔检查 |
| `assets/comparison_board_*.png` | 18 张工况对比板（2 层 × 9 工况，含报告对应 CST 图） |
| `V27路线还原对比报告.md` | 汇总报告（含 V2.7 RLGC 结论、映射表、已知差异） |

## 已知差异（路线表达能力的边界）

1. V2.7 并联损耗用 tanδ（G∝f）表示，报告网表为常数 G，100 MHz 锚定后低频行为不同。
2. local_R 工况所需的常数串联 R 不在 V2.7 参数空间内（其导体损耗为趋肤 R∝√f），
   以最小扩展方式注入该段单位长度串联阻抗。
3. local_C4pF 要求 epsr<1，物理同轴不可实现，V2.7 内部下限 1.05 使该工况欠强。
4. RG58 路线自带的 BNC 接头随机反射与噪声层为保证确定性对比未启用（等价 clean 层口径）。
