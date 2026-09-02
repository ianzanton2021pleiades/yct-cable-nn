# CST_Reproduction

这是报告第三章《电缆缺陷仿真》的单文件 Python 复现程序。

## 短时验证

```powershell
E:\Anaconda\envs\gpushare_cu124\python.exe .\cst_fdr_reproduction.py --smoke --output .\smoke_output
```

## 正式运行

```powershell
E:\Anaconda\envs\gpushare_cu124\python.exe .\cst_fdr_reproduction.py
```

正式运行包含9个工况。算法2逐项采用 REF GUI 默认值：电缆长度95 m、VF=0.6715、阶跃/脉冲平滑5/50点、时间点10000、测试线校正0、阶跃偏置0、脉冲归一化系数6.5、测试电压10 V、参考阻抗50 Ω、无频率筛选、无下采样、忽略首个测量点。算法2使用 REF 源码的 `3.0e8 m/s`，程序使用 NumPy 分块积分以控制内存，计算公式不变；与 REF `compute_response` 的完整字段对比保持浮点误差范围。

默认输出：`assets/` 保存9个完整复现对照板、9个局部性差分诊断板和报告参考图；完整对照板第一行包含线性频率轴的 S11 实部、幅值和 `[-180°,180°]` 包裹相位，第二/三行包含算法1/算法2的阶跃和脉冲，第四行仅以原始纵横比居中显示报告对应 CST 图。距离轴统一为0–60 m（40 m电缆本体的1.5倍），确保整体电容增加后的迟到终端仍可见。

`s11_output/` 保存9个可直接交给 REF 两个算法读取的 S11 CSV，统一为实测目录常用的三列表头 `Frequency_Hz,S11_Real,S11_Imag`；`output/` 保存同一份 S11 分别经算法1/算法2得到的派生结果和 `summary.json`，不是两套不同的 S11；根目录 `comparison_report.md` 引用全部图像并记录各区间 RMS 差分。

`--smoke` 仅用于短时验证，会把频率点和算法2时间点临时改为300，不代表正式结果。三个输出目录已加入本试验目录的 `.gitignore`，避免生成数据进入 Git 更新列表。

当前没有 CST 的 ASCII S11 导出文件，因此报告只做报告内嵌图的定性对照，不报告 CST 逐点误差、RMSE 或“完全一致”。
