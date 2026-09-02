# CST_Reproduction V1

V1 是报告第三章 CST 等效电路的改进实验程序。V0 保持冻结；V1 只改变电路实现的单元离散化和对照组织，不修改 CST 工程、REF 算法或 `E:\FDR案例-csv` 原始数据。

## 默认短时验证

```powershell
E:\Anaconda\envs\gpushare_cu124\python.exe .\cst_fdr_reproduction.py --smoke
```

`--smoke` 使用300个频率点和300个算法2时间点，只用于检查程序流程，不代表正式波形和指标。

## 正式运行

```powershell
E:\Anaconda\envs\gpushare_cu124\python.exe .\cst_fdr_reproduction.py --chunk-size 128
```

默认 `--model all` 会运行：

- 0.4/0.2/0.1/0.05 m固定 RLGC 梯形基准收敛对照；
- 0.1 m固定梯形的全部9个工况；
- 连续固定 RLGC 参考的全部9个工况；
- 0.1 m频变损耗对照的全部9个工况；
- 特殊接头、300 Ω TL和内部接头消融基准；
- 现场幅频/相频参考和 RG58 缺陷局部性参考。

正式计算量较大，程序会为算法2每个模型/工况输出进度条。

## 模型口径

固定梯形细分按 `cell_length / 0.4` 等比例缩放 R/L/G/C，保持40 m主体和10/20/30 m接头位置不变。15 m缺陷保持 `14.8–15.2 m` 物理范围；0.1 m模型对应4个细分单元。

连续模型仅作固定 RLGC 的数值参考，不等同于 CST 原始梯形工程。频变损耗模型只作独立对照，100 MHz处锚定 V0 参数。

两套 FDR 均保持当前 REF 参数。S11 是唯一频域源，不按算法拆分，CSV表头固定为：

```text
Frequency_Hz,S11_Real,S11_Imag
```

## 输出

```text
assets/                         对照图、消融图、现场参考图
s11_output/<model>/             三列 S11 CSV
output/<model>/                 算法1、算法2距离域CSV
output/model_summary.csv        模型和震荡指标
output/resolution_convergence.csv
output/rg58_reference_metrics.csv
output/summary.json
comparison_report.md             中文分析报告
```

现场数据只读使用；未标注现场文件不会被自动标成 YJV。当前没有 CST ASCII S11 导出，因此报告不计算 CST 逐点 RMSE 或“完全一致”。
