# CST_Reproduction V1 执行结果

## 执行范围

本轮在 `main` 工作树下完成 `DifferentAgent/CodeX/CST_Reproduction/V1`。V0、CST 工程、REF 两套算法和 `E:\FDR案例-csv` 原始数据均未修改。V1只改进 CST 等效电路复现程序，不建立 Field/YJV 参数模型。

正式命令：

```powershell
E:\Anaconda\envs\gpushare_cu124\python.exe .\cst_fdr_reproduction.py --chunk-size 256
```

频率网格为 9 kHz–200 MHz、10000点；算法2为 REF 默认参数、10000时间点。程序输出进度条，并在最终结果中加入了小规模 Algorithm 2/REF 核心逐数组对照。

## 实现内容

- 固定 RLGC 梯形：0.4/0.2/0.1/0.05 m四种单元尺寸；0.1 m为默认主模型。
- 连续固定 RLGC 参考模型，用于识别离散单元色散，不作为严格 CST 复现。
- DG V3 风格有效 RLGC 频变损耗对照，100 MHz处锚定V0参数，不作为现场电缆标定。
- 保持50 Ω端口、1 m/135 Ω TL1、0.5 m/300 Ω TL3/TL4、特殊RLCG接头、10/20/30 m边界和末端开路。
- 15 m缺陷固定为14.8–15.2 m，细分网格覆盖单元数为1/2/4/8。
- 接头消融图现在包含完整0.1 m模型、去特殊接头、去300 Ω TL、去全部内部接头。
- S11统一保存三列 `Frequency_Hz,S11_Real,S11_Imag`，不按算法复制S11。
- 图像距离轴使用0–60 m；报告 CST 图保持原始纵横比并等比例放大。

## 验收结果

| 项目 | 结果 |
|---|---|
| S11输出 | 34个CSV、每个10000行、三列表头统一、频率递增、无NaN/Inf |
| 距离域输出 | 102个CSV，算法1和算法2输出分离，数组有限 |
| R/L/G/C守恒 | 四种细分总量相对误差约1e-14；40 m总量为R=1 Ω、L=30 μH、G=0.0005 S、C=1.6 nF |
| 0.1 m基准尾波 | 算法1=0.003623、算法2=0.036294，均不高于0.05 |
| 离散截止 | 0.4 m约145.288 MHz；0.1 m约581.152 MHz；0.05 m约1162.303 MHz |
| Algorithm 2/REF | 256个跳过首点频率、256个时间点的逐数组最大绝对误差为0 |
| RG58局部性 | CutPVC：算法1/2=49.233/49.162 m；guhua：49.183/49.148 m |

## 必须保留的限制

1. 0.1 m和0.05 m相对连续固定 RLGC 的全频/全距离域 RMS 仍未达到预设0.05阈值；只能说细分显著降低了0.4 m离散伪影，不能说已完全收敛。
2. 当前电路本征速度因子约0.609，而两套 FDR 使用约0.669/0.6715的位置标尺。因此默认基准事件按FDR坐标约位于11.5/23.8/35.4/46.7 m附近，而不是严格10/20/30/40 m；报告已将其记录为速度标尺差异，未强行调参。
3. 现场参考数据只用于外部现象检查。145 MHz前后四条现场曲线的幅值变化不统一，不能推出现场电缆共有145 MHz截止，也不能从未校准数据反推通用YJV RLGC。
4. 没有 CST ASCII S11 数值导出，因此没有报告 CST逐点 RMSE或“完全一致”。

## 结果入口

- 主报告：`DifferentAgent/CodeX/CST_Reproduction/V1/comparison_report.md`
- 主程序：`DifferentAgent/CodeX/CST_Reproduction/V1/cst_fdr_reproduction.py`
- 图像：`DifferentAgent/CodeX/CST_Reproduction/V1/assets`
- S11：`DifferentAgent/CodeX/CST_Reproduction/V1/s11_output`
- 距离域和指标：`DifferentAgent/CodeX/CST_Reproduction/V1/output`
