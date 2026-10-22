# Core-16 现场探索版

保留 **16类机制 × 3条物理路线 × 2个响应算法**，开路主对照。独立于旧版Core-16，不改REF、GUI或DG V3。参数处于现场约束的探索阶段，尚未冻结为训练数据参数。

主要证据、统计口径和限制见[现场证据与实现说明](../../../AgentsDoc/2026-10-22_Core16现场版_现场证据与实现说明.md)。`evidence_notes.md`是TXT佐证盘点。

`scope_evidence.json`保留人工判读的长度范围及来源，供统计重跑使用；它不是波形数据集。当前材料拟合仍有约73–108m的末端峰位差，不能据此冻结训练参数。

## 生成

正式命令（CPU；默认600/1500/2400m、9kHz–200MHz；有逐组进度）：

```powershell


```

可加`--lengths 1500`选择长度，`--cases C00 C09 C14`选机制。配置是本目录`field_config.json`；`body.velocity_m_s`表示有效材料参数的传播速度，`response.velocity_m_s`表示响应图距离换算速度，两者含义不同。

小带宽预览使用独立输出，避免覆盖正式数据：

```powershell
& 'E:\Anaconda\envs\gpushare_cu124\python.exe' -X utf8 -B 'D:\GitRepository\Cable-NN\DifferentAgent\CodeX\Core16_Field\generate_field_core16.py' --lengths 600 --quick --output output/core16_quick --assets assets/core16_quick
```

`--quick`限制到20MHz，不能替代200MHz结果。`profile_summary.json`记录实际频率网格、采样步长、带宽分辨率近似、物理平滑宽度和耗时。S11、响应、truth及指标在`output/`；图板在`assets/`，均已忽略Git。

## 分析脚本

| 脚本 | 职责 |
|---|---|
| `analyze_corpus.py` | 内存解包、测量/短接分开统计、保留少量代表信号 |
| `summarize_scope.py` | 将文本长度佐证连接到曲线，分开短线/长线/未知 |
| `characterize_wet.py` | 1500m受控试验的差分衰减与展宽 |
| `fit_nearend.py` | 三个代表案例的短接与测量联合近端等效拟合 |
| `compare_anchor.py` | 同一2430m案例下双算法形态比较 |
| `fit_body_loss.py` | 该案例的有效端到端材料参数拟合；不作独立验证 |

上述拟合不唯一，也不是每一类机制的现场确诊。未知测量噪声未伪造为已识别噪声；没有将实测波形作为生成器模板。

`output/signals/`来自原库抽取，`vendor_*`是厂商已存响应；本项目A1/A2另算，二者不混称。源`E:\20261010截至的全部fdr文件`完全只读，禁止运行会向该目录导出CSV的原转换器命令行入口。
