# 邦润C相数据说明

**读取对象：** E:\20261010截至的全部fdr文件\2025-7-7 11.55.08邦润线(浸水)\邦润线2025-7-7 11.55.08.fdr  
**方法：** 使用项目现有 DifferentAgent/Zcode/DG/dg_v4/fdr_io.py 解密到内存SQLite，只读查询 Sweeps 和相应测量表；未写回原始FDR。

| Sweeps rowid | SweepName | S11表 | 采样点 | 频段 | Real / Image有效非空值 | 全零 | 厂商响应表 |
|---:|---|---|---:|---|---|---|---|
| 3 | C相 | MeasureData_3 | 50,000 | 9 kHz–1 GHz | 各50,000；各50,000个非零值 | 否 | ImpulseResponse_3、StepResponse_3缺失 |
| 4 | 新的测试_4 | MeasureData_4 | 50,000 | 9 kHz–1 GHz | 各50,000；各50,000个非零值 | 否 | ImpulseResponse_4 19,999行；StepResponse_4 20,000行 |

C相初测的原始S11完整存在。项目已有 DifferentAgent/Zcode/DG/dg_v4/transform.py 中的 s11_to_responses 统一变换，可从S11的实部重建派生脉冲和阶跃响应。此处只确认输入数据和厂商表状态，没有生成曲线。

rowid=4的原始 SweepName 是“新的测试_4”，相位未知；不能把它当作C相复测或C相替代输入。C相实际监督待主流程从rowid=3 S11重建响应并完成审阅后决定。2026-11-10用户复核确认的三相420m局部异常作为独立标签来源保留，不因C相厂商响应表缺失而撤销。
