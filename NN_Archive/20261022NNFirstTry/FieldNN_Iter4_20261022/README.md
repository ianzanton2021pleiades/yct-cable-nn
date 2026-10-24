> 已归档，现存权重、结果与复现缺口以 [归档索引](../README.md) 为准；下面保留实验原说明。

# Iter4：宽弱反射与夹具适应

直接启动正式一轮：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File "D:\GitRepository\Cable-NN\NN_Archive\20261022NNFirstTry\FieldNN_Iter4_20261022\run_adaptation.ps1"
```

默认复用Iter3的12k S11，构建raw缓存与6来源真实校准扰动bank；从两个最佳模型微调，最多24轮，batch32，6训练worker，保留一个物理核心。依次完成原模型增强集基线、新模型微调、mixed/clean测试、213条现场评价与长度视图。中断加`-Resume`。

实测线缆整组保留评价，本轮不使用现场线缆弱标签训练。波速按各FDR源设置映射；可靠物理末端估计可按1.2L重算并缩图，NN输入与候选保留全范围。

结果入口：`output/adaptation4/adaptation_summary.json`，日志在`output/adaptation4/logs/`。正式训练尚未完成，短检查不能当模型效果。完整状态、已知问题与接续要求见 [HANDOFF.md](HANDOFF.md)。
