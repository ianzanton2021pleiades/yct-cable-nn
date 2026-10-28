# DG V2.6

面向 2.5 km 以内 RG58 与现场电力电缆的复数 S11 数据生成工程。

## 运行

- 单样本检查：`python DG_GUI_dataset_max2.5km.py`
- 批量生成：`python "[V2.6]DG_dataset_max2.5km.py" --help`
- 回归测试：`pytest -q`
- 固定案例验证：`python tools/validate_v26.py --output_dir <目录> --real_data_root <实测CSV目录>`

## 核心约束

保存的复数 S11 是唯一信号源；脉冲响应和阶跃响应始终由同一套 Client IFFT 重新计算，不保存或叠加与 S11 不一致的独立时域曲线。
