# 2026-09-08 CodeX Core-16路线扩展记忆

## Session

- 日期：2026年9月8日
- Agent：CodeX
- 项目：Cable-NN
- 主题：DG V3 RLGC、CST V1 0.1 m梯形、CST V1连续固定RLGC的Core-16工况扩展

## 已确认

- 原 `DifferentAgent/CodeX/DG_Route_Evaluation` 保持只读，本轮新建独立目录 `DifferentAgent/CodeX/DG_Core16_Route_Evaluation`。
- 三条路线、40 m主体、1 m/135 Ω前置无损线、10 m special_joint1、20/30 m的0.5 m/300 Ω无损线、开路终端、9 kHz–200 MHz/10000点网格、Algorithm 1/2和统一测量链均按原评估协议保留。
- Core-16 参数写入 `core16_cases.yaml`，RLGC单位是每0.4 m等效单元；CST 0.1 m脚本使用0.25缩放，连续路线换算为单位长度，DG V3按100 MHz effective_rlgc锚定。
- C05的30–40 m段为G=500 μS/0.4 m；按用户指定的原 segmented_loss 语义，末端附加支路也为G=500 μS、C=16 pF。
- 检查表中全部R/L/C组合的100 MHz锚定外部电感 `L' - R'/omega_ref` 为正，没有发现需要向用户升级的参数化冲突。

## 已建立文件

- `DifferentAgent/CodeX/DG_Core16_Route_Evaluation/.gitignore`
- `DifferentAgent/CodeX/DG_Core16_Route_Evaluation/core16_cases.yaml`
- `DifferentAgent/CodeX/DG_Core16_Route_Evaluation/compare_core16_routes.py`
- `DifferentAgent/CodeX/DG_Core16_Route_Evaluation/README.md`

## 执行状态

- 已执行正式 Core-16 生成：`E:\\Anaconda\\envs\\gpushare_cu124\\python.exe -B .\\compare_core16_routes.py`，运行约249.87 s。
- 产出96条指标、32张PNG对照图、96个S11 CSV和96个响应NPZ；S11单文件为10000个数据点加表头。
- 三条路线的clean/measured数值检查均为16/16通过；clean层最大|S11|约为DG V3 0.99273、CST梯形0.99392、CST连续0.99297。
- 原 `DifferentAgent/CodeX/DG_Route_Evaluation` 没有新增或修改文件。Core-16运行报告为 `DifferentAgent/CodeX/DG_Core16_Route_Evaluation/Core-16路线对比评估报告.md`。
- 用户指出原资产图底部表格留白过大，要求按主体区段列出六列区段/R/L/G/C/描述。未修改源代码或配置，只对32张既有PNG做了一次性版式重排：保留上方六个图，统一输出为2609×2850像素，表格顶部对齐且按最多6个主体区段预留高度。
