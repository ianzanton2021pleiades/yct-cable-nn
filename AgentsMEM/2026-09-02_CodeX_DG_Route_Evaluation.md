# CodeX 对话记忆：DG 路线对比评估

## Session

- 日期：2026-09-02（Asia/Shanghai）
- Agent：CodeX
- 工作区：`D:\GitRepository\Cable-NN`
- 任务目录：`DifferentAgent\CodeX\DG_Route_Evaluation`
- 用户要求：执行 DG V3、CST V0、CST V1 的路线评估；V2.7 只作真实感参照；不得修改 `CST_Reproduction`；工况图参考 V0 的 `comparison_board_local_C4pF_15m.png`；可使用不超过两个 GPT-5.6 Luna 子智能体。

## 已锁定口径

- 正式候选：DG V3 RLGC、CST V0 0.4 m 梯形、CST V1 0.1 m 梯形、CST V1 连续固定 RLGC。
- CST V1 的频变损耗模型只作消融；V2.7 不具备获胜资格。
- 使用物理一致性、九工况趋势、实测域真实性三道独立硬门槛，不计算掩盖短板的综合分。
- 输出分 clean core 与统一测量链两层；DG V3 原生测量链只作消融。
- 全部未标注现场数据进入分布真实性门槛，但不赋予健康/缺陷伪标签。
- S11 是权威数据，算法1/2响应只作派生分析，禁止 IFFT 回写或距离域修形。

## 当前进度

- 已建立独立评估工程；现有 DG、CST、V2.7 与实测目录保持只读。
- 两个 Luna 子智能体完成候选适配和实测证据口径的只读审查。
- 正式九工况运行完成：9 kHz–200 MHz/10000点、算法2 10000时间点、72条候选×层×工况指标、18张对照板，耗时182.61 s；四路方向检查均为clean 9/9、统一测量链9/9。
- 当前实测聚合重新扫描405个CSV，排除129个IFFT文件和26个超2.5 km Field文件，纳入250条：RG58=139、Field=85、correction=14、wet_1500m=12；重复组61、pairwise=136。
- DG V3原生分布检查：RG58 24/24成功、核心特征3/5；Field在96次尝试中仅20次成功，76次报`dielectric loss exceeds the requested 100 MHz attenuation`，核心特征2/5。两类均未通过实测分布门槛。
- 结论：当前没有候选通过三道门槛。下一版应保留DG V3连续频变RLGC和S11协议作为底座，先修正Field衰减预算/介损联合采样，再加入健康段空间相关物理随机场；CST九工况作为回归集，V2.7距离域修形不移植。
