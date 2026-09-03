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

## 2026-09-02 后续Agent修改后的独立验收更新

- 用户说明其他Agent已继续处理本任务，并明确移除`CST V0 0.4 m梯形`：该候选效果差、没有继续比较价值。当前正式候选改为三路：`dg_v3_rlgc`、`cst_v1_ladder_0p1m`、`cst_v1_continuous`。
- 已核对移除一致性：`evaluation_cases.yaml`、`compare_dg_routes.py`、README、正式报告、`summary.json`、`metrics.csv/json`及S11/响应目录均不再包含V0；当前为54条指标、54个S11 CSV、54个响应NPZ、18张工况板，报告图片链接无缺失，3个Python脚本均可通过AST解析。
- 当前验收结论为`Needs revision`，不能记为整个原计划已经闭环。原因如下：
  1. 实测域门槛只实现了DG V3的5项频域特征、每类目标24个样本；没有实现原计划中的FDR本体纹理、峰密度、阶跃斜率、尾波比、实测随机二分自距离或多变量分布距离。它能证明当前DG V3分布未通过，但不能称为完整门槛C。
  2. `scalability_check.py`实际只测DG V3健康、无测量链、固定约5段、200 MHz/10000点；没有测1 GHz/50000点、缺陷/接头复杂拓扑或批量吞吐。报告中的“40–2500 m均<0.1 s、<3 MB”只适用于该窄测试，不能泛化为完整DG生成性能。CST V1仅计算理论单元数，没有实际运行扩展比较。
  3. 报告把9/9称为“工况硬门槛”，但实现主要是方向布尔检查；配置中的`effect_floor_relative`没有进入`metrics_for`判定。9/9只能说明三路能表达九工况方向，不能单独区分路线优劣。
- 其他一致性问题：`Milestone.md`仍记录四路候选并包含已移除的V0，属于过时描述；评估目录存在`__pycache__`，虽被gitignore忽略但不符合README强调的`python -B`整洁预期。
- 当前可保留的有效结论：V0已正确退出比较；三路九工况输出内部一致；当前`provisional_rlgc_v1`的Field联合采样存在大量`dielectric loss exceeds the requested 100 MHz attenuation`失败，且DG V3现有频域分布门槛未通过；因此当前参数仍不得生成正式训练集。
