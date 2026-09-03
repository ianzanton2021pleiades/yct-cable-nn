# 新的里程碑

## 2026年8月31日 21点 Agent: CodeX

建立DG V3断代架构：直接物理/频域生成双频复数S11，采用稀疏物理真值与可重建空间响应协议，补充实测统计校准、NumPy Dataset、固定案例和多进程冒烟验证；正式参数需待用户全量校准后冻结。

## 2026年9月1日 3点 Agent: CodeX

将DG 2.7的RLGC材料优势迁入DG V3：RG58采用有效同轴模型，Field采用有效分布参数，老化与受潮使用连续Debye渐变；保留V3纯频域生成边界，不引入模板、末端重构或IFFT反投影，版本仍为3.0.0。

## 2026年9月2日 0点 Agent: CodeX

建立 CST_Reproduction V1：将0.4 m固定RLGC梯形细分为0.2/0.1/0.05 m，对照连续模型、频变损耗和接头消融，并保持两套REF FDR参数及V0冻结。

## 2026年9月2日 17点 Agent: Antigravity

建立DG路线对比评估工程（DifferentAgent/CodeX/DG_Route_Evaluation/）：对DG V3 RLGC、CST V0 0.4m梯形、CST V1 0.1m梯形和CST V1连续固定RLGC四路候选进行clean/common-measurement双层九工况评估。四路均通过被动性和方向检查（9/9），但门槛C实测域门槛均未通过：CST三路不具备随机拓扑生成能力；DG V3的Field衰减预算与介损参数不自洽导致高失败率。扩展性检查确认DG V3在40–2500 m全部通过（<0.1 s，<3 MB），而CST梯形在2500 m需6250–25000个单元。结论：不整体替换为CST，以DG V3为底座先修正联合采样，再加入物理随机场。
