# CodeX 本轮记忆：CST_Reproduction V1 执行完成

## Session metadata

- Agent: CodeX
- Date: 2026-09-02 Asia/Shanghai
- Workspace: `D:\GitRepository\Cable-NN`
- Branch: `main`（用户明确要求不切换分支）
- Scope: 只修改 `DifferentAgent\CodeX\CST_Reproduction\V1`，另新增本轮结果文档；V0、CST、REF和现场原始数据冻结。
- Parent conversation memory: `AgentsMEM\2026-09-01_221701_CodeX_CST_Reproduction_Conversation_Memory.md`

## User request handled

执行 `CST_Reproduction V1：细分梯形单元与频变损耗对照`：固定RLGC四种细分、连续固定RLGC、DG V3风格频变损耗、接头消融、两套REF FDR、现场145 MHz参照、RG58缺陷局部性、图/CSV/JSON/Markdown交付。

## Code changes

主程序：`DifferentAgent\CodeX\CST_Reproduction\V1\cst_fdr_reproduction.py`

- 模型变体：`fixed_ladder_0p4m`、`0p2m`、`0p1m`、`0p05m`、`fixed_continuous`、`dg_loss_0p1m`和四个接头消融模型。
- 细分按 `cell_length/0.4` 等比例缩放R/L/G/C；15 m缺陷覆盖14.8–15.2 m。
- Algorithm 2沿用 `REF\算法2\fdr_response_core.py` 的默认参数及公式，使用NumPy分块；新增小规模逐数组REF parity检查并写入summary。
- RG58局部峰指标窗口修正为45–60 m，避免把末端峰误标成局部缺陷峰；正式10000时间点结果恢复为约49.2 m定位。300点冒烟结果的局部峰不可用于正式定位。
- 消融图加入完整0.1 m基线；报告增加频域依据、FDR依据、算法/离散伪影判别、速度标尺差异和现场限制。
- 诊断增加三接头及末端的FDR事件位置字段，写入`output/model_summary.csv`和`summary.json`。

## Formal outputs

- 正式运行：10000频率点、10000算法2时间点、运行约138.6 s。
- `s11_output`：34个CSV；每个10000行，表头唯一为 `Frequency_Hz,S11_Real,S11_Imag`，无NaN/Inf、频率严格递增。
- `output`：102个距离域CSV；算法1文件和算法2 step/impulse文件分开，数值有限。
- `assets`：分辨率收敛、接头消融（含完整模型）、固定/连续/DG损耗、9个V0/V1工况、局部性、现场参照图；报告CST图保持自然纵横比。
- `comparison_report.md`：最终报告，含算法2/REF parity、事件位置偏移和所有主要限制。
- `AgentsDoc\CST_Reproduction_V1_执行结果.md`：本轮人类可读的正式结果摘要。

## Key numerical evidence

- 0.4/0.2/0.1/0.05 m总R/L/G/C分别守恒到约`1e-14`，缺陷单元数为1/2/4/8。
- 离散截止估计：145.288/290.576/581.152/1162.303 MHz。
- 0.1 m固定梯形接头后尾波/主峰：算法1=`0.003623`，算法2=`0.036294`，通过0.05局部指标。
- 相对连续参考RMS仍未完全收敛：0.05 m的S11/算法1脉冲/算法2脉冲约`0.249/0.131/0.110`，不能声称完全连续等效。
- REF核心 parity：256跳过首点频率、256时间点，step/impulse raw及平滑最大绝对误差均为0。
- RG58：CutPVC算法1/2定位`49.233/49.162 m`，guhua定位`49.183/49.148 m`；后区/局部RMS分别为`0.0898/0.1825`与`0.0227/0.0289`。
- 现场四条参考在145 MHz前后幅值变化为`-0.715/+0.022/-0.088/+0.169 dB`，不能统一解释为145 MHz共性截止。

## Important caveats for future agents

- 本征线路速度因子约0.609，而FDR位置标尺约0.669/0.6715；默认事件被映射到约11.5/23.8/35.4/46.7 m附近。这是锁定速度口径下的空间标尺差异，不要为了看起来落在10/20/30/40 m而改电路长度。
- 现场数据只读、未校准或未统一标注；不用于V1参数建模，不据此生成Field/YJV通用RLGC。
- 没有CST ASCII S11导出，不能计算CST逐点误差。
- V1已经包含DG V3风格频变损耗对照；这不是DG V3现场模型本身，也不是正式训练参数。
- 不要修改旧的其他Agent记忆；本文件是本轮执行状态的最新记忆。
