# 2026-10-23 WorkBuddy FieldNN PINN Session

## Session 基本信息

- 日期：2026-10-23，Asia/Shanghai；Agent：WorkBuddy；工作目录 `D:/GitRepository/Cable-NN`。
- 用户诉求：检查 AGENTS.md 与既有模型/试验，针对 DeepSeek（CMDC）`20261022_FieldNN_Hybrid_pretrain50_curves.png` 的 50/100 轮结果做过拟合诊断与改进；要求 50 轮后 loss 低于原基线、50→100 轮不过拟合、加入物理信息约束作为 loss；优先 Transformer/U-Net。
- 约束：不动 DeepSeek 现有文件（只读复用其数据集与评估函数）；新试验目录 `DifferentAgent/Workbuddy/FieldNN_PINN_20261023`；不提交 Git；GPU 预算 3 小时。
- 资源：`E:\Anaconda\envs\gpushare_cu124\python.exe`；CMP 90HX 10 GiB。

## 诊断结论（基于 CMDC history.csv）

- 50 轮：train 0.445 / val 0.543，gap 0.098，val F1@5m 0.712。
- 100 轮：train 0.255 / val 0.584，gap 0.329，val F1 0.734。
- 后 50 轮 val F1 仅 +0.022 而 loss 掉 0.19 → 主要是置信度推高，不是学到新特征。
- 分项：val event 0.515 主导，terminal F1 0.971 早已饱和，interval 停滞。
- 根因三条：9600 合成样本对 3.7M 参数偏少且增广形态重复；Iter4 标签是"孤立一次反射主瓣"近似（`is_exact_full_network_peak: false`），未标注多次反射构成标签噪声；三个 head 全无波形解释约束。

## 用户 grill-me 决策（第 1 轮）

1. GPU 预算 3 小时以内。
2. 交付重点：曲线 + 消融 + 现场案例。
3. 现场口径：沿用现有 213 条实测（147 条有工程师位置）；要求 >20 条可准确识别、2~4 个完全正确案例；当前最重要的是曲线和指标好看、体现物理约束、优于 DeepSeek。
4. 模型路线由 Agent 决定，不参考 ChatGPT/CodeX 的结果。

## 本轮实现

目录 `DifferentAgent/Workbuddy/FieldNN_PINN_20261023`：

1. `models_pinn.py`：Iter5 U-Net/Transformer 拓扑 + 第 4 个 signed 反射系数头 `a(d)`；reflection head bias 置零。
2. `losses_pinn.py`：三个监督项与 CMDC 逐字一致（保证 val loss 可比）；新增 `PhysicsResidual`：
   - reconstruction：`a ⊛ g`（g 为 5 m 高斯测量脉冲）与观测脉冲的余弦残差；
   - step：`cumsum(a)` 高通后与观测阶跃的余弦残差；
   - support：事件质量必须落在观测包络高能处，`1 - x/(1+x)` 有界；
   - ordering：末端不得早于最后一次反射（softmax 期望位置 + 25 m 余量）；
   - sparse：`|a|` 的 L1；consistency：事件图与 `|a|` 的余弦一致性。
   权重默认 recon .10 / step .03 / support .05 / order .02 / sparse .002 / consist .05。
3. `data_pinn.py`：复用 Iter5 ShardDataset 基础视图；新增距离轴 warp（±1.2%，标签同步缩放，对应波速不确定度）、mixup（p=0.25，标签取并集）、特征噪声。
4. `train_pinn.py`：100 轮，AdamW lr3e-4 / wd0.03 / dropout0.15 / EMA0.999 / cosine，记录逐步与逐轮，保存 epoch050 快照便于 50↔100 对比。
5. `finetune_field.py`：复用 `E:\CableNNDataSet\Deepseek_CMDC\FieldNN_Hybrid_Long_20261022`（只读），hybrid:syn:fieldpos = 1:0.35:0.15 加权采样，lr 8e-6。
6. `eval_pinn.py`：合成 test clean/mixed + hybrid test + 现场 test，附加 P_ann 与"全部位置命中曲线数"。
7. `case_study_pinn.py`：现场案例，含"末端必须在最后事件之后"的物理后处理。

## 运行中

- `output/pinn100`：10:28 启动，67 s/epoch，预计 12:20 结束。epoch2 train_sup 2.07 / val_phys 0.100（首轮 0.117），物理残差在下降。
- 待办：50/100 对比曲线、现场微调、案例图、消融（物理项 on/off）、技术说明与 Milestone。

## 实际训练经过（最终）

- 阶段 1（1–50 轮，cosine 3e-4→1e-6）：val 0.556 / F1 0.702。基线同轮 0.543 / 0.712，基本持平。
- 阶段 2（51–92 轮，5e-5 重启）：val 0.509 / F1 0.738。第 79 轮 val 最低 0.479。
- 阶段 3（93–150 轮，3e-5 重启）：第 95 轮 val 最低 **0.448**，随后回升，150 轮 0.567 / F1 0.756。
- 阶段 4（151–158 轮，mixup 0.30 + wd 0.04 加强正则）：val 0.537–0.546，无进一步收益。
- 用户在第 158 轮后叫停，判定已过拟合，要求保存数据并做详细分析。**不再续训。**

## 关键结论

- 合成域超过基线 100 轮：F1@5m .796 vs .782，interval .787 vs .714，terminal .980 vs .971；100 轮即饱和（.795），150 轮无增益，interval 反降到 .787。
- 现场域与基线打平：非浸水 hit@20 .416@4.30（零样本）/ .405@3.72（微调），基线 .426@3.60；同预算无优势。
- 现场按长度分桶：500–1000 m .607；1000–1500 m .533；1500–2000 m .196；2000–2600 m .244。**1500 m 是明显分界。**
- 末端误差中位 16.7 m，50 m 内 57%；15 条短接参考误报事件 0。
- 训练损失 0.42 < 验证 0.55 是因为训练视图开了 mixup+warp；关闭后训练损失立刻 0.21，两者不可直接相减。

## 踩坑记录（重要）

- **Windows 上训练时用 Python 读 history.csv 会导致训练进程 PermissionError 崩溃**（CSV 写入被打断）。要么先 `cp` 再读，要么用 bash 的 tail/cut。曾在第 43 轮崩溃一次。
- 该 GPU 显存 10 GiB，训练占 4.8 GiB。训练期间**不能**再跑现场微调（同样 4.8 GiB），会 OOM；只能跑推理（0.7 GiB）。
- 50→100 用 1e-4 重启会带来 ~0.55 的损失跳变，曲线难看；5e-5 只跳 0.05；从 92 轮用 3e-5 几乎无跳变。

## 产物

- 报告：`AgentsDoc/2026-10-23_PINN_完整分析报告.md`（结论版）、`2026-10-23_PINN_150轮分析.md`（150 轮节点版）、`2026-10-23_PINN_92轮训练数据.md`
- 图：`Image/20261023_PINN_curves_150.png`、`20261023_PINN_diagnostics.png`、`20261023_PINN150_gallery.png`、`20261023_PINN150_metrics.png`、`20261023_PINN150_case1..4.png`、`20261023_PINN_pretrain92_curves.png`
- 结果包：`output/results_bundle/`（历史、评估 json、阈值扫描、延迟、对比表）
- 检查点：`output/pinn200_phase4/unet/{epoch008.pt(100轮), best.pt(117轮), epoch058.pt(150轮), last.pt}`、`output/field_ft_150/best.pt`


## 注意事项

- 现场标签只有弱正例，只能报告已知位置命中率与候选密度，不能报 precision/recall；P_ann 是下界。
- 用户明确不要过度宣传：曲线必须来自真实 history.csv，不得编造。
