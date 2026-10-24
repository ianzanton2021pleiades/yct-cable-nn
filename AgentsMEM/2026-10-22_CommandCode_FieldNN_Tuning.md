# FieldNN_20261022 调优 Session（CommandCode）

- 日期：2026-10-22（Asia/Shanghai）；Agent：CommandCode；工作目录 D:/GitRepository/Cable-NN。
- 用户目标：接续 CodeX 未完成的 FieldNN 项目；明天组会，需要尽快用现有 10k 数据训练/调优 Transformer 与 U-Net，Mamba 暂缓；14:30 前留下完整接续 Markdown 给 CodeX。
- 资源：E:/Anaconda/envs/gpushare_cu124/python.exe；NVIDIA CMP 90HX 10 GiB；不得改功率/风扇/驱动。用户已授权直接启动训练。
- 约束：冻结生成器和 E:/CableNNDataSet/CodeX/FieldNN_20261022 的 10k 数据；不改 Core16_Field、DG、GUI；不自行 git commit。现场标签 unknown，不报告现场准确率。
- 提问预算：本次对话 2 轮 grill-me，每轮最多 4 问；目前尚未使用（用户本轮给的信息足够，先执行）。

## CodeX 遗留状态

- 运行目录 output/tu_baseline_20261022：Transformer 5 epoch 完成并完成 val 校准 + test；U-Net 5 epoch 完成但尚未评估。
- Transformer baseline（test，1000 条合成）：event F1@2/5/10m = 0.273/0.444/0.521；interval F1 IoU0.5 = 0.397（grid IoU 0.563）；terminal F1 = 0.560，匹配误差 4.15 m。
- U-Net baseline（本 session 补评估）：event F1@2/5/10m = 0.264/0.433/0.508；interval F1 = 0.227（grid IoU 0.237）；terminal F1 = 0.516；val 阈值全部落在 0.8 上边界。
- baseline 训练均 batch4、5 epoch、lr 3e-4，val loss 仍单调下降，明显欠训练。

## 两个只读子代理审计结论

- 代码审计：5/6 个 head 的阈值选择落在 0.8 上边界，阈值网格被截断；Decoder 线性插值存在约 +1.4 m 亚像素偏置；Dice 只用于 interval 且 AMP 下 fp16 累加；event target σ=3m 的 float32 尾部延伸到 ±44m，和 pos_weight=30 一起让 loss 中“正样本”质量虚高；val loss 被电缆末端以后 66% 的平凡负样本主导。
- 数据审计：10k 曲线中 25% 无区间、3.58% 无事件；事件绝大多数是接头（66%）或区间边界（34%）；约 400 个 local 区间宽度 < 解码最小间隔 3m；joint 与区间边界可能相距 <5m；terminal 永远等于 length；split 是 sample_id%10 的随机内插，不是分组/盲测；事件目标正点仅 0.94%，区间 2.1%，terminal 0.21%。
- 建议优先级：延长训练 > 阈值网格 > 区间解码（hysteresis/gap fill）> loss 重加权/掩码 > 模型容量。

## 本 session 已做代码改动

- train.py task_loss：
  - event target 小于 0.135（约 2σ）的尾部置零；
  - 以 terminal target argmax 估计电缆末端，末端 +10 m 之后的点全部 mask 掉；
  - pos_weight 从 [30,2,80] 改为 [12,6,50]；
  - BCE 按样本、按通道做 masked mean；
  - Dice 从只监督 interval 改为三头都监督，权重 [0.5,1.0,0.5]；Dice 用 fp32 sigmoid。
- train.py 新增 CLI：--lr、--weight-decay、--workers、--patience；run_info.json 记录 training 与 loss 配置。
- evaluate.py 阈值候选从 5 个扩展为 (0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95)。
- 未改生成器、数据、模型结构、Core16/DG/GUI。

## 当前运行

- 新运行目录：output/tu_tuned_20261022/{transformer,unet}。
- 命令：train.py --epochs 36 --batch-size 8 --lr 5e-4 --workers 2 --patience 12。
- 两模型并行，单 epoch 约 74–78 s（共享 GPU，约 115–122 samples/s），预计 45 min；若 14:00 前未完成，用 best/last checkpoint 收尾。
- 首 epoch：transformer train/val = 1.1956/1.0782；unet = 1.2638/1.1323（新 loss 尺度，不能与 baseline 数值直接比较）。
- 后续：val dense-threshold 校准 → test → development 31 条现场推理 → summarize_tu.py → 接续文档。

## 训练与评估结果（2026-10-22 14:02）

- Transformer tuned：实际 20 epoch；best=19，val 0.5748（新损失）。Test（1000 条）：event F1 @2/5/10m = 0.380/0.563/0.624；event P/R@5m = 0.743/0.453，MAE 1.65m；interval F1 IoU0.5 = 0.585，grid IoU 0.760，matched IoU 0.822；terminal F1 0.403，matched error 3.60m。阈值 event 0.85 / interval 0.95 / terminal 0.95。
- U-Net tuned：实际 33 epoch（best=29，val 0.7897）；Test：event F1 @2/5/10m = 0.379/0.552/0.617；P/R@5m = 0.682/0.464；interval F1 0.245，grid IoU 0.319；terminal F1 0.454，error 3.49m。
- 对比 baseline：Transformer event 0.444→0.563，interval 0.397→0.585；U-Net event 0.433→0.552，interval 0.227→0.245。terminal 回退（Transformer 0.560→0.403，U-Net 0.516→0.454），阈值曲线平坦，判定为新损失导致而非阈值，建议下一轮单独恢复 terminal loss。
- 训练故障：`num_workers=2` 在 Windows 上两次死锁（Transformer epoch13、U-Net epoch33，GPU 0%、worker 存活）。Transformer 用 `--workers 0 --resume --epochs 20` 恢复到 20；U-Net 直接用 best epoch29。后续长训用 workers=0 或先修 DataLoader。
- 运行目录 output/tu_tuned_20261022；thresholds/evaluation_val/evaluation_test 已生成；现场 development 31/31 全部完成，summary.json 的 field_metrics=null。
- 现场覆盖：Transformer 每条曲线事件中位数 20 [15,38]、区间 7 [2,22]、terminal 候选 27/31（consistent 8/uncertain 19/none 4）；U-Net 事件 12 [6,41]、区间 12 [0,62]、terminal 候选 19/31（consistent 0/uncertain 19/none 12）。相对合成训练集（事件中位数约 5、区间约 1）严重过检，域差仍主导，不能做现场判读。
- infer.py 原有 bug：terminal_consistency 的 checks 为 np.bool_，有 terminal 候选时 json.dumps 抛 TypeError；已改为 Python bool，修复后两模型 31 条现场推理约 6s/模型。
- 已写 HANDOFF_20261022_1430.md；Milestone 已追加；README/PLAN 已加最新状态指针。

## 待办 / 下一轮

1. 现场 development 31 条推理完成后，把 processed/not_processed、事件数中位数、区间数中位数、terminal 一致性填入 HANDOFF 第 7 节。
2. 恢复 terminal：terminal pos_weight 80–100、去掉 terminal Dice，或冻结 backbone 只微调 terminal 头；用 val 选，test 冻结。
3. 区间解码 hysteresis/gap-fill（约 350 条曲线过分割）；event min separation 2m 消融；terminal 容差按 max(10m, 1%L) 报告。
4. 两模型都只跑了 20/33 epoch，未做 2 seed；<0.02 的差异不要当真。
5. 未做三相融合、GUI 接入、生成器域差修复。
