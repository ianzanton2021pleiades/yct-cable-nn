# README 交付说明：明天早上的操作手册

> ZCode ｜ 2026-10-30 凌晨 ｜ 交给用户的执行顺序清单。全部命令在 **pwsh（PowerShell 7）** 中执行，Python 环境为 `Python 3.11 (gpushare_cu124)`（脚本内已写死 `E:\Anaconda\envs\gpushare_cu124\python.exe`）。
> 工作目录：`D:\GitRepository\Cable-NN\DifferentAgent\Zcode\Model`

## 总览：三步走

```
(a) 等 DG_V4_S1 数据集生成完成
(b) .\run_train_synthetic.ps1     合成预训练 90 轮（约 20–30 小时，可断点续训）
(c) .\run_finetune_field.ps1      现场微调 30 轮 + field_val 评估 + blind 盲测评估
(d) 按 §四 解读结果
```

---

## (a) 等待数据集生成完成

后台脚本 `DG/scripts/generate_dataset.py` 正在生成 **31250 条**合成样本（625 模板 × 50 实测实现）到 `E:\CableNNDataSet\ZCode\DG_V4_S1`。

**完成判据（唯一标准）：`E:\CableNNDataSet\ZCode\DG_V4_S1\manifest.csv` 存在，且数据行数为 31200（含表头共 31201 行）。**

> 注：31250 中有 50 条被拒绝采样淘汰（生成日志 `ok=31200, infeasible=50`），31200 才是正确结果。

注意事项：

- 生成脚本**只在全部样本完成时重写 manifest.csv**。当前目录下已有一个 1549 行的 manifest.csv 是早期 smoke 运行的部分清单（`make_manifest_partial.py` 所写），**不算数**——以被完整生成覆盖后的版本为准。
- 也可以旁证进度：`phases\` 目录下 `sample_*.npz` 文件数逼近 31250。
- manifest 未就绪前**不要启动 (b)**（train.py 按 manifest 的 split 列加载数据）。

## (b) 合成预训练（90 轮长程）

```powershell
cd D:\GitRepository\Cable-NN\DifferentAgent\Zcode\Model
.\run_train_synthetic.ps1
```

- 内容：`train.py --data E:\CableNNDataSet\ZCode\DG_V4_S1 --train-split train --val-split synth_val --arch v2 --width 48 --dsconv --hard-k 0 --out runs\unet_v2_main --epochs 90 --batch 16 --lr 2e-3 --workers 6 --val-limit 0 --resume`（v2 主配置：多尺度对比度前端 + 按通道 pos_weight + dsconv width48；`--val-limit 0` 表示用满 1600 条合成验证集）
- 预计 **20–30 小时**（90 轮 × 约 28050/16 步，视数据加载速度浮动）。
- **随时可以 Ctrl+C 中断**，重新执行同一条命令即从 `runs\unet_v2_main\last.pt` 断点续训（epoch、优化器、历史全部恢复）。
- 进度信息：每轮打印一行 `ep k/90 ... train_loss=... val_loss=... val_f1@20=... val_term=...m`；`runs\unet_v2_main\history.json` 每轮更新，`best.pt`（按 val loss 最优）与 `last.pt` 都会保存。
- 中途健康检查（可选，不强制）：看 history.json 里 `val_f1@20` 是否在前 20 轮内爬过 0.5、`val_term_mae_m` 是否降到数十米量级；**train/val loss 拉开是合成域正常现象，不要因此中断或加训**。

## (c) 现场微调 + 评估

(b) 正常结束后（`runs\unet_v2_main\best.pt` 存在）：

```powershell
.\run_finetune_field.ps1
```

脚本按顺序做三件事（约 1–2 小时）：

1. **现场微调 30 轮**：加载 `runs\unet_v2_main\best.pt`，`--data E:\CableNNDataSet\ZCode\field_train --train-split field_train --val-split field_val --arch v2 --width 48 --dsconv --out runs\unet_v2_field --epochs 30 --batch 12 --lr 2e-4`，同样支持 Ctrl+C 断点续训；
2. **field_val 评估**（固定 3.0 候选/条）→ `runs\unet_v2_field\eval_field_val.json`；
3. **blind 盲测评估** → `runs\unet_v2_field\eval_blind.json`。

> 协议红线提醒：第 3 步的 blind 评估是**最终汇报的一次性运行**（`02_评估协议.md` §八）。如果你打算在看完 field_val 结果后继续迭代微调，请先把脚本最后一段（blind 评估）注释掉，留到真正定稿时再跑一次。

## (d) 结果解读要点

**先看哪份文件**：`runs\unet_v2_field\eval_field_val.json`（主结论），`eval_blind.json`（最终汇报数字），`runs\unet_v2_field\history.json`（训练过程）。

**每个 JSON 里看什么**：

| 字段 | 含义 | 怎么判断 |
|---|---|---|
| `candidates_per_curve` | 候选预算 | 必须 =3.0（主口径），不是 3.0 的结果不作比较 |
| `precision` / `recall` / `f1` | **adaptive 容差 max(5 m, 2%·d)**（`evaluate.py` 默认 `tol_mode="adaptive"`，比固定 20 m 宽） | 主指标。**首轮达标线：f1 ≥ 0.55**（按 adaptive 口径判） |
| `tp / fp / fn` | 原始计数 | 用来判断瓶颈：fn 高=漏检（远端问题），fp 高=误报（夹具/杂波问题） |
| `terminal_output_rate` | 末端输出率（全分母） | 太低（<0.6）说明可见性头过保守 |
| `terminal_hit50` / `terminal_total` | 末端 50 m 内命中（全分母） | **首轮达标线：hit50/total ≥ 0.70** |
| `terminal_mae_m` | 末端位置 MAE | 达标后用于定位精度汇报 |
| `hit_by_distance_bucket` | 250 m 分桶命中表 | 重点看最远两个桶：历史痛点是 1500 m+ 漏检；近端好远端差=衰减补偿问题依旧 |

**达标判据（首轮，两个条件同时满足）**：

- field_val **F1 ≥ 0.55**（adaptive 口径，3.0 候选/条）；
- 末端 **hit@50 ≥ 0.70**（全分母）。

参考基准：历史 12 轮最好现场非浸水 hit@20 = 0.425（3.0 候选）。达标即确立基础版基线，再考虑高级版组件；不达标时按 §d 表格定位瓶颈（漏检/误报/远端/末端），**不要**先动架构、加参数或延长训练——这些路径已被 00_失败总结证伪。

**严禁事项**（评估协议冻结）：不在 blind/synth_test 上扫阈值；阈值只能在 field_val 上单向调高；改动评估代码后必须重新声明口径。

## 产出物清单

| 路径 | 内容 |
|---|---|
| `runs\unet_v2_main\{best,last}.pt, history.json` | 合成预训练权重与全程历史 |
| `runs\unet_v2_field\{best,last}.pt, history.json` | 现场微调权重与历史 |
| `runs\unet_v2_field\eval_field_val.json` | 现场验证集评估（主结论） |
| `runs\unet_v2_field\eval_blind.json` | 盲测评估（最终汇报用） |
