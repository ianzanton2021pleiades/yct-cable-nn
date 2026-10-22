# Transformer / U-Net 首轮训练接续

更新日期：2026-10-22（北京时间）；主导CodeX；常规执行委派GPT-6 Luna Max。

## 当前用户决定（优先于此前暂停建议）

- 时间/额度有限，明确不改生成器、不重新生成数据，Mamba搁置。
- 使用现有10000条数据，只做Transformer和U-Net训练、推理、评估与必要调优。
- 用户已明确同意：由Agent现在直接运行两模型各5个完整epoch，验证损失选best，再合成评估与开发现场推理。无需再次询问GPU/启动权限。
- Python：E:/Anaconda/envs/gpushare_cu124/python.exe；GPU：90HX 10GiB。不改功率、风扇、驱动等设置。
- 本请求允许必要检查/测试；不自行Git提交。原实测目录只读，不改GUI。
- 本请求提问预算2轮、每轮最多4问；第1轮仅问启动/epoch，已答同意直接各5epoch；尚余1轮。

## 路径与运行协议

- 实验目录：D:/GitRepository/Cable-NN/DifferentAgent/CodeX/FieldNN_20261022。
- 数据：E:/CableNNDataSet/CodeX/FieldNN_20261022，manifest为8000/1000/1000，157分片。只读复用。
- 本次运行目录：实验目录/output/tu_baseline_20261022/，内含transformer/、unet/。禁止与旧smoke、pilot_v1混淆。
- 训练入口train.py；--model指定transformer或unet，--epochs 5，--output指本次运行目录。完整训练不传--max-train-batches/--max-val-batches。
- 每模型保存history.csv、best.pt、last.pt、config.json、run_info.json；日志在对应运行目录。中断后同命令加--resume；Epochs为总轮数。
- 同时只跑一份训练。Transformer训练结束后的CPU阈值/指标计算可与U-Net训练并行；执行Agent负责两个进程，不能启动重复训练。两模型最终均为batch4、各5完整epoch。
- 评价：evaluate.py先--split val --calibrate，再--split test，分别写evaluation_val/evaluation_test。阈值来自val，不能用test调参。
- 现场：prepare_real_manifest.py；infer.py --manifest output/real_manifest.jsonl --subset development --checkpoint各自best.pt；只对已开发31条做探索推理，不把182条身份未确认记录用于调参。
- 训练/评估/现场全部结束后读取文件核实，不以进程启动或exit0代替结果。

## 关键科学边界

S11_REVIEW.md记录了现有生成域缺陷：单例低Z0、高损耗、近端覆盖窄；本次是用户要求的冻结数据基线，不能据训练好看宣称现场泛化成立。4输入为脉冲/累计阶跃及logRMS尺度，8192点固定0–3250m，不用真长度裁剪；3输出event/interval/terminal。合成区间是RLGC参数变化几何真值。213条现场机器标签均unknown；只报告候选及一致性，不计算现场准确率/F1。

## 实时进度（接续Agent先核对，不重复启动）

- 2026-10-22 preflight：无匹配FieldNN Python进程；本次output/tu_baseline_20261022原先不存在。E盘manifest=train/val/test 8000/1000/1000，共157分片；冻结复用、不重生成。
- Transformer实际启动命令：train.py --data E:/CableNNDataSet/CodeX/FieldNN_20261022 --output output/tu_baseline_20261022 --model transformer --epochs 5；未传max-batches，默认batch=4。exec session_id=19306。
- Transformer epoch 1/2已完整保存：loss train/val分别0.664392/0.566739（51.53s）和0.517745/0.436458（49.88s）；best=epoch 2，last.pt=epoch 2，peak GPU allocation=0.152 GiB。
- 父代理一度根据启动瞬时速率考虑B32；之后核对epoch实测约50s，认定B4表现正常并撤销调整。Transformer在epoch 3 train约18%处收到Ctrl+C；该不完整epoch无history/checkpoint，不计入5轮；epoch 1/2及checkpoint保留。初始“约10 batch/s”只是启动瞬态，不能代表完整epoch吞吐。
- 最终两模型均统一batch=4、lr不变、各完整5 epoch。Transformer已完成：先前完整epoch1–2，resume后epoch3–5；train/val loss依次0.664392/0.566739、0.517745/0.436458、0.434159/0.399693、0.381492/0.344626、0.347035/0.328837。best=epoch5；单epoch47–52s、吞吐174.6–192.1 samples/s、peak GPU allocation约0.152GiB。一次epoch3约18%中断未计轮数；恢复训练session_id=69392，exit0。
- 下一步Transformer val全1000校准→test全1000→development推理；然后U-Net从零batch4完整5 epoch。
- 首次Transformer全量val触发CPU校准瓶颈：旧实现为每头5阈值各重跑全部decode/metrics（15遍）；已停止该轮且未产出evaluation目录。按父代理授权将三头独立阈值校准合并为每候选一次全量评估（5遍）；N=2确定性CPU对照与旧算法阈值及candidate_scores完全相同。
- Transformer val校准已全量1000条完成：event F1@5m=0.447874，interval IoU0.5 F1=0.427331，terminal F1=0.530902；evaluation_val/evaluation.json及val thresholds.json已生成。按val阈值完成全量test：event F1@5m=0.443971，interval IoU0.5 F1=0.397100，terminal F1=0.560445；evaluation_test/evaluation.json已生成。
- 按父代理授权，Transformer val/test评估期间并行启动唯一U-Net训练：exec session_id=31622，epoch1/5运行中，batch=4；暂无OOM/错误。
- 下一步复用既有real_manifest.jsonl，只推理subset=development（31条），随后继续等U-Net epoch并同步记录。
- 随后每模型val校准/test → development 31条现场推理 → 汇总run/summary.json。

## 故障与续接

先查看训练/评价日志末尾、history.csv、checkpoint epoch。若仍运行，等待其结束，不启动重复任务。训练异常仅修当前必需问题，不改生成器/数据、不扩大模型。不自动续到20/80epoch；先比较首轮结果再决定。若额度不足，保留已完成checkpoint并在本文件记录进程、阶段、错误和精确下一条命令。

## 精确接续命令（确认无重复进程后使用）

```powershell
$Experiment = 'D:\GitRepository\Cable-NN\DifferentAgent\CodeX\FieldNN_20261022'
$PythonPath = 'E:\Anaconda\envs\gpushare_cu124\python.exe'
$DataRoot = 'E:\CableNNDataSet\CodeX\FieldNN_20261022'
$RunRoot = "$Experiment\output\tu_baseline_20261022"
# 按实时进度选择模型；不要重跑已完成阶段
$ModelName = 'transformer' # 或unet
& $PythonPath -X utf8 -B "$Experiment\train.py" --data $DataRoot --output $RunRoot --model $ModelName --epochs 5 --resume
& $PythonPath -X utf8 -B "$Experiment\evaluate.py" --data $DataRoot --checkpoint "$RunRoot\$ModelName\best.pt" --split val --calibrate
& $PythonPath -X utf8 -B "$Experiment\evaluate.py" --data $DataRoot --checkpoint "$RunRoot\$ModelName\best.pt" --split test
& $PythonPath -X utf8 -B "$Experiment\infer.py" --checkpoint "$RunRoot\$ModelName\best.pt" --manifest "$Experiment\output\real_manifest.jsonl" --subset development --output "$RunRoot\$ModelName\field_development" --plots 4
# 两模型产物齐备后汇总
& $PythonPath -X utf8 -B "$Experiment\summarize_tu.py" --run $RunRoot
```

若manifest缺失，先运行同目录prepare_real_manifest.py；它仅读已有统计和源证据，输出到output/real_manifest.jsonl。每个命令成功后再进入下一个，不能忽略失败继续。
