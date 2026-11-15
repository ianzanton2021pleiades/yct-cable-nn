# 模型v3 全链（长程版）：100ep 合成预训练 + 35ep 混合微调 + field_val/blind 评估
# 与 run_pretrain_v3/run_mixed_v3 的区别仅在预训练轮数（100 vs 70）与输出目录
# 断点续训：每一步随时 Ctrl+C，重新执行本脚本即从各自 last.pt 恢复（--resume 已带上）
# 注意：末段 blind 评估按协议只允许在最终汇报时运行一次；若还要迭代，
#       请注释掉最后一段（blind）再执行。
# 耗时参考：第一轮实测约 103 s/epoch，100ep 预训练约 3 小时，混合微调约 1 小时内。
cd $PSScriptRoot

# ---- 步骤1：合成预训练 100ep ----
& "E:\Anaconda\envs\gpushare_cu124\python.exe" train.py `
    --stage pretrain `
    --synth-data "E:\CableNNDataSet\ZCode\DG_V5_S1" `
    --field-data "E:\CableNNDataSet\ZCode\field_train_v2" `
    --arch v1 --out runs\v3_pretrain_long `
    --epochs 100 --batch 16 --lr 2e-3 --workers 6 `
    --val-limit 0 --resume

# ---- 步骤2：混合微调 35ep（field_train + 25%合成回放） ----
& "E:\Anaconda\envs\gpushare_cu124\python.exe" train.py `
    --stage mixed `
    --synth-data "E:\CableNNDataSet\ZCode\DG_V5_S1" `
    --field-data "E:\CableNNDataSet\ZCode\field_train_v2" `
    --init runs\v3_pretrain_long\best.pt `
    --arch v1 --out runs\v3_mixed_long `
    --epochs 35 --batch 16 --lr 2e-4 --workers 4 `
    --val-limit 300 --resume

# ---- 步骤3：field_val 评估（NN + 规则基线 + 组级bootstrap CI，固定3.0候选/条） ----
& "E:\Anaconda\envs\gpushare_cu124\python.exe" evaluate.py `
    --data "E:\CableNNDataSet\ZCode\field_train_v2" --split field_val `
    --weights runs\v3_mixed_long\best.pt --budget 3.0 `
    --out runs\v3_mixed_long\eval_field_val.json

# ---- 步骤4：blind 盲测评估（最终汇报一次性运行，执行前确认不再迭代！） ----
& "E:\Anaconda\envs\gpushare_cu124\python.exe" evaluate.py `
    --data "E:\CableNNDataSet\ZCode\field_train_v2" --split blind `
    --weights runs\v3_mixed_long\best.pt --budget 3.0 `
    --out runs\v3_mixed_long\eval_blind.json
