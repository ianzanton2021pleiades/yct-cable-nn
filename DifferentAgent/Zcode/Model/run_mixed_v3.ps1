# 模型v3 阶段2：混合微调 35 epochs（field_train 295相 + 每轮约25%合成回放）
# 先完成 run_pretrain_v3.ps1（runs\v3_pretrain\best.pt 存在）再执行
# 选型 = field_val F1@20；synth_val loss 作为防遗忘监控列打印
# 断点续训：随时 Ctrl+C，重新执行本脚本即从 runs\v3_mixed\last.pt 恢复
cd $PSScriptRoot
& "E:\Anaconda\envs\gpushare_cu124\python.exe" train.py `
    --stage mixed `
    --synth-data "E:\CableNNDataSet\ZCode\DG_V5_S1" `
    --field-data "E:\CableNNDataSet\ZCode\field_train_v2" `
    --init runs\v3_pretrain\best.pt `
    --arch v1 --out runs\v3_mixed `
    --epochs 35 --batch 16 --lr 2e-4 --workers 4 `
    --val-limit 300 --resume
