# 模型v3 阶段1：合成预训练 70 epochs（DG_V5_S1）
# 每轮打印 field_val F1@20 迁移监控列（只监控，不选型；选型=synth_val loss）
# 断点续训：随时 Ctrl+C，重新执行本脚本即从 runs\v3_pretrain\last.pt 恢复
cd $PSScriptRoot
& "E:\Anaconda\envs\gpushare_cu124\python.exe" train.py `
    --stage pretrain `
    --synth-data "E:\CableNNDataSet\ZCode\DG_V5_S1" `
    --field-data "E:\CableNNDataSet\ZCode\field_train_v2" `
    --arch v1 --out runs\v3_pretrain `
    --epochs 70 --batch 16 --lr 2e-3 --workers 6 `
    --val-limit 0 --resume
