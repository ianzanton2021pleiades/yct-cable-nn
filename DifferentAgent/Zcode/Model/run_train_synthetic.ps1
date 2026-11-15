# 合成预训练 90 epochs（全量 28050 训练 / 1600 验证样本，约 20-30 小时）
# 可随时 Ctrl+C 中断；重新执行同一脚本即从 runs\unet_v2_main\last.pt 断点续训
# v2主配置：多尺度对比度前端 + 按通道pos_weight + 自适应难例挖掘 + dsconv(width48)
cd $PSScriptRoot
& "E:\Anaconda\envs\gpushare_cu124\python.exe" train.py `
    --data "E:\CableNNDataSet\ZCode\DG_V4_S1" `
    --train-split train --val-split synth_val `
    --arch v2 --width 48 --dsconv --hard-k 0 `
    --out runs\unet_v2_main --epochs 90 --batch 16 --lr 2e-3 --workers 6 `
    --val-limit 0 --resume
