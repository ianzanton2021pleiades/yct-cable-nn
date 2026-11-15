# 现场微调（先完成合成预训练再执行；约1-2小时）
# 加载合成预训练最优权重，低学习率微调，现场验证集选型
# 架构参数必须与预训练一致（v2/width48/dsconv），否则权重无法加载
cd $PSScriptRoot
& "E:\Anaconda\envs\gpushare_cu124\python.exe" train.py `
    --data "E:\CableNNDataSet\ZCode\field_train" `
    --train-split field_train --val-split field_val `
    --init runs\unet_v2_main\best.pt `
    --arch v2 --width 48 --dsconv --hard-k 0 `
    --out runs\unet_v2_field --epochs 30 --batch 12 --lr 2e-4 --workers 4 --resume

# 现场验证集评估（固定候选预算3.0）
& "E:\Anaconda\envs\gpushare_cu124\python.exe" evaluate.py `
    --data "E:\CableNNDataSet\ZCode\field_train" --split field_val `
    --arch v2 --width 48 `
    --weights runs\unet_v2_field\best.pt --budget 3.0 `
    --out runs\unet_v2_field\eval_field_val.json

# 封存盲测集评估（只允许在最终汇报时运行一次！）
& "E:\Anaconda\envs\gpushare_cu124\python.exe" evaluate.py `
    --data "E:\CableNNDataSet\ZCode\field_train" --split blind `
    --arch v2 --width 48 `
    --weights runs\unet_v2_field\best.pt --budget 3.0 `
    --out runs\unet_v2_field\eval_blind.json
