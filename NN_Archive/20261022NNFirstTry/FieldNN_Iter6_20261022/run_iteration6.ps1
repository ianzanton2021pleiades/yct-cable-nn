param(
    [string]$Data='E:\CableNNDataSet\CodeX\FieldNN_Iter6_20261022',
    [string]$Synthetic='E:\CableNNDataSet\CodeX\FieldNN_Iter4_20261022',
    [string]$Output="$PSScriptRoot\output\iter6",
    [string]$PythonPath='E:\Anaconda\envs\gpushare_cu124\python.exe',
    [string]$Models='unet,transformer',
    [string]$Device='cuda',
    [int]$Epochs=18,
    [int]$BatchSize=64,
    [int]$Workers=6,
    [switch]$Resume,
    [switch]$Smoke
)

$ErrorActionPreference='Stop'
$RunArgs=@(
    '-X','utf8','-B',"$PSScriptRoot\run_iteration6.py",
    '--data',$Data,'--synthetic',$Synthetic,'--output',$Output,
    '--models',$Models,'--device',$Device,
    '--epochs',$Epochs,'--batch-size',$BatchSize,'--workers',$Workers
)
if($Resume){$RunArgs+='--resume'}
if($Smoke){$RunArgs+='--smoke'}

& $PythonPath @RunArgs
if($LASTEXITCODE -ne 0){throw "Iter6 中断，退出码 $LASTEXITCODE；请保留 output 日志。"}
