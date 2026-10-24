param(
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
    '-X','utf8','-B',"$PSScriptRoot\run_iteration8.py",
    '--models',$Models,'--device',$Device,
    '--epochs',$Epochs,'--batch-size',$BatchSize,'--workers',$Workers
)
if($Resume){$RunArgs+='--resume'}
if($Smoke){$RunArgs+='--smoke'}

& $PythonPath @RunArgs
if($LASTEXITCODE -ne 0){throw "Iter8 中断，退出码 $LASTEXITCODE；请保留 output 日志。"}
