param(
    [string]$PythonPath='E:\Anaconda\envs\gpushare_cu124\python.exe',
    [int]$Epochs=50,
    [int]$BatchSize=96,
    [int]$Workers=6,
    [switch]$Resume
)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$Runner=Join-Path $PSScriptRoot '..\FieldNN_Iter10_20261023\run_iteration10.py'
$RunArgs=@('-X','utf8','-B',$Runner,'--output',"$PSScriptRoot\output\iter11_50ep",
    '--label','Iter11-U-Net-stem','--epochs',$Epochs,'--record-every','100',
    '--batch-size',$BatchSize,'--workers',$Workers,'--device','cuda',
    '--init-iter8','--stem-only','--weak-weight','0.05','--stem-lr','0.00001')
if($Resume){$RunArgs+='--resume'}
& $PythonPath @RunArgs
if($LASTEXITCODE -ne 0){throw "Iter11中断，退出码 $LASTEXITCODE；请保留output日志。"}
