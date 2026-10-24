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
$RunArgs=@('-X','utf8','-B',$Runner,'--output',"$PSScriptRoot\output\iter12_50ep",
    '--label','Iter12-Transformer-stem','--model','transformer',
    '--epochs',$Epochs,'--record-every','100','--batch-size',$BatchSize,
    '--workers',$Workers,'--device','cuda','--stem-only',
    '--weak-weight','0.2','--stem-lr','0.00003')
if($Resume){$RunArgs+='--resume'}
& $PythonPath @RunArgs
if($LASTEXITCODE -ne 0){throw "Iter12中断，退出码 $LASTEXITCODE；请保留output日志。"}
