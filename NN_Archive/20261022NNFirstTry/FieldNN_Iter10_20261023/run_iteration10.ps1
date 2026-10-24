param(
    [string]$PythonPath='E:\Anaconda\envs\gpushare_cu124\python.exe',
    [string]$Device='cuda',
    [int]$Epochs=50,
    [int]$Steps=0,
    [int]$RecordEvery=100,
    [int]$BatchSize=64,
    [int]$Workers=6,
    [switch]$Resume
)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$RunArgs=@('-X','utf8','-B',"$PSScriptRoot\run_iteration10.py",
    '--device',$Device,'--epochs',$Epochs,'--record-every',$RecordEvery,
    '--batch-size',$BatchSize,'--workers',$Workers)
if($Steps -gt 0){$RunArgs+=@('--steps',$Steps)}
if($Resume){$RunArgs+='--resume'}
& $PythonPath @RunArgs
if($LASTEXITCODE -ne 0){throw "Iter10中断，退出码 $LASTEXITCODE；保留output日志，可加-Resume继续。"}
