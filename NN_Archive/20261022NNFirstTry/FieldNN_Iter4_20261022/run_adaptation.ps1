param([int]$Epochs=24,[int]$BatchSize=32,[int]$Workers=6,[switch]$Resume,[switch]$SkipPrepare,[switch]$Smoke)
$ErrorActionPreference='Stop'
$PythonPath='E:\Anaconda\envs\gpushare_cu124\python.exe'
$RunArgs=@('-X','utf8','-B',"$PSScriptRoot\run_adaptation.py",'--epochs',$Epochs,'--batch-size',$BatchSize,'--workers',$Workers)
if($Resume){$RunArgs+='--resume'}
if($SkipPrepare){$RunArgs+='--skip-prepare'}
if($Smoke){$RunArgs+='--smoke'}
& $PythonPath @RunArgs
if($LASTEXITCODE -ne 0){throw "适应训练中断，退出码 $LASTEXITCODE；请保留output日志。"}
