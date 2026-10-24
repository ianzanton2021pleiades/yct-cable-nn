param([int]$Epochs=20,[int]$BatchSize=32,[int]$Workers=7,[switch]$Resume,[switch]$SkipField,[string]$Models='unet,transformer',[switch]$Smoke)
$ErrorActionPreference='Stop'
$PythonPath='E:\Anaconda\envs\gpushare_cu124\python.exe'
$RunArgs=@('-X','utf8','-B',"$PSScriptRoot\run_iteration5.py",'--epochs',$Epochs,'--batch-size',$BatchSize,'--workers',$Workers,'--models',$Models)
if($Resume){$RunArgs+='--resume'}
if($SkipField){$RunArgs+='--skip-field'}
if($Smoke){$RunArgs+='--models','unet';$RunArgs+='--epochs','1'}
& $PythonPath @RunArgs
if($LASTEXITCODE -ne 0){throw "Iter5 中断，退出码 $LASTEXITCODE；请保留 output 日志。"}
