param(
    [int]$Epochs = 40,
    [int]$BatchSize = 32,
    [int]$Workers = 6,
    [switch]$Resume,
    [switch]$SkipGenerate,
    [switch]$Smoke
)
$ErrorActionPreference = 'Stop'
$PythonPath = 'E:\Anaconda\envs\gpushare_cu124\python.exe'
$RunArgs = @('-X', 'utf8', '-B', "$PSScriptRoot\run_iteration.py", '--epochs', $Epochs, '--batch-size', $BatchSize, '--workers', $Workers)
if ($Resume) { $RunArgs += '--resume' }
if ($SkipGenerate) { $RunArgs += '--skip-generate' }
if ($Smoke) { $RunArgs += '--smoke' }
& $PythonPath @RunArgs
if ($LASTEXITCODE -ne 0) { throw "迭代中断，退出码 $LASTEXITCODE。请保留 output 下的日志。" }

