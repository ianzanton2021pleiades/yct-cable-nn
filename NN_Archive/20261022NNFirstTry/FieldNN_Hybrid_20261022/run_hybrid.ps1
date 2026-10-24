param(
    [switch]$Smoke,
    [switch]$SkipPrepare,
    [switch]$SkipTrain,
    [switch]$FreezeEncoder,
    [string[]]$Models = @('unet', 'transformer')
)

$ErrorActionPreference = 'Stop'
$Experiment = 'D:\GitRepository\Cable-NN\NN_Archive\20261022NNFirstTry\FieldNN_Hybrid_20261022'
$Python = 'E:\Anaconda\envs\gpushare_cu124\python.exe'

$Arguments = @(
    '-X', 'utf8', '-B', "$Experiment\run_hybrid.py",
    '--models', $Models
)
if ($Smoke) { $Arguments += '--smoke' }
if ($SkipPrepare) { $Arguments += '--skip-prepare' }
if ($SkipTrain) { $Arguments += '--skip-train' }
if ($FreezeEncoder) { $Arguments += '--freeze-encoder' }

& $Python @Arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
