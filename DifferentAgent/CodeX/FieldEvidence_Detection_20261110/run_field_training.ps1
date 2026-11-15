param(
    [ValidateSet("A", "B")]
    [string]$Arm = "A",
    [string]$DataDir = "output\field_cache",
    [string]$Split = "config\split_template.json",
    [string]$Config = "config\first_round.json",
    [string]$RunDir = "",
    [string]$EncoderInit = ""
)

$ErrorActionPreference = "Stop"
$ExperimentRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($RunDir)) {
    $RunDir = Join-Path "output\field_training" $Arm
}
if ($Arm -eq "B" -and [string]::IsNullOrWhiteSpace($EncoderInit)) {
    throw "B臂需要-EncoderInit指定共享单相encoder初始化文件；现有单条合成电缆数据不会拼成三相真值。"
}

Push-Location $ExperimentRoot
try {
    $CondaArguments = @(
        "run", "--no-capture-output", "-n", "gpushare_cu124", "python",
        (Join-Path $ExperimentRoot "train.py"),
        "--arm", $Arm,
        "--data-dir", $DataDir,
        "--split", $Split,
        "--config", $Config,
        "--run-dir", $RunDir
    )
    if ($Arm -eq "B") {
        $CondaArguments += @("--encoder-init", $EncoderInit)
    }
    & conda @CondaArguments
    if ($LASTEXITCODE -ne 0) {
        throw "train.py exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
