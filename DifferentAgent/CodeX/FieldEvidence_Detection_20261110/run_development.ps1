param(
    [int]$Epochs = 40
)

$ErrorActionPreference = "Stop"
$ExperimentRoot = $PSScriptRoot
$OutputRoot = Join-Path $ExperimentRoot "output"
$RunName = "development_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
$RunRoot = Join-Path $OutputRoot $RunName
$ConsoleLog = Join-Path $RunRoot "console.log"
$TranscriptStarted = $false

Push-Location -LiteralPath $ExperimentRoot
try {
    New-Item -ItemType Directory -Path $RunRoot -Force | Out-Null
    Start-Transcript -Path $ConsoleLog -Force | Out-Null
    $TranscriptStarted = $true

    $SourceConfigPath = Join-Path $ExperimentRoot "config\first_round.json"
    $SourceConfigSnapshot = Join-Path $RunRoot "config_source.json"
    $FoldSplitRoot = Join-Path $RunRoot "fold_splits"
    $FoldConfigRoot = Join-Path $RunRoot "fold_configs"
    $CacheRoot = Join-Path $OutputRoot "field_cache"
    $TrainScript = Join-Path $ExperimentRoot "train.py"
    $EvaluateScript = Join-Path $ExperimentRoot "evaluate.py"

    Copy-Item -LiteralPath $SourceConfigPath -Destination $SourceConfigSnapshot
    New-Item -ItemType Directory -Path $FoldSplitRoot, $FoldConfigRoot -Force | Out-Null

    $Folds = @(
        [pscustomobject]@{
            Name = "fold1_A02"
            Train = @("physical-case-A03", "physical-case-A04")
            Val = @("physical-case-A02")
        },
        [pscustomobject]@{
            Name = "fold2_A03"
            Train = @("physical-case-A02", "physical-case-A04")
            Val = @("physical-case-A03")
        },
        [pscustomobject]@{
            Name = "fold3_A04"
            Train = @("physical-case-A02", "physical-case-A03")
            Val = @("physical-case-A04")
        }
    )
    $FoldResults = [System.Collections.Generic.List[object]]::new()

    Write-Host "三案例 leave-one-case-out 开发试运行：A02、A03、A04；训练臂 A；epochs=$Epochs"
    Write-Host "运行目录：$RunRoot"

    foreach ($Fold in $Folds) {
        $SplitPath = Join-Path $FoldSplitRoot "$($Fold.Name).json"
        $ConfigPath = Join-Path $FoldConfigRoot "$($Fold.Name).json"
        $FoldRunRoot = Join-Path $RunRoot $Fold.Name
        $EvaluationPath = Join-Path $FoldRunRoot "evaluation.json"

        $Split = [ordered]@{
            train = @($Fold.Train)
            val = @($Fold.Val)
        }
        $Split | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $SplitPath -Encoding utf8

        $RunConfig = Get-Content -LiteralPath $SourceConfigPath -Raw | ConvertFrom-Json
        $RunConfig.training.epochs = $Epochs
        $RunConfig | ConvertTo-Json -Depth 32 | Set-Content -LiteralPath $ConfigPath -Encoding utf8

        Write-Host ""
        Write-Host "[$($Fold.Name)] train=$($Fold.Train -join ', ') ; val=$($Fold.Val -join ', ')"
        $TrainArguments = @(
            "run", "--no-capture-output", "-n", "gpushare_cu124", "python", "-u", $TrainScript,
            "--arm", "A",
            "--data-dir", $CacheRoot,
            "--split", $SplitPath,
            "--config", $ConfigPath,
            "--run-dir", $FoldRunRoot
        )
        & conda @TrainArguments
        if ($LASTEXITCODE -ne 0) {
            throw "train.py failed for $($Fold.Name), exit code $LASTEXITCODE"
        }

        $BestCheckpoint = Join-Path $FoldRunRoot "best.pt"
        $EvaluateArguments = @(
            "run", "--no-capture-output", "-n", "gpushare_cu124", "python", "-u", $EvaluateScript,
            "--checkpoint", $BestCheckpoint,
            "--data-dir", $CacheRoot,
            "--output", $EvaluationPath
        )
        & conda @EvaluateArguments
        if ($LASTEXITCODE -ne 0) {
            throw "evaluate.py failed for $($Fold.Name), exit code $LASTEXITCODE"
        }

        $FoldResults.Add([pscustomobject]@{
            fold = $Fold.Name
            train_case_groups = @($Fold.Train)
            val_case_groups = @($Fold.Val)
            checkpoint = $BestCheckpoint
            last_checkpoint = Join-Path $FoldRunRoot "last.pt"
            run_info = Join-Path $FoldRunRoot "run_info.json"
            history = Join-Path $FoldRunRoot "history.csv"
            evaluation = $EvaluationPath
            console_log = $ConsoleLog
        })
    }

    $Summary = [ordered]@{
        run_name = $RunName
        run_type = "three_case_leave_one_case_out_development"
        arm = "A"
        epochs = $Epochs
        data_dir = $CacheRoot
        config_source = $SourceConfigSnapshot
        console_log = $ConsoleLog
        folds = @($FoldResults.ToArray())
    }
    $SummaryJsonPath = Join-Path $RunRoot "summary.json"
    $SummaryMarkdownPath = Join-Path $RunRoot "summary.md"
    $Summary | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $SummaryJsonPath -Encoding utf8

    $SummaryLines = @(
        "# 三案例开发试运行"
        ""
        "训练臂 A，按物理案例 leave-one-case-out；每折列出验证案例与结果文件入口。"
        ""
        "| Fold | 验证案例 | 训练案例 | Checkpoint | History | Evaluation |"
        "|---|---|---|---|---|---|"
    )
    foreach ($Result in $FoldResults) {
        $RelativeCheckpoint = [System.IO.Path]::GetRelativePath($RunRoot, $Result.checkpoint)
        $RelativeHistory = [System.IO.Path]::GetRelativePath($RunRoot, $Result.history)
        $RelativeEvaluation = [System.IO.Path]::GetRelativePath($RunRoot, $Result.evaluation)
        $SummaryLines += "| $($Result.fold) | $($Result.val_case_groups -join ', ') | $($Result.train_case_groups -join ', ') | ``$RelativeCheckpoint`` | ``$RelativeHistory`` | ``$RelativeEvaluation`` |"
    }
    $SummaryLines += @(
        ""
        "完整控制台记录：``console.log``。本轮只覆盖A02、A03、A04；B06保持未知，B臂未运行。"
    )
    $SummaryLines | Set-Content -LiteralPath $SummaryMarkdownPath -Encoding utf8

    Write-Host ""
    Write-Host "三折训练与评价完成。汇总：$SummaryMarkdownPath"
}
finally {
    try {
        if ($TranscriptStarted) {
            Stop-Transcript | Out-Null
        }
    }
    finally {
        Pop-Location
    }
}
