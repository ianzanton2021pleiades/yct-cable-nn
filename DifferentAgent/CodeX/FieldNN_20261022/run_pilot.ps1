[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Generate", "Audit", "Train", "Evaluate", "Field")]
    [string]$Stage,
    [ValidateSet("all", "mamba", "transformer", "unet")]
    [string]$Model = "all",
    [ValidateRange(1, 100000)]
    [int]$Epochs = 5,
    [switch]$Resume,
    [string]$DataRoot = "E:\CableNNDataSet\CodeX\FieldNN_20261022"
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)

$PythonPath = "E:\Anaconda\envs\gpushare_cu124\python.exe"
$ExperimentRoot = $PSScriptRoot
$RunRoot = Join-Path $ExperimentRoot "output\pilot_v1"
$AuditRoot = Join-Path $ExperimentRoot "output\audit"
$RealManifest = Join-Path $ExperimentRoot "output\real_manifest.jsonl"
$ModelNames = if ($Model -eq "all") {
    @("mamba", "transformer", "unet")
} else {
    @($Model)
}

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Required Python environment was not found: $PythonPath"
}

function Invoke-PythonScript {
    param(
        [Parameter(Mandatory = $true)][string]$EntryPoint,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    & $script:PythonPath -X utf8 -B $EntryPoint @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw ("Python command failed with exit code {0}: {1} {2}" -f $exitCode, $EntryPoint, ($Arguments -join " "))
    }
}

switch ($Stage) {
    "Generate" {
        Invoke-PythonScript -EntryPoint (Join-Path $ExperimentRoot "generate_dataset.py") -Arguments @("--output", $DataRoot, "--resume")
    }
    "Audit" {
        Invoke-PythonScript -EntryPoint (Join-Path $ExperimentRoot "audit_dataset.py") -Arguments @("--data", $DataRoot, "--max-samples", "128", "--output", $AuditRoot)
    }
    "Train" {
        $arguments = @("--data", $DataRoot, "--model", $Model, "--output", $RunRoot, "--epochs", "$Epochs")
        if ($Resume) {
            $arguments += "--resume"
        }
        Invoke-PythonScript -EntryPoint (Join-Path $ExperimentRoot "train.py") -Arguments $arguments
    }
    "Evaluate" {
        foreach ($name in $ModelNames) {
            $modelRoot = Join-Path $RunRoot $name
            $checkpoint = Join-Path $modelRoot "best.pt"
            if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
                throw "Train $name first; checkpoint not found: $checkpoint"
            }
            $valOutput = Join-Path $modelRoot "evaluation_val"
            $testOutput = Join-Path $modelRoot "evaluation_test"
            Invoke-PythonScript -EntryPoint (Join-Path $ExperimentRoot "evaluate.py") -Arguments @("--data", $DataRoot, "--checkpoint", $checkpoint, "--split", "val", "--calibrate", "--output", $valOutput)
            Invoke-PythonScript -EntryPoint (Join-Path $ExperimentRoot "evaluate.py") -Arguments @("--data", $DataRoot, "--checkpoint", $checkpoint, "--split", "test", "--output", $testOutput)
        }
    }
    "Field" {
        Invoke-PythonScript -EntryPoint (Join-Path $ExperimentRoot "prepare_real_manifest.py") -Arguments @("--output", $RealManifest)
        foreach ($name in $ModelNames) {
            $modelRoot = Join-Path $RunRoot $name
            $checkpoint = Join-Path $modelRoot "best.pt"
            if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
                throw "Train $name first; checkpoint not found: $checkpoint"
            }
            $fieldOutput = Join-Path $modelRoot "field_development"
            Invoke-PythonScript -EntryPoint (Join-Path $ExperimentRoot "infer.py") -Arguments @("--checkpoint", $checkpoint, "--manifest", $RealManifest, "--subset", "development", "--output", $fieldOutput)
        }
    }
}
