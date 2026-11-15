param(
    [string]$Groups = "review\comparison_groups.json",
    [string]$Labels = "review\accepted_labels.json",
    [string]$OutputDir = "output\field_cache"
)

$ErrorActionPreference = "Stop"
$ExperimentRoot = $PSScriptRoot
$PrepareScript = Join-Path $ExperimentRoot "scripts\prepare_field.py"

Push-Location $ExperimentRoot
try {
    $CondaArguments = @(
        "run", "--no-capture-output", "-n", "gpushare_cu124", "python", $PrepareScript,
        "--groups", $Groups,
        "--labels", $Labels,
        "--output-dir", $OutputDir
    )
    & conda @CondaArguments
    if ($LASTEXITCODE -ne 0) {
        throw "prepare_field.py exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
