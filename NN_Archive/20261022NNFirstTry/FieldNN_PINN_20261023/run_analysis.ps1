param(
    [Parameter(Mandatory = $true)][string]$Tag,
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [Parameter(Mandatory = $true)][string]$Phase4,
    [Parameter(Mandatory = $true)][int]$Phase4Epochs
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

$base = "D:\GitRepository\Cable-NN\NN_Archive\20261022NNFirstTry\FieldNN_PINN_20261023"
$py = "E:\Anaconda\envs\gpushare_cu124\python.exe"
$out = Join-Path $base "output"
$img = "D:\GitRepository\Cable-NN\Image"
$baseline = "D:\GitRepository\Cable-NN\NN_Archive\20261022NNFirstTry\FieldNN_Hybrid_20261022\output\hybrid_long\unet\evaluation_e05\evaluation.json"

Write-Output "=== stitch ==="
& $py "$base\stitch_history.py" --segment "$out\pinn100_phase1\unet=50" --segment "$out\pinn100_phase2\unet=42" --segment "$out\$Phase4\unet=$Phase4Epochs" --output-dir "$out\hist_$Tag"

Write-Output "=== curves ==="
& $py "$base\plot_history.py" --history "$out\hist_$Tag\unet\history.csv" --output "$img\20261023_PINN_curves_$Tag.png" --label "PINN U-Net" --boundaries "50,92"

Write-Output "=== eval pretrain ==="
& $py "$base\eval_pinn.py" --checkpoint $Checkpoint --output "$out\eval_$Tag.json"

Write-Output "=== field adaptation ==="
& $py "$base\finetune_field.py" --checkpoint $Checkpoint --output "$out\field_ft_$Tag" --epochs 12 --workers 3

Write-Output "=== eval field ==="
& $py "$base\eval_pinn.py" --checkpoint "$out\field_ft_$Tag\best.pt" --output "$out\eval_field_$Tag.json"

Write-Output "=== threshold sweep ==="
& $py "$base\sweep_thresholds.py" --checkpoint "$out\field_ft_$Tag\best.pt" --output "$out\sweep_$Tag.json"

Write-Output "=== cases ==="
& $py "$base\case_study_pinn.py" --checkpoint "$out\field_ft_$Tag\best.pt" --prefix "20261023_PINN$Tag" --count 4

Write-Output "=== gallery ==="
& $py "$base\case_gallery.py" --checkpoint "$out\field_ft_$Tag\best.pt" --output "$img\20261023_PINN${Tag}_gallery.png" --count 6

Write-Output "=== latency ==="
& $py "$base\measure_latency.py" --checkpoint "$out\field_ft_$Tag\best.pt" --output "$out\latency_$Tag.json"

Write-Output "=== metrics figure ==="
& $py "$base\metrics_figure.py" --entry "CMDC baseline=$baseline" "PINN $Tag pretrain=$out\eval_$Tag.json" "PINN $Tag field=$out\eval_field_$Tag.json" --output "$img\20261023_PINN${Tag}_metrics.png"

Write-Output "=== table ==="
& $py "$base\compare_tables.py" --entry "CMDC long U-Net=$baseline" "PINN $Tag pretrain=$out\eval_$Tag.json" "PINN $Tag field=$out\eval_field_$Tag.json" --output "$out\table_$Tag.md"

Write-Output "analysis done"
