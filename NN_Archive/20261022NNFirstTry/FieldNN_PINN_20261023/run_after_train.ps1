$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

$base = "D:\GitRepository\Cable-NN\NN_Archive\20261022NNFirstTry\FieldNN_PINN_20261023"
$py = "E:\Anaconda\envs\gpushare_cu124\python.exe"
$out = Join-Path $base "output"

Write-Output "=== 1. stitch two phases ==="
& $py "$base\stitch_history.py" --phase1 "$out\pinn100_phase1\unet" --phase2 "$out\pinn100_phase2\unet" --output-dir "$out\pinn100_combined"

Write-Output "=== 2. evaluate epoch-50 snapshot ==="
& $py "$base\eval_pinn.py" --checkpoint "$out\pinn100_phase1\unet\epoch050.pt" --output "$out\eval_epoch050.json"

Write-Output "=== 3. evaluate 100-epoch best ==="
& $py "$base\eval_pinn.py" --checkpoint "$out\pinn100_phase2\unet\best.pt" --output "$out\eval_100.json"

Write-Output "=== 4. field adaptation ==="
& $py "$base\finetune_field.py" --checkpoint "$out\pinn100_phase2\unet\best.pt" --output "$out\field_ft" --epochs 12 --workers 3

Write-Output "=== 5. evaluate field model ==="
& $py "$base\eval_pinn.py" --checkpoint "$out\field_ft\best.pt" --output "$out\eval_field.json"

Write-Output "=== 6. threshold sensitivity ==="
& $py "$base\sweep_thresholds.py" --checkpoint "$out\field_ft\best.pt" --output "$out\threshold_sweep.json"

Write-Output "=== 7. field cases ==="
& $py "$base\case_study_pinn.py" --checkpoint "$out\field_ft\best.pt" --prefix 20261023_PINN

Write-Output "=== 8. curves ==="
& $py "$base\plot_curves.py" --history "$out\pinn100_combined\unet\history.csv" --steps "$out\pinn100_combined\unet\steps.csv" --output "$base\assets\20261023_PINN_curves.png" --label "PINN U-Net"

Write-Output "=== 9. numbers ==="
& $py "$base\report_numbers.py" --mine "$out\pinn100_combined\unet\history.csv" --output "$out\numbers.json"

Write-Output "all done"
