#requires -Version 7.0
param([switch]$Apply)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$ArchiveRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'NN_Archive/20261022NNFirstTry'))
$Manifest = Get-Content -LiteralPath (Join-Path $ArchiveRoot 'cleanup_candidates.json') -Raw | ConvertFrom-Json
$Targets = @(foreach ($Entry in $Manifest) {
    $FullPath = [IO.Path]::GetFullPath((Join-Path $ArchiveRoot $Entry.Path))
    if (-not $FullPath.StartsWith($ArchiveRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "目标超出本次归档：$FullPath"
    }
    if (Test-Path -LiteralPath $FullPath) {
        $Files = @(Get-ChildItem -LiteralPath $FullPath -File -Recurse -Force)
        if ($Files | Where-Object Extension -in '.pt', '.pth', '.ckpt') { throw "候选内发现权重，停止：$FullPath" }
        [pscustomobject]@{ Path = $FullPath; Files = $Files.Count; Bytes = [int64](($Files | Measure-Object Length -Sum).Sum); Reason = $Entry.Reason }
    }
})
$Targets | Select-Object Path, Files, @{N='MiB'; E={[math]::Round($_.Bytes/1MB, 3)}}, Reason | Format-Table -AutoSize -Wrap
$Bytes = [int64](($Targets | Measure-Object Bytes -Sum).Sum)
Write-Output "现存候选 $($Targets.Count) 个目录，$([int64](($Targets | Measure-Object Files -Sum).Sum)) 个文件，共 $([math]::Round($Bytes/1MB, 3)) MiB。"
if (-not $Apply) { Write-Output '仅预览；确认建议后，加 -Apply 才会删除。'; return }
foreach ($Target in $Targets) { Remove-Item -LiteralPath $Target.Path -Recurse -Force }
Write-Output '已删除本次清单中的现存缓存/smoke目录。正式结果、正式日志、权重、源码和E盘数据不在清单中。'
