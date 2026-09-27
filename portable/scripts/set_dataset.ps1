# set_dataset.ps1 — point both engines at a dataset/threshold (BOM-safe).
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File set_dataset.ps1 -Ds mushrooms.txt -Sup 0.3
param(
    [Parameter(Mandatory=$true)][string]$Ds,
    [Parameter(Mandatory=$true)][string]$Sup,
    [string]$Root = (Split-Path -Parent $PSScriptRoot)   # package root; engines live in anyfim_bmma\src
)
$ErrorActionPreference = 'Stop'
# Defensive: cmd may pass -Root "C:\...\pkg\" where the \" escaped the quote -
# strip any trailing quote/backslash debris so Join-Path always works.
$Root = $Root.TrimEnd('"').TrimEnd('\')
$srcDir = Join-Path $Root 'anyfim_bmma\src'
# C++ string literal needs escaped backslashes: "..\\TransactionSets\\file"
$newline = "CString transSetFile = _T(`"..\\TransactionSets\\$Ds`"); double supportThreshold = $Sup;"
$utf8Bom = New-Object System.Text.UTF8Encoding($true)
foreach ($f in 'kernel_bmma.cu', 'kernel_bitmap.cu') {
    $p = Join-Path $srcDir $f
    $t = [System.IO.File]::ReadAllText($p)
    $t2 = [regex]::Replace($t, '(?m)^CString transSetFile = .*$', $newline, 1)
    if ($t2 -eq $t) { Write-Host "[WARN] no active transSetFile line in $f" }
    [System.IO.File]::WriteAllText($p, $t2, $utf8Bom)
}
Write-Host "dataset -> $Ds @ $Sup"
