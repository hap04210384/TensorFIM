# compare.ps1 — harvest logs, verify itemsets against archived references, write REPORT.txt.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File compare.ps1 [-ConfigFile <path>] [-LogDir <run folder>]
param([string]$ConfigFile = "", [string]$LogDir = "")
$ErrorActionPreference = 'Continue'
$Root    = Split-Path -Parent $PSScriptRoot
if (-not $LogDir) { $LogDir = Join-Path $Root 'logs' }
# References live in a dedicated folder: engine runs write fresh Results next to
# the datasets, so the archived gold copies must be kept out of harm's way.
$TsDir   = Join-Path $Root 'references'
if (-not $ConfigFile) { $ConfigFile = Join-Path $PSScriptRoot 'configs.txt' }
$Report  = Join-Path $LogDir 'REPORT.txt'

function Get-Itemsets($path) {
    $s = @{}
    Get-Content $path -Encoding UTF8 | ForEach-Object {
        if ($_ -match '\{([^}]*)\}') {
            $items = $Matches[1] -split '\s+' | Where-Object { $_ -ne '' } |
                     ForEach-Object { [int]$_ } | Sort-Object
            if ($items.Count -gt 0) { $s[($items -join ',')] = $true }
        }
    }
    return $s
}

function Get-TotalTime($path) {
    $m = Select-String -Path $path -Pattern 'total_time\(s\):\s*([\d.]+)' |
         Select-Object -Last 1
    if ($m) { return [double]$m.Matches[0].Groups[1].Value } else { return $null }
}

function Get-MFIsNumber($path) {
    $m = Select-String -Path $path -Pattern 'MFIsNumber:\s*(\d+)' | Select-Object -Last 1
    if ($m) { return [int]$m.Matches[0].Groups[1].Value } else { return -1 }
}

$out = New-Object System.Collections.Generic.List[string]
function W([string]$s) { $out.Add($s); Write-Host $s }

W "======================================================================="
W " TensorFIM cross-platform validation report"
W " generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
W "======================================================================="
W ""
W "--- Machine ---"
try {
    $gpu = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null
    if ($gpu -and ($gpu -notmatch 'Failed|Error')) { W "GPU: $gpu" }
} catch {}
if (Test-Path (Join-Path $LogDir 'sm.txt')) { W (Get-Content (Join-Path $LogDir 'sm.txt') -Raw).Trim() }
$cpu = (Get-CimInstance Win32_Processor).Name
$ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
W "CPU: $cpu"
W "RAM: ${ram} GB"
$os = (Get-CimInstance Win32_OperatingSystem).Caption
W "OS:  $os"
W ""

$allOk = $true
$table = New-Object System.Collections.Generic.List[string]
$table.Add(("{0,-18}{1,8}  {2,-9}{3,8}{4,12}  {5}" -f 'dataset','sup','engine','MFIs','med_time(s)','result'))

foreach ($line in Get-Content $ConfigFile) {
    $line = $line.Trim()
    if ($line -eq '' -or $line.StartsWith('#')) { continue }
    $parts = $line -split '\s+'
    $ds = $parts[0]; $supRaw = $parts[1]; $sup = [double]$supRaw
    $refPath = Join-Path $TsDir ("{0}-{1:F6}=Results.txt" -f $ds, $sup)
    $ref = if (Test-Path $refPath) { Get-Itemsets $refPath } else { $null }

    $rowsBefore = $table.Count
    foreach ($engine in 'baseline', 'bmma') {
        $logs = @(Get-ChildItem $LogDir -Filter "${ds}_${supRaw}_${engine}_r*.log" -ErrorAction SilentlyContinue |
                Sort-Object Name)
        if ($logs.Count -eq 0) { continue }
        $times = @($logs | ForEach-Object { Get-TotalTime $_.FullName } | Where-Object { $_ -ne $null } | Sort-Object)
        $med = if ($times.Count -gt 0) { $times[[int]($times.Count / 2)] } else { -1 }
        $first = $logs[0].FullName
        $mnum = Get-MFIsNumber $first
        $mine = Get-Itemsets $first
        $same = ($null -ne $ref) -and ($mine.Count -eq $ref.Count)
        if ($same) { foreach ($k in $mine.Keys) { if (-not $ref.ContainsKey($k)) { $same = $false; break } } }
        $ok = $same -and ($mnum -eq $mine.Count)
        if (-not $ok) { $script:allOk = $false }
        $table.Add(("{0,-18}{1,8}  {2,-9}{3,8}{4,12}  {5}" -f $ds, $supRaw, $engine, $mnum, $med, $(if ($ok) {'PASS'} else {'FAIL'})))
    }
    if ($table.Count -eq $rowsBefore) {
        # no logs matched this config at all - never pass silently
        $table.Add(("{0,-18}{1,8}  {2,-9}{3,8}{4,12}  {5}" -f $ds, $supRaw, '-', '-', '-', 'MISSING'))
        $script:allOk = $false
    }
}
W "--- Results (median of repetitions; correctness vs archived reference) ---"
W ($table -join "`r`n")
W ""
W ($(if ($allOk) { 'ALL PASS - results match the archived references exactly; timing data is valid for the paper' } else { 'SOME CONFIGS FAILED - do NOT use the timing data from this run; bring back REPORT.txt and the logs for analysis' }))
[System.IO.File]::WriteAllLines($Report, $out, (New-Object System.Text.UTF8Encoding($true)))
Write-Host ""
Write-Host "Report written to: $Report"
