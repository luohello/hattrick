param(
    [string]$Python = "C:\Users\12634\anaconda3\envs\hattrick-local\python.exe"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ManifestPath = Join-Path $ProjectRoot "manifest\geant_manifest.txt"
$SolverPath = Join-Path $ProjectRoot "frameworks\gurobi_refactored.py"
$ResultsRoot = Join-Path $ProjectRoot "results\geant\4sp"
$StateRoot = Join-Path $ResultsRoot ".full_solve_state"
$ExpectedSnapshots = 10772

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python executable not found: $Python"
}
if (-not (Test-Path -LiteralPath $ManifestPath)) {
    throw "GEANT manifest not found: $ManifestPath"
}
if (-not (Test-Path -LiteralPath "$env:USERPROFILE\gurobi.lic")) {
    throw "Gurobi license not found at $env:USERPROFILE\gurobi.lic"
}

$ManifestCount = (Get-Content -LiteralPath $ManifestPath).Count
if ($ManifestCount -ne $ExpectedSnapshots) {
    throw "Expected $ExpectedSnapshots GEANT manifest rows, found $ManifestCount"
}

foreach ($ClassId in 1..3) {
    $TrafficDir = Join-Path $ProjectRoot "traffic_matrices\geant_$ClassId"
    $TrafficCount = (Get-ChildItem -LiteralPath $TrafficDir -File -Filter "*.pkl").Count
    if ($TrafficCount -ne $ExpectedSnapshots) {
        throw "Expected $ExpectedSnapshots matrices in geant_$ClassId, found $TrafficCount"
    }
}

New-Item -ItemType Directory -Force -Path $StateRoot | Out-Null
Set-Location -LiteralPath $ProjectRoot
$env:PYTHONUNBUFFERED = "1"

$Stages = @(
    @{ Name = "01_mf";          Priority = 1; Objectives = @("mf");                Output = "gt_optimal_values_mf.txt" },
    @{ Name = "02_mf_mf";       Priority = 2; Objectives = @("mf", "mf");          Output = "gt_optimal_values_mf_mf.txt" },
    @{ Name = "03_mf_mf_mf";    Priority = 3; Objectives = @("mf", "mf", "mf");    Output = "gt_optimal_values_mf_mf_mf.txt" },
    @{ Name = "04_mlu";         Priority = 1; Objectives = @("mlu");               Output = "gt_optimal_values_mlu.txt" },
    @{ Name = "05_mlu_mlu";     Priority = 2; Objectives = @("mlu", "mlu");        Output = "gt_optimal_values_mlu_mlu.txt" },
    @{ Name = "06_mlu_mlu_mlu"; Priority = 3; Objectives = @("mlu", "mlu", "mlu"); Output = "gt_optimal_values_mlu_mlu_mlu.txt" }
)

function Get-ResultLineCount {
    param([Parameter(Mandatory = $true)][string]$FileName)

    $Files = @(Get-ChildItem -LiteralPath $ResultsRoot -Directory |
        ForEach-Object { Join-Path $_.FullName $FileName } |
        Where-Object { Test-Path -LiteralPath $_ })

    $Total = 0
    foreach ($File in $Files) {
        $Total += (Get-Content -LiteralPath $File).Count
    }
    return $Total
}

$RunStarted = Get-Date
Write-Output "[$($RunStarted.ToString('s'))] Full GEANT optimization started"
Write-Output "Project: $ProjectRoot"
Write-Output "Snapshots: $ExpectedSnapshots"

foreach ($Stage in $Stages) {
    $DoneMarker = Join-Path $StateRoot "$($Stage.Name).done"
    if (Test-Path -LiteralPath $DoneMarker) {
        $ExistingCount = Get-ResultLineCount -FileName $Stage.Output
        if ($ExistingCount -eq $ExpectedSnapshots) {
            Write-Output "[$((Get-Date).ToString('s'))] Skipping completed stage $($Stage.Name)"
            continue
        }
        Remove-Item -LiteralPath $DoneMarker -Force
    }

    $StageStarted = Get-Date
    Write-Output "[$($StageStarted.ToString('s'))] Starting stage $($Stage.Name)"

    $Arguments = @(
        $SolverPath,
        "--num_paths_per_pair", "4",
        "--opt_start_idx", "0",
        "--opt_end_idx", "$ExpectedSnapshots",
        "--topo", "geant",
        "--framework", "gurobi",
        "--pred", "0",
        "--pred_type", "esm",
        "--cluster", "0",
        "--priority", "$($Stage.Priority)",
        "--objs"
    ) + $Stage.Objectives + @(
        "--gur_mode", "flexile",
        "--tol", "0.000001"
    )

    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Stage $($Stage.Name) failed with exit code $LASTEXITCODE"
    }

    $ResultCount = Get-ResultLineCount -FileName $Stage.Output
    if ($ResultCount -ne $ExpectedSnapshots) {
        throw "Stage $($Stage.Name) produced $ResultCount results; expected $ExpectedSnapshots"
    }

    $StageFinished = Get-Date
    "completed=$($StageFinished.ToString('o'))`nresults=$ResultCount" |
        Set-Content -LiteralPath $DoneMarker -Encoding UTF8
    Write-Output "[$($StageFinished.ToString('s'))] Completed stage $($Stage.Name) in $([math]::Round(($StageFinished - $StageStarted).TotalMinutes, 2)) minutes"
}

$RunFinished = Get-Date
Write-Output "[$($RunFinished.ToString('s'))] Full GEANT optimization completed in $([math]::Round(($RunFinished - $RunStarted).TotalHours, 2)) hours"
