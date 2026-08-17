param(
    [string]$Python = "C:\Users\12634\anaconda3\envs\hattrick-local\python.exe"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$SolverPath = Join-Path $ProjectRoot "frameworks\gurobi_refactored.py"
$ManifestPath = Join-Path $ProjectRoot "manifest\geant_manifest.txt"
$ResultsDir = Join-Path $ProjectRoot "results\geant\8sp\0"
$StateDir = Join-Path $ResultsDir ".baseline_state"
$SwanSplitDir = Join-Path (Split-Path -Parent $ProjectRoot) "scratch\split_ratios\geant\8sp\swan\esm"
$TotalSnapshots = 10772
$TestStart = 8618
$TestEnd = 10772
$ExpectedSnapshots = $TestEnd - $TestStart
$ExpectedSimulationLines = 6 * $ExpectedSnapshots

function Get-LineCount {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return 0
    }
    return (Get-Content -LiteralPath $Path).Count
}

function Get-PickleCount {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return 0
    }
    return (Get-ChildItem -LiteralPath $Path -File -Filter "*.pkl").Count
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python executable not found: $Python"
}
if (-not (Test-Path -LiteralPath $SolverPath)) {
    throw "Gurobi solver script not found: $SolverPath"
}
if (-not (Test-Path -LiteralPath "$env:USERPROFILE\gurobi.lic")) {
    throw "Gurobi license not found at $env:USERPROFILE\gurobi.lic"
}

$ManifestCount = Get-LineCount -Path $ManifestPath
if ($ManifestCount -ne $TotalSnapshots) {
    throw "Expected $TotalSnapshots GEANT manifest rows, found $ManifestCount"
}

foreach ($ClassId in 1..3) {
    foreach ($Suffix in @("", "_esm")) {
        $DirectoryName = "geant_$ClassId$Suffix"
        $TrafficDir = Join-Path $ProjectRoot "traffic_matrices\$DirectoryName"
        $TrafficCount = Get-PickleCount -Path $TrafficDir
        if ($TrafficCount -ne $TotalSnapshots) {
            throw "Expected $TotalSnapshots matrices in $DirectoryName, found $TrafficCount"
        }
    }
}

New-Item -ItemType Directory -Force -Path $ResultsDir, $StateDir | Out-Null

# The predictive Best-MC run reuses flexile_runtime_*.txt. Preserve the
# ground-truth runtime logs produced by the previous optimization.
foreach ($Priority in 1..3) {
    $RuntimePath = Join-Path $ResultsDir "flexile_runtime_$Priority.txt"
    $BackupPath = Join-Path $ResultsDir "gt_flexile_runtime_$Priority.txt"
    if ((Test-Path -LiteralPath $RuntimePath) -and -not (Test-Path -LiteralPath $BackupPath)) {
        Copy-Item -LiteralPath $RuntimePath -Destination $BackupPath
    }
}

$Stages = @(
    @{
        Name = "01_bestmc_priority1"
        Label = "Best-MC priority 1/3"
        Mode = "flexile"
        Priority = 1
        Objectives = @("mf")
        Result = "esm_optimal_values_mf.txt"
        Runtime = "flexile_runtime_1.txt"
        Simulation = $null
        NeedsSwanSplits = $false
    },
    @{
        Name = "02_bestmc_priority2"
        Label = "Best-MC priority 2/3"
        Mode = "flexile"
        Priority = 2
        Objectives = @("mf", "mf")
        Result = "esm_optimal_values_mf_mf.txt"
        Runtime = "flexile_runtime_2.txt"
        Simulation = $null
        NeedsSwanSplits = $false
    },
    @{
        Name = "03_bestmc_priority3"
        Label = "Best-MC priority 3/3"
        Mode = "flexile"
        Priority = 3
        Objectives = @("mf", "mf", "mf")
        Result = "esm_optimal_values_mf_mf_mf.txt"
        Runtime = "flexile_runtime_3.txt"
        Simulation = "flexile_sim_results_esm_mf_mf_mf.txt"
        NeedsSwanSplits = $false
    },
    @{
        Name = "04_swan_priority1"
        Label = "SWAN priority 1/3"
        Mode = "swan"
        Priority = 1
        Objectives = @("mf")
        Result = $null
        Runtime = "swan_runtime_1.txt"
        Simulation = $null
        NeedsSwanSplits = $true
    },
    @{
        Name = "05_swan_priority2"
        Label = "SWAN priority 2/3"
        Mode = "swan"
        Priority = 2
        Objectives = @("mf", "mf")
        Result = $null
        Runtime = "swan_runtime_2.txt"
        Simulation = $null
        NeedsSwanSplits = $true
    },
    @{
        Name = "06_swan_priority3"
        Label = "SWAN priority 3/3"
        Mode = "swan"
        Priority = 3
        Objectives = @("mf", "mf", "mf")
        Result = $null
        Runtime = "swan_runtime_3.txt"
        Simulation = "swan_sim_results_esm_mf_mf_mf.txt"
        NeedsSwanSplits = $true
    }
)

function Test-StageComplete {
    param([Parameter(Mandatory = $true)][hashtable]$Stage)

    $RuntimeCount = Get-LineCount -Path (Join-Path $ResultsDir $Stage.Runtime)
    if ($RuntimeCount -ne $ExpectedSnapshots) {
        return $false
    }

    if ($null -ne $Stage.Result) {
        $ResultCount = Get-LineCount -Path (Join-Path $ResultsDir $Stage.Result)
        if ($ResultCount -ne $ExpectedSnapshots) {
            return $false
        }
    }

    if ($Stage.NeedsSwanSplits) {
        $SplitCount = Get-PickleCount -Path $SwanSplitDir
        if ($SplitCount -ne $ExpectedSnapshots) {
            return $false
        }
    }

    if ($null -ne $Stage.Simulation) {
        $SimulationCount = Get-LineCount -Path (Join-Path $ResultsDir $Stage.Simulation)
        if ($SimulationCount -ne $ExpectedSimulationLines) {
            return $false
        }
    }

    return $true
}

Set-Location -LiteralPath $ProjectRoot
$env:PYTHONUNBUFFERED = "1"

$RunStarted = Get-Date
Write-Output "[$($RunStarted.ToString('s'))] GEANT baseline run started"
Write-Output "Project: $ProjectRoot"
Write-Output "Test interval: [$TestStart, $TestEnd) ($ExpectedSnapshots snapshots)"

foreach ($Stage in $Stages) {
    $DoneMarker = Join-Path $StateDir "$($Stage.Name).done"
    if (Test-StageComplete -Stage $Stage) {
        if (-not (Test-Path -LiteralPath $DoneMarker)) {
            "verified=$((Get-Date).ToString('o'))" | Set-Content -LiteralPath $DoneMarker -Encoding UTF8
        }
        Write-Output "[$((Get-Date).ToString('s'))] Skipping verified stage: $($Stage.Label)"
        continue
    }

    if (Test-Path -LiteralPath $DoneMarker) {
        Remove-Item -LiteralPath $DoneMarker -Force
    }

    $StageStarted = Get-Date
    Write-Output "[$($StageStarted.ToString('s'))] Starting $($Stage.Label)"

    $Arguments = @(
        $SolverPath,
        "--num_paths_per_pair", "8",
        "--opt_start_idx", "$TestStart",
        "--opt_end_idx", "$TestEnd",
        "--topo", "geant",
        "--framework", "gurobi",
        "--pred", "1",
        "--pred_type", "esm",
        "--cluster", "0",
        "--priority", "$($Stage.Priority)",
        "--objs"
    ) + $Stage.Objectives + @(
        "--gur_mode", $Stage.Mode,
        "--tol", "0.000001"
    )

    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$($Stage.Label) failed with exit code $LASTEXITCODE"
    }

    if (-not (Test-StageComplete -Stage $Stage)) {
        throw "$($Stage.Label) finished but its output validation failed"
    }

    $StageFinished = Get-Date
    "completed=$($StageFinished.ToString('o'))" | Set-Content -LiteralPath $DoneMarker -Encoding UTF8
    Write-Output "[$($StageFinished.ToString('s'))] Completed $($Stage.Label) in $([math]::Round(($StageFinished - $StageStarted).TotalMinutes, 2)) minutes"
}

$RunFinished = Get-Date
Write-Output "[$($RunFinished.ToString('s'))] All GEANT baselines completed in $([math]::Round(($RunFinished - $RunStarted).TotalHours, 2)) hours"
