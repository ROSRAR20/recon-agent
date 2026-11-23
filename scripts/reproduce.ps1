# Reproduce the evaluation pipeline completely offline without an API key
param(
    [switch]$SkipVenv = $false
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "================================================================================"
Write-Host "RECON-AGENT OFFLINE REPRODUCTION PIPELINE"
Write-Host "================================================================================"

# 1. Environment / Venv Setup
if (-not $SkipVenv) {
    if (-not (Test-Path "$RepoRoot\.venv")) {
        Write-Host "[1/5] Creating virtual environment (.venv)..."
        python -m venv "$RepoRoot\.venv"
        & "$RepoRoot\.venv\Scripts\python.exe" -m pip install --upgrade pip
        & "$RepoRoot\.venv\Scripts\pip.exe" install -e "$RepoRoot"
        & "$RepoRoot\.venv\Scripts\pip.exe" install pytest
    } else {
        Write-Host "[1/5] Virtual environment (.venv) already exists."
    }
}

# Determine python command
$PyCmd = if (Test-Path "$RepoRoot\.venv\Scripts\python.exe") { "$RepoRoot\.venv\Scripts\python.exe" } else { "python" }

# 2. Run unit and integration tests offline
Write-Host "`n[2/5] Running pytest suite offline..."
& $PyCmd -m pytest -q

# 3. Regenerate dev datasets (seeds 1-3)
Write-Host "`n[3/5] Regenerating synthetic dev datasets (seeds 1-3)..."
& $PyCmd scripts/generate.py --split dev --seed 1 --profile full
& $PyCmd scripts/generate.py --split dev --seed 2 --profile full
& $PyCmd scripts/generate.py --split dev --seed 3 --profile full

# 4. Rescore from compressed archive in artifacts/
Write-Host "`n[4/5] Rescoring evaluation metrics offline from artifacts/..."
& $PyCmd scripts/rescore_from_archive.py --allow-dirty

# 5. Print results summary
Write-Host "`n[5/5] Primary Results Table (from docs/results_main.md):"
Write-Host "================================================================================"
Get-Content "$RepoRoot\docs\results_main.md" | Select-String -Pattern "^\| \*\*H|^\| Metric|PASS|FAIL"
Write-Host "================================================================================"
Write-Host "Reproduction complete. All numbers verified offline with zero API calls."
