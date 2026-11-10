# ──────────────────────────────────────────────────────────────────
# ADHIE End-to-End Demo Script (Windows PowerShell)
#
# Usage:
#   .\scripts\demo.ps1 -Model "MODEL_REF" [-Draft "DRAFT_MODEL_REF"]
#
# Examples:
#   .\scripts\demo.ps1 -Model "Qwen/Qwen2.5-7B-Instruct-GGUF:q4_k_m"
#   .\scripts\demo.ps1 -Model "Qwen/Qwen2.5-14B-Instruct-GGUF:q4_k_m" -Draft "Qwen/Qwen2.5-1.5B-Instruct-GGUF:q4_k_m"
# ──────────────────────────────────────────────────────────────────

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Model,

    [string]$Draft = "",

    [string]$BackendPath = "",

    [int]$Context = 8192,
    [int]$Runs = 3,
    [int]$Tokens = 128
)

$ErrorActionPreference = "Stop"
$ResultsDir = "benchmarks\results\demo"

# Auto-detect project virtual environment if adhie is not in PATH
if (-not (Get-Command adhie -ErrorAction SilentlyContinue)) {
    $venvScripts = Join-Path $PSScriptRoot "..\.venv\Scripts"
    if (Test-Path (Join-Path $venvScripts "adhie.exe")) {
        $env:PATH = "$venvScripts;$env:PATH"
    }
}

function Write-Divider([string]$Title) {
    Write-Host ""
    Write-Host ("=" * 65) -ForegroundColor Cyan
    Write-Host "  $Title" -ForegroundColor Cyan
    Write-Host ("=" * 65) -ForegroundColor Cyan
    Write-Host ""
}

function Invoke-Adhie {
    <#
    .SYNOPSIS
        Runs adhie and aborts the script if it exits with a non-zero code.
    #>
    & adhie @args
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "ERROR: adhie exited with code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# ─── Step 1: Hardware Profile ─────────────────────────────────────
Write-Divider "Step 1/7: Hardware Profile"
Invoke-Adhie profile -o "$ResultsDir\hardware_profile.yaml"

# ─── Step 2: Model Inspection ────────────────────────────────────
Write-Divider "Step 2/7: Model Inspection"
Invoke-Adhie inspect $Model -o "$ResultsDir\model_profile.yaml"

# ─── Step 3: Execution Plan ──────────────────────────────────────
Write-Divider "Step 3/7: Execution Plan Generation"
$planArgs = @("plan", $Model, "--context", $Context, "-o", "$ResultsDir\plans.yaml")
if ($Draft) { $planArgs += @("--draft", $Draft) }
Invoke-Adhie @planArgs

# ─── Step 4: Memory Estimation ───────────────────────────────────
Write-Divider "Step 4/7: Memory Estimation"
Invoke-Adhie estimate $Model --context $Context

# ─── Step 5: Baseline Benchmark (CPU-only) ───────────────────────
Write-Divider "Step 5/7: Baseline Benchmark (CPU-only)"
$baseArgs = @("benchmark", $Model, "--context", $Context, "--mode", "baseline",
              "--runs", $Runs, "--tokens", $Tokens, "-o", $ResultsDir)
if ($BackendPath) { $baseArgs += @("--backend-path", $BackendPath) }
Invoke-Adhie @baseArgs

# ─── Step 6: ADHIE-optimised Benchmark ──────────────────────────
Write-Divider "Step 6/7: ADHIE-optimised Benchmark"
$benchArgs = @("benchmark", $Model, "--context", $Context, "--mode", "auto",
               "--runs", $Runs, "--tokens", $Tokens, "-o", $ResultsDir)
if ($Draft) { $benchArgs += @("--draft", $Draft) }
if ($BackendPath) { $benchArgs += @("--backend-path", $BackendPath) }
Invoke-Adhie @benchArgs

# ─── Step 7: Comparison ─────────────────────────────────────────
Write-Divider "Step 7/7: Results Comparison"
$resultFiles = Get-ChildItem -Path $ResultsDir -Filter "*.json" |
    Where-Object { $_.Name -notlike "spec_*" -and $_.Name -notlike "ablation_*" } |
    Sort-Object Name |
    Select-Object -ExpandProperty FullName

if ($resultFiles.Count -ge 1) {
    $compareArgs = @("compare") + $resultFiles + @("-o", "$ResultsDir\comparison_report.md")
    Invoke-Adhie @compareArgs
}

# ─── Optional: Ablation Study ───────────────────────────────────
if ($Draft) {
    Write-Divider "Bonus: Ablation Study"
    $ablArgs = @("ablation", $Model, "--draft", $Draft, "--context", $Context,
                 "--runs", $Runs, "--tokens", $Tokens, "-o", "$ResultsDir\ablation")
    if ($BackendPath) { $ablArgs += @("--backend-path", $BackendPath) }
    Invoke-Adhie @ablArgs
}

# ─── Summary ────────────────────────────────────────────────────
Write-Divider "Demo Complete"
Write-Host "Results saved to: $ResultsDir\"
Write-Host ""
Write-Host "Files:"
Get-ChildItem -Path $ResultsDir -Recurse -File | ForEach-Object {
    $size = "{0:N1} KB" -f ($_.Length / 1KB)
    Write-Host ("  {0,-10} {1}" -f $size, $_.FullName)
}
Write-Host ""
Write-Host "Generate plots with:"
Write-Host "  python scripts\generate_plots.py --input $ResultsDir"

