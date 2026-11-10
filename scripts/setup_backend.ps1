# setup_backend.ps1 — Clone, build, and pin llama.cpp with CUDA support (Windows)
#
# Usage:
#   .\scripts\setup_backend.ps1 [-Commit "master"]
#
# Prerequisites:
#   - Git
#   - CMake (3.21+)
#   - Visual Studio 2019/2022 with C++ workload
#   - CUDA Toolkit (matching your driver)
#
# Alternative (faster): Download a pre-built release
#   https://github.com/ggml-org/llama.cpp/releases
#   Extract and add the bin/ folder to PATH.

[CmdletBinding()]
param(
    [string]$Commit = "master"
)

$ErrorActionPreference = "Stop"

$RepoUrl = "https://github.com/ggml-org/llama.cpp.git"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $ProjectRoot) { $ProjectRoot = Split-Path -Parent $PSScriptRoot }
$BuildDir = Join-Path $ProjectRoot "vendor\llama.cpp"

Write-Host "=== ADHIE Backend Setup (Windows) ===" -ForegroundColor Cyan
Write-Host "Repository:  $RepoUrl"
Write-Host "Commit:      $Commit"
Write-Host "Build dir:   $BuildDir"
Write-Host ""

# ── Check prerequisites ──────────────────────────────────────────
function Test-Command($cmd) {
    return [bool](Get-Command $cmd -ErrorAction SilentlyContinue)
}

$missing = @()
if (-not (Test-Command "git"))   { $missing += "git" }
if (-not (Test-Command "cmake")) { $missing += "cmake" }
if (-not (Test-Command "nvcc"))  { $missing += "nvcc (CUDA Toolkit)" }

if ($missing.Count -gt 0) {
    Write-Host "ERROR: Missing prerequisites: $($missing -join ', ')" -ForegroundColor Red
    Write-Host ""
    Write-Host "Alternatively, download a pre-built release:" -ForegroundColor Yellow
    Write-Host "  https://github.com/ggml-org/llama.cpp/releases" -ForegroundColor Yellow
    Write-Host "  Extract llama-server.exe and add its folder to PATH," -ForegroundColor Yellow
    Write-Host "  or pass --backend-path to adhie benchmark." -ForegroundColor Yellow
    exit 1
}

# ── Clone if not present ─────────────────────────────────────────
if (-not (Test-Path $BuildDir)) {
    Write-Host ">>> Cloning llama.cpp..." -ForegroundColor Green
    git clone $RepoUrl $BuildDir
} else {
    Write-Host ">>> llama.cpp already cloned at $BuildDir" -ForegroundColor Green
}

Push-Location $BuildDir
try {
    # Checkout pinned commit
    Write-Host ">>> Checking out $Commit..." -ForegroundColor Green
    git fetch origin
    git checkout $Commit

    $ActualHash = git rev-parse HEAD
    Write-Host ">>> Pinned at commit: $ActualHash" -ForegroundColor Green

    # Build with CUDA support
    Write-Host ">>> Building with CUDA support..." -ForegroundColor Green
    if (-not (Test-Path "build")) { New-Item -ItemType Directory -Path "build" | Out-Null }
    Push-Location "build"
    try {
        cmake .. `
            -DGGML_CUDA=ON `
            -DCMAKE_BUILD_TYPE=Release `
            -DLLAMA_CURL=OFF

        cmake --build . --config Release -j $env:NUMBER_OF_PROCESSORS

        Write-Host ""
        Write-Host "=== Build complete ===" -ForegroundColor Green

        # Find the binary (could be in bin/ or bin/Release/)
        $serverExe = $null
        $searchPaths = @(
            "bin\Release\llama-server.exe",
            "bin\llama-server.exe",
            "Release\llama-server.exe",
            "llama-server.exe"
        )
        foreach ($sp in $searchPaths) {
            $candidate = Join-Path (Get-Location) $sp
            if (Test-Path $candidate) {
                $serverExe = $candidate
                break
            }
        }

        if ($serverExe) {
            Write-Host "Server binary: $serverExe" -ForegroundColor Cyan
            Write-Host ""
            Write-Host ">>> Verifying server..." -ForegroundColor Green
            & $serverExe --help 2>&1 | Select-Object -First 5
            Write-Host "..."
            Write-Host "✓ llama-server built successfully" -ForegroundColor Green
            Write-Host ""
            Write-Host "Add to PATH or use:" -ForegroundColor Yellow
            Write-Host "  adhie benchmark MODEL --backend-path `"$serverExe`"" -ForegroundColor Yellow
        } else {
            Write-Host "✗ llama-server.exe not found — build may have failed" -ForegroundColor Red
            Write-Host "Searched in: $((Get-Location).Path)" -ForegroundColor Red
            exit 1
        }

        # Save build info
        @"
commit: $ActualHash
date: $(Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ" -AsUTC)
cuda: $(nvcc --version 2>&1 | Select-String "release")
cmake: $(cmake --version | Select-Object -First 1)
os: Windows $([System.Environment]::OSVersion.Version)
"@ | Set-Content "BUILD_INFO.txt"

        Write-Host "Build info saved to $(Join-Path (Get-Location) 'BUILD_INFO.txt')"

    } finally {
        Pop-Location
    }

} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green

