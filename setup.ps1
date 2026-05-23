# Setup script for Verifex on Windows
# This script installs Python 3.11 if needed, creates a .venv, and installs backend dependencies.
$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $repo

function Get-PythonExe {
    param([string]$Version)
    try {
        $path = py -$Version -c "import sys; print(sys.executable)" 2>$null
        return if ($path) { $path.Trim() } else { $null }
    } catch {
        return $null
    }
}

$pythonExe = Get-PythonExe '3.11'
if (-not $pythonExe) {
    Write-Host 'Python 3.11 not found. Installing with winget...'
    winget install --id Python.Python.3.11 -e --source winget
    $pythonExe = Get-PythonExe '3.11'
}

if (-not $pythonExe) {
    Write-Error 'Python 3.11 installation failed or is not available. Please install Python 3.11 manually.'
    exit 1
}

Write-Host "Using Python: $pythonExe"
& "$pythonExe" -m venv .venv

$venvPython = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    Write-Error 'Failed to create virtual environment.'
    exit 1
}

& "$venvPython" -m pip install --upgrade pip setuptools wheel
& "$venvPython" -m pip install -r requirements.txt

Write-Host 'Setup complete. Use .\.venv\Scripts\Activate.ps1 to activate the environment.'
