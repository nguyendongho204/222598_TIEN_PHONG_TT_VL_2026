# Windows PowerShell startup script for EBM-SVM API

# Set error action preference
$ErrorActionPreference = "Continue"

# Get the directory where this script is located
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "=" * 60
Write-Host "EBM-SVM API Server Startup"
Write-Host "=" * 60

# Create .env if it doesn't exist
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env file from template..."
    Copy-Item ".env.example" ".env" -Force
}

# Create necessary directories
$directories = @("logs", "utils/upload_temp", "utils/download", "models")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created directory: $dir"
    }
}

# Install dependencies
Write-Host "Checking dependencies..."
$pythonCmd = if (Get-Command python3 -ErrorAction SilentlyContinue) { "python3" } else { "python" }

if ($pythonCmd) {
    Write-Host "Installing required packages..."
    & $pythonCmd -m pip install -q -r requirements.txt
} else {
    Write-Host "WARNING: Python not found. Please install Python 3.10+"
    exit 1
}

# Start the server
Write-Host ""
Write-Host "Starting EBM-SVM API server..."
Write-Host "Access the API at: http://localhost:8000"
Write-Host "Swagger Docs at: http://localhost:8000/docs"
Write-Host "ReDoc at: http://localhost:8000/redoc"
Write-Host ""

& $pythonCmd run_api.py @args

Pause
