# run_app.ps1
# OculoXplain - Unified Interface Launcher

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  OculoXplain - Unified Web Interface" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Yellow
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Python not found. Please install Python 3.8+" -ForegroundColor Red
    exit
}

# Check/Install dependencies
Write-Host ""
Write-Host "Checking dependencies..." -ForegroundColor Yellow

$deps = @("streamlit", "torch", "torchvision", "pytorch_grad_cam")
$missing = @()

foreach ($dep in $deps) {
    python -c "import $dep" 2>$null
    if ($LASTEXITCODE -ne 0) {
        $missing += $dep
    }
}

if ($missing.Count -gt 0) {
    Write-Host "Installing missing packages: $($missing -join ', ')" -ForegroundColor Yellow
    pip install -q @missing
}

Write-Host "✅ All dependencies ready!" -ForegroundColor Green
Write-Host ""
Write-Host "Launching OculoXplain..." -ForegroundColor Cyan
Write-Host "Opening http://localhost:8501" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

streamlit run app_unified.py --logger.level=error