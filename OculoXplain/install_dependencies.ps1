# install_dependencies.ps1
# Install all required dependencies

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  OculoXplain - Dependency Installation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip -q

# Install from requirements
Write-Host "Installing dependencies from requirements_app.txt..." -ForegroundColor Yellow
pip install -r requirements_app.txt

Write-Host ""
Write-Host "✅ Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To run the application, execute: .\run_app.ps1" -ForegroundColor Cyan