# Wiki Deployment Script for PYAI
# This script will push wiki content once you create the first wiki page manually

Write-Host "📚 PYAI Wiki Deployment Script" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# Check if wiki repo exists
$wikiUrl = "https://github.com/gitpavleenbali/PYAI.wiki.git"
$wikiPath = Join-Path $PSScriptRoot ".." "PYAI.wiki"
$wikiSourcePath = Join-Path $PSScriptRoot ".." "docs" "wiki"

Write-Host "`nStep 1: Checking if wiki repository exists..." -ForegroundColor Yellow

# Try to clone
git clone $wikiUrl $wikiPath 2>&1 | Out-Null

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Wiki repository found!" -ForegroundColor Green
    
    # Copy all wiki files
    Write-Host "`nStep 2: Copying wiki pages..." -ForegroundColor Yellow
    Copy-Item -Path "$wikiSourcePath\*.md" -Destination $wikiPath -Force
    
    # Commit and push
    Set-Location $wikiPath
    git add -A
    git commit -m "Deploy PYAI wiki documentation"
    git push
    
    Write-Host "`n✅ Wiki deployed successfully!" -ForegroundColor Green
    Write-Host "🔗 View at: https://github.com/gitpavleenbali/PYAI/wiki" -ForegroundColor Cyan
} else {
    Write-Host "⚠️ Wiki repository not initialized yet." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "GitHub Limitation: Wiki repo is only created after first page via UI." -ForegroundColor Red
    Write-Host ""
    Write-Host "Quick Steps:" -ForegroundColor Cyan
    Write-Host "1. Open: https://github.com/gitpavleenbali/PYAI/wiki" -ForegroundColor White
    Write-Host "2. Click 'Create the first page'" -ForegroundColor White
    Write-Host "3. Type 'Home' as title, any text as content" -ForegroundColor White
    Write-Host "4. Click 'Save Page'" -ForegroundColor White
    Write-Host "5. Run this script again" -ForegroundColor White
    Write-Host ""
    Write-Host "Alternative: Wiki content is already in docs/wiki/ folder in the main repo!" -ForegroundColor Green
}
