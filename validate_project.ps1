# AutoDataPipeline Project Validation Script
Write-Host "AutoDataPipeline Project Validation" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host "Test Date: $(Get-Date)" -ForegroundColor Yellow
Write-Host "Project Directory: $(Get-Location)" -ForegroundColor Yellow
Write-Host ""

# Test Project Structure
Write-Host "=== Testing Project Structure ===" -ForegroundColor Cyan

$requiredFiles = @(
    'requirements.txt',
    'config.py',
    'main.py',
    'src/__init__.py',
    'src/logging_config.py',
    'src/data_ingestion.py',
    'src/data_transformation.py',
    'src/anomaly_detection.py',
    'src/data_storage.py',
    'src/visualization.py',
    'src/pipeline.py',
    'src/api.py'
)

$existingFiles = @()
$missingFiles = @()

foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "✓ $file" -ForegroundColor Green
        $existingFiles += $file
    } else {
        Write-Host "✗ $file - MISSING" -ForegroundColor Red
        $missingFiles += $file
    }
}

Write-Host ""
Write-Host "Summary: $($existingFiles.Count)/$($requiredFiles.Count) files exist" -ForegroundColor Yellow

if ($missingFiles.Count -gt 0) {
    Write-Host "Missing files: $($missingFiles -join ', ')" -ForegroundColor Red
}

# Test Directory Structure
Write-Host ""
Write-Host "=== Testing Directory Structure ===" -ForegroundColor Cyan

$requiredDirs = @('src')
$optionalDirs = @('data', 'plots', 'reports', 'logs', 'models')

$dirTestPassed = $true
foreach ($dir in $requiredDirs) {
    if (Test-Path $dir -PathType Container) {
        Write-Host "✓ $dir/ - Required directory exists" -ForegroundColor Green
    } else {
        Write-Host "✗ $dir/ - Required directory missing" -ForegroundColor Red
        $dirTestPassed = $false
    }
}

foreach ($dir in $optionalDirs) {
    if (Test-Path $dir -PathType Container) {
        Write-Host "✓ $dir/ - Optional directory exists" -ForegroundColor Green
    } else {
        Write-Host "○ $dir/ - Optional directory (will be created at runtime)" -ForegroundColor Yellow
    }
}

# Test Configuration File
Write-Host ""
Write-Host "=== Testing Configuration ===" -ForegroundColor Cyan

$configTestPassed = $false
if (Test-Path 'config.py') {
    $configContent = Get-Content 'config.py' -Raw
    $requiredConfigs = @(
        'PROJECT_CONFIG',
        'DATABASE_CONFIG',
        'DATA_GENERATION_CONFIG',
        'TRANSFORMATION_CONFIG',
        'ANOMALY_DETECTION_CONFIG',
        'VISUALIZATION_CONFIG',
        'LOGGING_CONFIG',
        'API_CONFIG',
        'EXPORT_CONFIG'
    )
    
    $missingConfigs = @()
    foreach ($config in $requiredConfigs) {
        if ($configContent -match $config) {
            Write-Host "✓ $config - Found" -ForegroundColor Green
        } else {
            Write-Host "✗ $config - Missing" -ForegroundColor Red
            $missingConfigs += $config
        }
    }
    
    if ($missingConfigs.Count -eq 0) {
        Write-Host "✓ All required configurations found" -ForegroundColor Green
        $configTestPassed = $true
    } else {
        Write-Host "Missing configurations: $($missingConfigs -join ', ')" -ForegroundColor Red
    }
} else {
    Write-Host "✗ config.py not found" -ForegroundColor Red
}

# Test Requirements File
Write-Host ""
Write-Host "=== Testing Requirements ===" -ForegroundColor Cyan

$reqTestPassed = $false
if (Test-Path 'requirements.txt') {
    $requirements = Get-Content 'requirements.txt' | Where-Object { $_.Trim() -ne '' -and -not $_.StartsWith('#') }
    $requiredPackages = @('pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn', 'fastapi', 'uvicorn', 'pydantic')
    
    $foundPackages = @()
    foreach ($req in $requirements) {
        $packageName = ($req -split '==|>=|<=' | Select-Object -First 1).Trim()
        $foundPackages += $packageName
    }
    
    $missingPackages = @()
    foreach ($package in $requiredPackages) {
        $found = $false
        foreach ($foundPkg in $foundPackages) {
            if ($foundPkg -like "*$package*") {
                $found = $true
                break
            }
        }
        
        if ($found) {
            Write-Host "✓ $package - Found" -ForegroundColor Green
        } else {
            Write-Host "✗ $package - Missing" -ForegroundColor Red
            $missingPackages += $package
        }
    }
    
    Write-Host ""
    Write-Host "Total packages in requirements.txt: $($foundPackages.Count)" -ForegroundColor Yellow
    
    if ($missingPackages.Count -eq 0) {
        $reqTestPassed = $true
    } else {
        Write-Host "Missing critical packages: $($missingPackages -join ', ')" -ForegroundColor Red
    }
} else {
    Write-Host "✗ requirements.txt not found" -ForegroundColor Red
}

# Generate Summary Report
Write-Host ""
Write-Host "============================================================" -ForegroundColor White
Write-Host "TEST SUMMARY" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor White

$tests = @(
    @{Name="Project Structure"; Passed=($missingFiles.Count -eq 0)},
    @{Name="Directory Structure"; Passed=$dirTestPassed},
    @{Name="Configuration"; Passed=$configTestPassed},
    @{Name="Requirements"; Passed=$reqTestPassed}
)

$passedTests = 0
foreach ($test in $tests) {
    if ($test.Passed) {
        $status = "PASS"
        $passedTests++
        $color = "Green"
    } else {
        $status = "FAIL"
        $color = "Red"
    }
    Write-Host ("{0,-20} : {1}" -f $test.Name, $status) -ForegroundColor $color
}

Write-Host ""
Write-Host "Overall Result: $passedTests/$($tests.Count) tests passed" -ForegroundColor Yellow

if ($passedTests -eq $tests.Count) {
    Write-Host ""
    Write-Host "🎉 All tests passed! The AutoDataPipeline project structure is complete." -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Install Python 3.8+ if not already installed" -ForegroundColor White
    Write-Host "2. Run: pip install -r requirements.txt" -ForegroundColor White
    Write-Host "3. Test the pipeline: python main.py --mode generate" -ForegroundColor White
    Write-Host "4. Run full pipeline: python main.py --mode full" -ForegroundColor White
    Write-Host "5. Start API server: python main.py --mode api" -ForegroundColor White
    exit 0
} else {
    Write-Host ""
    Write-Host "❌ Some tests failed. Please review the issues above." -ForegroundColor Red
    exit 1
}