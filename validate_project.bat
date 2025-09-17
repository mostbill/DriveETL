@echo off
echo AutoDataPipeline Project Validation
echo =====================================
echo Test Date: %date% %time%
echo Project Directory: %cd%
echo.

echo === Testing Project Structure ===
set /a total_files=0
set /a existing_files=0

set files=requirements.txt config.py main.py src/__init__.py src/logging_config.py src/data_ingestion.py src/data_transformation.py src/anomaly_detection.py src/data_storage.py src/visualization.py src/pipeline.py src/api.py

for %%f in (%files%) do (
    set /a total_files+=1
    if exist "%%f" (
        echo [32m✓ %%f[0m
        set /a existing_files+=1
    ) else (
        echo [31m✗ %%f - MISSING[0m
    )
)

echo.
echo Summary: %existing_files%/%total_files% files exist

echo.
echo === Testing Directory Structure ===
if exist "src" (
    echo [32m✓ src/ - Required directory exists[0m
    set dir_test=PASS
) else (
    echo [31m✗ src/ - Required directory missing[0m
    set dir_test=FAIL
)

for %%d in (data plots reports logs models) do (
    if exist "%%d" (
        echo [32m✓ %%d/ - Optional directory exists[0m
    ) else (
        echo [33m○ %%d/ - Optional directory (will be created at runtime)[0m
    )
)

echo.
echo === Testing Configuration ===
if exist "config.py" (
    findstr /C:"PROJECT_CONFIG" config.py >nul && (
        echo [32m✓ config.py - Found with required configurations[0m
        set config_test=PASS
    ) || (
        echo [31m✗ config.py - Missing required configurations[0m
        set config_test=FAIL
    )
) else (
    echo [31m✗ config.py not found[0m
    set config_test=FAIL
)

echo.
echo === Testing Requirements ===
if exist "requirements.txt" (
    findstr /C:"pandas" requirements.txt >nul && (
        echo [32m✓ requirements.txt - Found with required packages[0m
        set req_test=PASS
    ) || (
        echo [31m✗ requirements.txt - Missing required packages[0m
        set req_test=FAIL
    )
) else (
    echo [31m✗ requirements.txt not found[0m
    set req_test=FAIL
)

echo.
echo ============================================================
echo TEST SUMMARY
echo ============================================================

set /a passed_tests=0
set /a total_tests=4

if %existing_files%==%total_files% (
    echo [32mProject Structure    : PASS[0m
    set /a passed_tests+=1
) else (
    echo [31mProject Structure    : FAIL[0m
)

if "%dir_test%"=="PASS" (
    echo [32mDirectory Structure  : PASS[0m
    set /a passed_tests+=1
) else (
    echo [31mDirectory Structure  : FAIL[0m
)

if "%config_test%"=="PASS" (
    echo [32mConfiguration        : PASS[0m
    set /a passed_tests+=1
) else (
    echo [31mConfiguration        : FAIL[0m
)

if "%req_test%"=="PASS" (
    echo [32mRequirements         : PASS[0m
    set /a passed_tests+=1
) else (
    echo [31mRequirements         : FAIL[0m
)

echo.
echo Overall Result: %passed_tests%/%total_tests% tests passed

if %passed_tests%==%total_tests% (
    echo.
    echo [32m🎉 All tests passed! The AutoDataPipeline project structure is complete.[0m
    echo.
    echo Next steps:
    echo 1. Install Python 3.8+ if not already installed
    echo 2. Run: pip install -r requirements.txt
    echo 3. Test the pipeline: python main.py --mode generate
    echo 4. Run full pipeline: python main.py --mode full
    echo 5. Start API server: python main.py --mode api
    exit /b 0
) else (
    echo.
    echo [31m❌ Some tests failed. Please review the issues above.[0m
    exit /b 1
)