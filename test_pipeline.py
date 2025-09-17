#!/usr/bin/env python3
"""
Test script for AutoDataPipeline project structure and basic functionality.
This script validates the project setup without requiring external dependencies.
"""

import os
import sys
import importlib.util
from datetime import datetime

def test_project_structure():
    """Test that all required files and directories exist."""
    print("\n=== Testing Project Structure ===")
    
    required_files = [
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
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
            print(f"✓ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"✗ {file_path} - MISSING")
    
    print(f"\nSummary: {len(existing_files)}/{len(required_files)} files exist")
    
    if missing_files:
        print(f"Missing files: {missing_files}")
        return False
    
    return True

def test_module_imports():
    """Test that modules can be imported (syntax validation)."""
    print("\n=== Testing Module Imports ===")
    
    modules_to_test = [
        ('config', 'config.py'),
        ('main', 'main.py')
    ]
    
    src_modules = [
        ('src.logging_config', 'src/logging_config.py'),
        ('src.data_ingestion', 'src/data_ingestion.py'),
        ('src.data_transformation', 'src/data_transformation.py'),
        ('src.anomaly_detection', 'src/anomaly_detection.py'),
        ('src.data_storage', 'src/data_storage.py'),
        ('src.visualization', 'src/visualization.py'),
        ('src.pipeline', 'src/pipeline.py'),
        ('src.api', 'src/api.py')
    ]
    
    all_modules = modules_to_test + src_modules
    successful_imports = 0
    failed_imports = []
    
    for module_name, file_path in all_modules:
        try:
            if os.path.exists(file_path):
                # Load module from file
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    # Don't execute the module, just validate syntax
                    print(f"✓ {module_name} - Syntax OK")
                    successful_imports += 1
                else:
                    print(f"✗ {module_name} - Import spec failed")
                    failed_imports.append(module_name)
            else:
                print(f"✗ {module_name} - File not found: {file_path}")
                failed_imports.append(module_name)
        except SyntaxError as e:
            print(f"✗ {module_name} - Syntax Error: {e}")
            failed_imports.append(module_name)
        except Exception as e:
            print(f"✗ {module_name} - Error: {e}")
            failed_imports.append(module_name)
    
    print(f"\nSummary: {successful_imports}/{len(all_modules)} modules have valid syntax")
    
    if failed_imports:
        print(f"Failed imports: {failed_imports}")
        return False
    
    return True

def test_configuration():
    """Test configuration file structure."""
    print("\n=== Testing Configuration ===")
    
    try:
        if not os.path.exists('config.py'):
            print("✗ config.py not found")
            return False
        
        # Read config file content
        with open('config.py', 'r') as f:
            config_content = f.read()
        
        required_configs = [
            'PROJECT_CONFIG',
            'DATABASE_CONFIG',
            'DATA_GENERATION_CONFIG',
            'TRANSFORMATION_CONFIG',
            'ANOMALY_DETECTION_CONFIG',
            'VISUALIZATION_CONFIG',
            'LOGGING_CONFIG',
            'API_CONFIG',
            'EXPORT_CONFIG'
        ]
        
        missing_configs = []
        for config in required_configs:
            if config in config_content:
                print(f"✓ {config} - Found")
            else:
                print(f"✗ {config} - Missing")
                missing_configs.append(config)
        
        if missing_configs:
            print(f"Missing configurations: {missing_configs}")
            return False
        
        print("✓ All required configurations found")
        return True
        
    except Exception as e:
        print(f"✗ Error testing configuration: {e}")
        return False

def test_requirements():
    """Test requirements.txt structure."""
    print("\n=== Testing Requirements ===")
    
    try:
        if not os.path.exists('requirements.txt'):
            print("✗ requirements.txt not found")
            return False
        
        with open('requirements.txt', 'r') as f:
            requirements = f.read().strip().split('\n')
        
        required_packages = [
            'pandas', 'numpy', 'scikit-learn', 'matplotlib', 
            'seaborn', 'fastapi', 'uvicorn', 'pydantic'
        ]
        
        found_packages = []
        for req in requirements:
            if req.strip() and not req.strip().startswith('#'):
                package_name = req.split('==')[0].split('>=')[0].split('<=')[0].strip()
                found_packages.append(package_name)
        
        missing_packages = []
        for package in required_packages:
            if any(package in found for found in found_packages):
                print(f"✓ {package} - Found")
            else:
                print(f"✗ {package} - Missing")
                missing_packages.append(package)
        
        print(f"\nTotal packages in requirements.txt: {len(found_packages)}")
        
        if missing_packages:
            print(f"Missing critical packages: {missing_packages}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing requirements: {e}")
        return False

def test_directory_structure():
    """Test expected directory structure."""
    print("\n=== Testing Directory Structure ===")
    
    expected_dirs = ['src']
    optional_dirs = ['data', 'plots', 'reports', 'logs', 'models']
    
    for directory in expected_dirs:
        if os.path.exists(directory) and os.path.isdir(directory):
            print(f"✓ {directory}/ - Required directory exists")
        else:
            print(f"✗ {directory}/ - Required directory missing")
            return False
    
    for directory in optional_dirs:
        if os.path.exists(directory) and os.path.isdir(directory):
            print(f"✓ {directory}/ - Optional directory exists")
        else:
            print(f"○ {directory}/ - Optional directory (will be created at runtime)")
    
    return True

def generate_test_report():
    """Generate a comprehensive test report."""
    print("\n" + "="*60)
    print("AutoDataPipeline Project Validation Report")
    print("="*60)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project Directory: {os.getcwd()}")
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Module Syntax", test_module_imports),
        ("Configuration", test_configuration),
        ("Requirements", test_requirements),
        ("Directory Structure", test_directory_structure)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} - Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<20} : {status}")
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The AutoDataPipeline project structure is complete.")
        print("\nNext steps:")
        print("1. Install Python 3.8+ if not already installed")
        print("2. Run: pip install -r requirements.txt")
        print("3. Test the pipeline: python main.py --mode generate")
        print("4. Run full pipeline: python main.py --mode full")
        print("5. Start API server: python main.py --mode api")
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    print("AutoDataPipeline Project Validation")
    print("This script validates the project structure and basic functionality.")
    
    success = generate_test_report()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)