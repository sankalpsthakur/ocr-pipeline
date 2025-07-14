#!/usr/bin/env python3
"""Test deployment configuration for Railway

Quick test to verify the web service starts correctly.
"""

import sys
import subprocess
import time
import requests
from pathlib import Path

def test_requirements():
    """Test that all required dependencies can be imported."""
    print("Testing requirements...")
    
    try:
        import fastapi
        print("✅ FastAPI available")
    except ImportError:
        print("❌ FastAPI not available")
        return False
    
    try:
        import uvicorn
        print("✅ Uvicorn available")
    except ImportError:
        print("❌ Uvicorn not available")
        return False
    
    try:
        # Test OCR pipeline import
        from pipeline import run_ocr
        print("✅ Main OCR pipeline available")
    except ImportError as e:
        print(f"⚠️  Main OCR pipeline not available: {e}")
        
        try:
            from pytorch_mobile.ocr_pipeline import run_ocr_with_tesseract
            print("✅ Mobile OCR pipeline available")
        except ImportError as e:
            print(f"❌ No OCR pipeline available: {e}")
            return False
    
    return True

def test_main_import():
    """Test that main.py can be imported."""
    print("\nTesting main.py import...")
    
    try:
        import main
        print("✅ main.py imports successfully")
        return True
    except ImportError as e:
        print(f"❌ main.py import failed: {e}")
        return False

def test_service_start():
    """Test that the service can start (quick test)."""
    print("\nTesting service startup...")
    
    try:
        # Start service in background
        proc = subprocess.Popen([
            sys.executable, "main.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Give it a moment to start
        time.sleep(3)
        
        # Check if it's running
        if proc.poll() is None:
            print("✅ Service started successfully")
            
            # Try to hit health endpoint
            try:
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    print("✅ Health endpoint responding")
                else:
                    print(f"⚠️  Health endpoint returned {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"⚠️  Could not reach health endpoint: {e}")
            
            # Stop the service
            proc.terminate()
            proc.wait()
            return True
        else:
            stdout, stderr = proc.communicate()
            print(f"❌ Service failed to start")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"❌ Service test failed: {e}")
        return False

def test_file_structure():
    """Test that required files exist."""
    print("\nTesting file structure...")
    
    required_files = [
        "main.py",
        "requirements.txt", 
        "railway.toml",
        ".railwayignore",
        "pipeline.py",
        "config.py"
    ]
    
    all_exist = True
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all deployment tests."""
    print("🚀 Railway Deployment Test")
    print("=" * 40)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Requirements", test_requirements),
        ("Main Import", test_main_import),
        ("Service Startup", test_service_start)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        result = test_func()
        results.append((name, result))
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Summary:")
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {name}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Ready for Railway deployment.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check issues before deploying.")
        return 1

if __name__ == "__main__":
    sys.exit(main())