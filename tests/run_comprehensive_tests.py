#!/usr/bin/env python3
"""
Script to run comprehensive tests and generate a consolidated report.

Usage:
    python run_comprehensive_tests.py                    # Run tests with detailed output
    python run_comprehensive_tests.py --consolidated     # Run tests and show consolidated report
    python run_comprehensive_tests.py --save-report      # Run tests and save report to file
"""

import subprocess
import sys
import os
from datetime import datetime


def run_tests_with_consolidated_report():
    """Run the comprehensive test suite with consolidated report."""
    
    # Ensure we're in the correct directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run the tests with consolidated report
    cmd = [
        sys.executable,
        "tests/test_consolidated_comprehensive.py",
        "--consolidated-report"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    return result.stdout, result.stderr, result.returncode


def run_detailed_tests():
    """Run the comprehensive test suite with detailed output."""
    
    # Ensure we're in the correct directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run pytest with verbose output
    cmd = [
        sys.executable,
        "-m", "pytest",
        "tests/test_consolidated_comprehensive.py",
        "-v", "-s"
    ]
    
    result = subprocess.run(cmd)
    
    return result.returncode


def save_report_to_file(report_content):
    """Save the consolidated report to a timestamped file."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"consolidated_test_report_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write(report_content)
    
    print(f"\nReport saved to: {filename}")
    return filename


def main():
    """Main entry point."""
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--consolidated":
            # Run with consolidated report output
            stdout, stderr, returncode = run_tests_with_consolidated_report()
            print(stdout)
            if stderr:
                print("\n[STDERR OUTPUT]")
                print(stderr)
            sys.exit(returncode)
            
        elif sys.argv[1] == "--save-report":
            # Run and save consolidated report
            stdout, stderr, returncode = run_tests_with_consolidated_report()
            
            # Save the report
            if stdout:
                filename = save_report_to_file(stdout)
                print(f"\nTests completed with return code: {returncode}")
                print(f"Full report saved to: {filename}")
                
                # Also print a summary
                print("\nSUMMARY:")
                # Extract key metrics from the report
                lines = stdout.split('\n')
                for line in lines:
                    if "Field-Level Accuracy:" in line:
                        print(f"  {line.strip()}")
                    elif "Status:" in line and "✓" in line:
                        print(f"  {line.strip()}")
                    elif "passed" in line and "warning" in line:
                        print(f"  {line.strip()}")
            
            sys.exit(returncode)
    
    # Default: run detailed tests
    print("Running comprehensive test suite with detailed output...")
    print("-" * 60)
    returncode = run_detailed_tests()
    sys.exit(returncode)


if __name__ == "__main__":
    main()