#!/usr/bin/env python3
"""
Comprehensive Test Runner for OCR Pipeline

Tests include:
- Downscaling tests (100%, 50%, 25% scale)
- Character, word, and field level accuracy
- Focus on electricity (299 kWh) and carbon footprint (120 kg CO2e)
- Confidence score correlation

Usage:
    python run_comprehensive_tests.py              # Run all tests
    python run_comprehensive_tests.py --quick      # Run only on original image
    python run_comprehensive_tests.py --json       # Output results as JSON
"""

import sys
import os
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_comprehensive import ComprehensiveOCRTester


def run_comprehensive_tests(quick_mode=False):
    """Run the comprehensive test suite."""
    tester = ComprehensiveOCRTester()
    
    if quick_mode:
        # Test only original image
        print("Running quick test (original image only)...")
        if Path("ActualBill.png").exists():
            tester.test_image("ActualBill.png", scale=1.0)
            tester.generate_summary_report()
        else:
            print("Error: ActualBill.png not found!")
            return 1
    else:
        # Run full test suite
        tester.run_comprehensive_tests()
    
    # Check if tests passed
    if tester.results:
        avg_accuracy = sum(r['field_accuracy'] for r in tester.results) / len(tester.results)
        return 0 if avg_accuracy >= 90 else 1
    
    return 1


def output_json_results():
    """Run tests and output results as JSON."""
    import io
    from contextlib import redirect_stdout
    
    # Capture stdout
    f = io.StringIO()
    with redirect_stdout(f):
        tester = ComprehensiveOCRTester()
        tester.run_comprehensive_tests()
    
    # Return JSON results
    if tester.results:
        summary = {
            "total_tests": len(tester.results),
            "average_field_accuracy": sum(r['field_accuracy'] for r in tester.results) / len(tester.results),
            "electricity_accuracy": sum(1 for r in tester.results if r['electricity_kwh']['correct']) / len(tester.results) * 100,
            "carbon_accuracy": sum(1 for r in tester.results if r['carbon_kgco2e']['correct']) / len(tester.results) * 100,
            "meets_target": sum(r['field_accuracy'] for r in tester.results) / len(tester.results) >= 90
        }
        
        print(json.dumps({
            "summary": summary,
            "results": tester.results
        }, indent=2))
        
        return 0 if summary["meets_target"] else 1
    
    return 1


def main():
    """Main entry point."""
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            # Run quick test
            returncode = run_comprehensive_tests(quick_mode=True)
            sys.exit(returncode)
            
        elif sys.argv[1] == "--json":
            # Output JSON results
            returncode = output_json_results()
            sys.exit(returncode)
            
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("Comprehensive OCR Pipeline Test Suite")
            print("\nUsage:")
            print("  python run_comprehensive_tests.py         # Run all tests (100%, 50%, 25% scale)")
            print("  python run_comprehensive_tests.py --quick # Test only original image")
            print("  python run_comprehensive_tests.py --json  # Output results as JSON")
            print("\nTests include:")
            print("  - Character-level accuracy")
            print("  - Word-level accuracy")
            print("  - Field-level accuracy (electricity: 299 kWh, carbon: 120 kg CO2e)")
            print("  - Confidence score correlation")
            print("  - Downscaling robustness (50% and 25% scale)")
            sys.exit(0)
        
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
            sys.exit(1)
    
    else:
        # Default: run comprehensive tests
        returncode = run_comprehensive_tests(quick_mode=False)
        sys.exit(returncode)


if __name__ == "__main__":
    main()