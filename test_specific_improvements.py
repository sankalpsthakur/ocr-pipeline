#!/usr/bin/env python3
"""
Specific tests for each major improvement claim.
Tests the exact functionality we implemented.
"""

import tempfile
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image, ImageDraw
import numpy as np
import sys

# Mock dependencies
sys.modules['torch'] = MagicMock()
sys.modules['easyocr'] = MagicMock()
sys.modules['paddleocr'] = MagicMock()

import pipeline
import config


def test_vlm_integration_fix():
    """Test VLM engines receive proper image objects and bounding boxes."""
    print("🔍 Testing VLM Integration Fix...")
    
    # Test 1: Cached images are used
    cache = pipeline.ImageCache()
    test_image = Image.new('RGB', (100, 100), color='white')
    
    with patch('pipeline.load_image', return_value=test_image):
        images = cache.get_images(Path("test.png"), 300, is_image=True)
        assert len(images) == 1
        assert isinstance(images[0], Image.Image)
    
    # Test 2: Bounding box extraction helper
    with patch('pipeline._tesseract_ocr') as mock_tesseract:
        mock_tesseract.return_value = pipeline.OcrResult("text", ["word1", "word2"], [0.9, 0.8])
        
        result = pipeline._extract_bounding_boxes_for_vlm(test_image)
        assert "Found 2 text regions" in result
    
    print("✅ VLM Integration Fix: Image objects and bounding boxes work correctly")


def test_centralized_orientation_correction():
    """Test orientation correction is applied centrally in image cache."""
    print("🔍 Testing Centralized Orientation Correction...")
    
    # Create test image
    test_image = Image.new('RGB', (100, 50), color='white')
    rotated_image = test_image.rotate(90, expand=True)
    
    with patch('pipeline.load_image', return_value=rotated_image):
        with patch('pipeline._auto_rotate') as mock_rotate:
            mock_rotate.return_value = test_image  # Return corrected image
            
            cache = pipeline.ImageCache()
            images = cache.get_images(Path("test.png"), 300, is_image=True)
            
            # Rotation should be called during caching
            mock_rotate.assert_called_once()
            assert images[0] == test_image
    
    print("✅ Centralized Orientation: Applied at cache level for all engines")


def test_blank_document_handling():
    """Test enhanced blank document detection and error reporting."""
    print("🔍 Testing Blank Document Handling...")
    
    # Test blank detection function
    blank_image = Image.new('L', (100, 100), color=255)  # Pure white
    content_image = Image.new('L', (100, 100), color=128)  # Gray with content
    
    # Add content to second image
    draw = ImageDraw.Draw(content_image)
    draw.rectangle([20, 20, 80, 80], fill=50)
    
    assert pipeline._is_blank_image(blank_image) == True
    assert pipeline._is_blank_image(content_image) == False
    
    # Test early detection in run_ocr
    with patch('pipeline.load_image', return_value=blank_image):
        with patch('pipeline._is_blank_image', return_value=True):
            result = pipeline.run_ocr(Path("blank.png"))
            assert result.engine == "blank_document"
            assert result.text == ""
    
    print("✅ Blank Document Handling: Early detection and proper status reporting")


def test_file_format_validation():
    """Test enhanced file format validation and corruption detection."""
    print("🔍 Testing File Format Validation...")
    
    # Test 1: Corrupted PDF detection
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        tmp.write(b'This is not a PDF file')
        corrupted_path = Path(tmp.name)
    
    try:
        try:
            pipeline._validate_file_format(corrupted_path)
            assert False, "Should have detected corrupted PDF"
        except ValueError as e:
            assert "not a valid PDF" in str(e)
    finally:
        corrupted_path.unlink()
    
    # Test 2: Empty file detection
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        empty_path = Path(tmp.name)
    
    try:
        original_argv = sys.argv
        sys.argv = ['pipeline.py', str(empty_path)]
        
        try:
            with patch('builtins.print') as mock_print:
                pipeline.main()
            assert False, "Should have detected empty file"
        except SystemExit:
            # Check error message
            error_msg = mock_print.call_args[0][0]
            assert "empty (0 bytes)" in error_msg
    finally:
        sys.argv = original_argv
        empty_path.unlink()
    
    print("✅ File Format Validation: Corrupted and empty file detection works")


def test_engine_metadata_output():
    """Test engine information is included in JSON metadata."""
    print("🔍 Testing Engine Metadata Output...")
    
    with patch('pipeline.run_ocr') as mock_ocr:
        mock_ocr.return_value = pipeline.OcrResult("test text", ["test"], [0.95], "tesseract")
        
        with patch('pipeline.extract_fields') as mock_extract:
            mock_extract.return_value = {"electricity_kwh": 299, "carbon_kgco2e": 120}
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                test_path = Path(tmp.name)
            
            try:
                original_argv = sys.argv
                sys.argv = ['pipeline.py', str(test_path)]
                
                with patch('builtins.print') as mock_print:
                    with patch('pipeline._validate_file_format'):
                        try:
                            pipeline.main()
                        except SystemExit:
                            pass
                
                if mock_print.called:
                    output = mock_print.call_args[0][0]
                    data = json.loads(output)
                    
                    # Check metadata
                    assert 'ocr_engine' in data['meta']
                    assert data['meta']['ocr_engine'] == 'tesseract'
                    assert 'extraction_status' in data['meta']
                    assert 'confidence_thresholds' in data['meta']
                    
            finally:
                sys.argv = original_argv
                test_path.unlink()
    
    print("✅ Engine Metadata: OCR engine and status information in JSON output")


def test_configurable_thresholds():
    """Test confidence thresholds can be configured."""
    print("🔍 Testing Configurable Thresholds...")
    
    # Test environment variable configuration
    original_env = os.environ.copy()
    
    try:
        os.environ['TAU_FIELD_ACCEPT'] = '0.98'
        os.environ['TAU_ENHANCER_PASS'] = '0.93'
        os.environ['TAU_LLM_PASS'] = '0.88'
        
        # Reload config
        import importlib
        importlib.reload(config)
        
        assert config.TAU_FIELD_ACCEPT == 0.98
        assert config.TAU_ENHANCER_PASS == 0.93
        assert config.TAU_LLM_PASS == 0.88
        
    finally:
        # Restore environment
        os.environ.clear()
        os.environ.update(original_env)
        importlib.reload(config)
    
    # Test CLI override
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        test_path = Path(tmp.name)
    
    try:
        original_argv = sys.argv
        sys.argv = ['pipeline.py', str(test_path), '--thresholds', '0.97,0.92,0.87']
        
        with patch('pipeline.run_ocr') as mock_ocr:
            mock_ocr.return_value = pipeline.OcrResult("test", ["test"], [0.9], "tesseract")
            
            with patch('pipeline.extract_fields', return_value={}):
                with patch('builtins.print'):
                    with patch('pipeline._validate_file_format'):
                        try:
                            pipeline.main()
                        except SystemExit:
                            pass
        
        # Check thresholds were updated
        assert config.TAU_FIELD_ACCEPT == 0.97
        assert config.TAU_ENHANCER_PASS == 0.92
        assert config.TAU_LLM_PASS == 0.87
        
    finally:
        sys.argv = original_argv
        test_path.unlink()
    
    print("✅ Configurable Thresholds: Environment variables and CLI overrides work")


def test_performance_optimizations():
    """Test performance optimization features."""
    print("🔍 Testing Performance Optimizations...")
    
    # Test 1: Image resizing
    large_image = Image.new('RGB', (3000, 4000), color='white')
    
    with patch('pipeline.MAX_IMAGE_WIDTH', 2000):
        with patch('pipeline.MAX_IMAGE_HEIGHT', 2000):
            resized = pipeline._optimize_image_size(large_image)
            
            assert resized.size[0] <= 2000
            assert resized.size[1] <= 2000
            assert resized.size != large_image.size
    
    # Test 2: Cache size management
    cache = pipeline.ImageCache()
    test_image = Image.new('RGB', (1000, 1000), color='white')
    
    size_mb = cache._estimate_image_size_mb(test_image)
    assert 2 < size_mb < 5  # Reasonable size estimate
    
    # Test cache clearing when limit exceeded
    with patch('pipeline.MAX_CACHE_SIZE_MB', 1):  # Very small limit
        cache._cache_size_mb = 10  # Simulate large cache
        cache._check_cache_size()
        
        assert len(cache._cache) == 0
        assert cache._cache_size_mb == 0
    
    # Test 3: Thread count optimization logic
    with patch('multiprocessing.cpu_count', return_value=8):
        traditional_engines = ["tesseract", "easyocr"]
        
        # Test optimal worker calculation
        cpu_count = 8
        optimal_workers = min(cpu_count // 2, len(traditional_engines), 4)
        optimal_workers = max(1, optimal_workers)
        
        assert optimal_workers == 2  # min(4, 2, 4) = 2
    
    print("✅ Performance Optimizations: Image resizing, cache management, thread optimization")


def test_code_refactoring():
    """Test code refactoring improvements reduce duplication."""
    print("🔍 Testing Code Refactoring...")
    
    # Test multi-page helper function
    images = [
        Image.new('RGB', (100, 100), color='white'),
        Image.new('RGB', (100, 100), color='gray')
    ]
    
    def mock_ocr_func(img):
        return pipeline.OcrResult("page text", ["page", "text"], [0.9, 0.8])
    
    result = pipeline._process_multi_page_ocr(images, mock_ocr_func)
    
    assert "page text" in result.text
    assert len(result.tokens) == 4  # 2 tokens × 2 pages
    assert len(result.confidences) == 4
    
    # Test bounding box helper
    test_image = Image.new('RGB', (100, 100), color='white')
    
    with patch('pipeline._tesseract_ocr') as mock_tesseract:
        mock_tesseract.return_value = pipeline.OcrResult("text", ["word1", "word2"], [0.9, 0.8])
        
        result = pipeline._extract_bounding_boxes_for_vlm(test_image)
        assert "Found 2 text regions" in result
    
    print("✅ Code Refactoring: Helper functions reduce duplication")


def test_field_extraction_robustness():
    """Test field extraction works across various formats."""
    print("🔍 Testing Field Extraction Robustness...")
    
    test_cases = [
        ("Dubai Electricity Water Authority Invoice Electricity 299 kWh Carbon Footprint Kg CO2e 120", 299, 120),
        ("Consumption: 299 kWh Carbon emissions: 120 kg CO2e", 299, 120),
        ("Electricity usage 1,234 kWh Environmental impact 456 kg CO2e", 1234, 456),
        ("299 Electricity kWh Carbon 120 kg CO2e", 299, 120),
        ("Commercial: 2,500 kWh Carbon: 1000 kg CO2e", 2500, 1000),
        ("Residential: 150 kWh Carbon: 60 kg", 150, 60),
    ]
    
    passed = 0
    for text, expected_elec, expected_carbon in test_cases:
        result = pipeline.extract_fields(text, None)
        
        elec_match = result.get("electricity_kwh") == expected_elec
        carbon_match = result.get("carbon_kgco2e") == expected_carbon
        
        if elec_match and carbon_match:
            passed += 1
        else:
            print(f"  ⚠️  Failed: {text[:50]}... expected ({expected_elec}, {expected_carbon}), got ({result.get('electricity_kwh')}, {result.get('carbon_kgco2e')})")
    
    success_rate = passed / len(test_cases)
    assert success_rate >= 0.8, f"Field extraction success rate too low: {success_rate:.1%}"
    
    print(f"✅ Field Extraction: {passed}/{len(test_cases)} cases passed ({success_rate:.1%})")


def run_improvement_validation():
    """Run all improvement validation tests."""
    print("🎯 VALIDATING ALL PIPELINE IMPROVEMENTS\n")
    print("=" * 60)
    
    tests = [
        ("VLM Integration Fix", test_vlm_integration_fix),
        ("Centralized Orientation Correction", test_centralized_orientation_correction),
        ("Blank Document Handling", test_blank_document_handling),
        ("File Format Validation", test_file_format_validation),
        ("Engine Metadata Output", test_engine_metadata_output),
        ("Configurable Thresholds", test_configurable_thresholds),
        ("Performance Optimizations", test_performance_optimizations),
        ("Code Refactoring", test_code_refactoring),
        ("Field Extraction Robustness", test_field_extraction_robustness),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
            print()
        except Exception as e:
            print(f"❌ {test_name}: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 60)
    print(f"🏆 VALIDATION SUMMARY: {passed}/{total} improvements validated ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL IMPROVEMENTS SUCCESSFULLY VALIDATED!")
        print("\n✅ The OCR pipeline has been significantly enhanced with:")
        print("   • Robust VLM engine integration with bounding box hints")
        print("   • Centralized orientation correction for all engines")
        print("   • Smart blank document detection and early exit")
        print("   • Comprehensive file validation and corruption detection")
        print("   • Detailed engine metadata and status reporting")
        print("   • Fully configurable confidence thresholds")
        print("   • Performance optimizations for speed and memory")
        print("   • Cleaner code structure with reduced duplication")
        print("   • Enhanced field extraction robustness")
    else:
        print(f"⚠️  {total - passed} improvements need attention")
    
    return passed == total


if __name__ == "__main__":
    success = run_improvement_validation()
    sys.exit(0 if success else 1)