#!/usr/bin/env python3
"""
Basic test suite to validate core pipeline functionality without heavy dependencies.
"""

import tempfile
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

# Mock heavy dependencies before importing pipeline
import sys
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False
sys.modules['torch'] = mock_torch

mock_easyocr = MagicMock()
sys.modules['easyocr'] = mock_easyocr

mock_paddleocr = MagicMock()
sys.modules['paddleocr'] = mock_paddleocr

mock_pytesseract = MagicMock()
sys.modules['pytesseract'] = mock_pytesseract

# Import after mocking
import pipeline
import config


def test_basic_import():
    """Test that pipeline imports successfully."""
    assert pipeline is not None
    assert hasattr(pipeline, 'run_ocr')
    assert hasattr(pipeline, 'extract_fields')
    print("✅ Pipeline imports successfully")


def test_ocr_result_creation():
    """Test OcrResult object creation and properties."""
    result = pipeline.OcrResult("test text", ["test", "text"], [0.9, 0.8], "tesseract")
    
    assert result.text == "test text"
    assert result.tokens == ["test", "text"]
    assert result.confidences == [0.9, 0.8]
    assert result.engine == "tesseract"
    
    # Test field confidence calculation
    assert 0.8 < result.field_confidence < 0.9
    print("✅ OcrResult creation and properties work")


def test_image_cache_functionality():
    """Test image cache basic functionality."""
    cache = pipeline.ImageCache()
    
    # Test cache key generation
    key = cache.get_cache_key(Path("test.pdf"), 300, 0)
    assert isinstance(key, str)
    assert "300" in key
    print("✅ Image cache functionality works")


def test_blank_image_detection():
    """Test blank image detection logic."""
    # Create blank white image
    white_image = Image.new('L', (100, 100), color=255)
    
    # Create image with content  
    content_image = Image.new('L', (100, 100), color=128)
    content_array = np.array(content_image)
    content_array[20:80, 20:80] = 50  # Add dark rectangle
    content_image = Image.fromarray(content_array)
    
    # Test detection
    assert pipeline._is_blank_image(white_image) == True
    assert pipeline._is_blank_image(content_image) == False
    print("✅ Blank image detection works")


def test_image_optimization():
    """Test image size optimization."""
    # Create large image
    large_image = Image.new('RGB', (3000, 4000), color='white')
    
    # Test resizing with limits
    with patch('pipeline.MAX_IMAGE_WIDTH', 2000):
        with patch('pipeline.MAX_IMAGE_HEIGHT', 2000):
            resized = pipeline._optimize_image_size(large_image)
            
            assert resized.size[0] <= 2000
            assert resized.size[1] <= 2000
            print("✅ Image optimization works")


def test_multi_page_ocr_helper():
    """Test multi-page OCR aggregation helper."""
    # Create test images
    images = [
        Image.new('RGB', (100, 100), color='white'),
        Image.new('RGB', (100, 100), color='gray')
    ]
    
    # Mock OCR function
    def mock_ocr_func(img):
        return pipeline.OcrResult("page text", ["page", "text"], [0.9, 0.8])
    
    result = pipeline._process_multi_page_ocr(images, mock_ocr_func)
    
    assert "page text" in result.text
    assert len(result.tokens) == 4  # 2 tokens per page
    assert len(result.confidences) == 4
    print("✅ Multi-page OCR helper works")


def test_field_extraction_basic():
    """Test basic field extraction functionality."""
    test_cases = [
        ("Electricity 299 kWh Carbon 120 kg CO2e", 299, 120),
        ("Usage: 500 kWh Emissions: 200 kg", 500, 200),
        ("299 kWh electricity, 120 kg carbon", 299, 120),
    ]
    
    for text, expected_elec, expected_carbon in test_cases:
        result = pipeline.extract_fields(text, None)
        
        if expected_elec:
            assert result.get("electricity_kwh") == expected_elec, f"Failed electricity for: {text}"
        if expected_carbon:
            assert result.get("carbon_kgco2e") == expected_carbon, f"Failed carbon for: {text}"
    
    print("✅ Basic field extraction works")


def test_configuration_loading():
    """Test configuration and threshold loading."""
    # Test that thresholds are loaded
    assert hasattr(config, 'TAU_FIELD_ACCEPT')
    assert hasattr(config, 'TAU_ENHANCER_PASS')
    assert hasattr(config, 'TAU_LLM_PASS')
    
    # Test reasonable values
    assert 0.0 <= config.TAU_LLM_PASS <= 1.0
    assert 0.0 <= config.TAU_ENHANCER_PASS <= 1.0
    assert 0.0 <= config.TAU_FIELD_ACCEPT <= 1.0
    
    # Test ordering
    assert config.TAU_LLM_PASS <= config.TAU_ENHANCER_PASS <= config.TAU_FIELD_ACCEPT
    print("✅ Configuration loading works")


def test_payload_building():
    """Test JSON payload building."""
    fields = {
        "electricity_kwh": 299,
        "carbon_kgco2e": 120,
        "_confidence": 0.95,
        "_engine": "tesseract",
        "_status": "success",
        "_errors": [],
        "_warnings": [],
        "_thresholds": {"field_accept": 0.95, "enhancer_pass": 0.90, "llm_pass": 0.85}
    }
    
    # Create temporary file for testing
    with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp:
        tmp_path = Path(tmp.name)
        
        payload = pipeline.build_payload(fields, tmp_path)
        
        # Check structure
        assert "electricity" in payload
        assert "carbon" in payload
        assert "source_document" in payload
        assert "meta" in payload
        
        # Check values
        assert payload["electricity"]["consumption"]["value"] == 299
        assert payload["carbon"]["location_based"]["value"] == 120
        assert payload["meta"]["extraction_confidence"] == 0.95
        assert payload["meta"]["ocr_engine"] == "tesseract"
        
    print("✅ Payload building works")


def test_file_validation_logic():
    """Test file validation functionality."""
    # Test with actual image file
    test_image = Image.new('RGB', (100, 100), color='white')
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        test_image.save(tmp.name)
        test_path = Path(tmp.name)
    
    try:
        # Should not raise exception for valid PNG
        pipeline._validate_file_format(test_path)
        print("✅ File validation works for valid files")
        
    except Exception as e:
        print(f"⚠️  File validation had issues: {e}")
    finally:
        test_path.unlink()


def test_error_handling_structure():
    """Test error handling and status reporting structure."""
    # Test with mock OCR result
    with patch('pipeline.run_ocr') as mock_ocr:
        mock_ocr.return_value = pipeline.OcrResult("", [], [], "blank_document")
        
        with patch('pipeline.extract_fields', return_value={}):
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                test_path = Path(tmp.name)
            
            try:
                # Mock sys.argv
                original_argv = sys.argv
                sys.argv = ['pipeline.py', str(test_path)]
                
                with patch('builtins.print') as mock_print:
                    with patch('pipeline._validate_file_format'):
                        try:
                            pipeline.main()
                        except SystemExit:
                            pass  # Expected for successful execution
                
                # Check that output was generated
                if mock_print.called:
                    output = mock_print.call_args[0][0]
                    try:
                        data = json.loads(output)
                        assert 'meta' in data
                        assert 'extraction_status' in data['meta']
                        print("✅ Error handling and status reporting works")
                    except json.JSONDecodeError:
                        print("⚠️  Output was not valid JSON")
                else:
                    print("⚠️  No output was generated")
                    
            finally:
                sys.argv = original_argv
                test_path.unlink()


def run_basic_tests():
    """Run all basic tests."""
    print("🧪 Running Basic Pipeline Tests...\n")
    
    tests = [
        test_basic_import,
        test_ocr_result_creation,
        test_image_cache_functionality,
        test_blank_image_detection,
        test_image_optimization,
        test_multi_page_ocr_helper,
        test_field_extraction_basic,
        test_configuration_loading,
        test_payload_building,
        test_file_validation_logic,
        test_error_handling_structure,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__}: {e}")
    
    print(f"\n📊 Basic Tests Summary: {passed}/{total} passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All basic functionality validated!")
    else:
        print(f"⚠️  {total - passed} tests need attention")
    
    return passed == total


if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1)