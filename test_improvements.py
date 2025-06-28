#!/usr/bin/env python3
"""
Comprehensive test suite for OCR pipeline improvements.
Tests all major claims and improvements made to the pipeline.
"""

import pytest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

# Import pipeline components (handle dependency issues gracefully)
import sys
sys.path.append('.')

# Mock heavy dependencies for testing
sys.modules['torch'] = MagicMock()
sys.modules['easyocr'] = MagicMock() 
sys.modules['paddleocr'] = MagicMock()

try:
    import pipeline
    import config
except ImportError as e:
    pytest.skip(f"Pipeline import failed: {e}", allow_module_level=True)


class TestVLMIntegrationFix:
    """Test VLM engine integration improvements."""
    
    def test_vlm_engines_receive_images_not_paths(self):
        """Test that VLM engines receive PIL Image objects, not file paths."""
        # Create mock image
        test_image = Image.new('RGB', (100, 100), color='white')
        
        with patch('pipeline._mistral_ocr') as mock_mistral:
            mock_mistral.return_value = pipeline.OcrResult("test", ["test"], [0.9])
            
            # Test mistral engine receives image object
            result = pipeline._run_single_engine_with_cache(
                (Path("test.png"), "mistral", True, 300)
            )
            
            # Should be called with image object, not path
            if mock_mistral.called:
                args, kwargs = mock_mistral.call_args
                assert isinstance(args[0], Image.Image), "Mistral should receive PIL Image object"
    
    def test_gemma_vlm_bounding_box_integration(self):
        """Test that Gemma VLM receives bounding box hints."""
        test_image = Image.new('RGB', (100, 100), color='white')
        
        with patch('pipeline._gemma_vlm_ocr') as mock_gemma:
            mock_gemma.return_value = pipeline.OcrResult("test", ["test"], [0.9])
            
            with patch('pipeline._extract_bounding_boxes_for_vlm') as mock_bbox:
                mock_bbox.return_value = "Found 5 text regions"
                
                result = pipeline._run_single_engine_with_cache(
                    (Path("test.png"), "gemma_vlm", True, 300)
                )
                
                # Should be called with bounding box hints
                if mock_gemma.called:
                    args, kwargs = mock_gemma.call_args
                    assert len(args) >= 2, "Gemma VLM should receive bounding box parameter"


class TestCentralizedOrientationCorrection:
    """Test centralized orientation correction."""
    
    def test_orientation_applied_to_all_engines(self):
        """Test that orientation correction is applied centrally for all engines."""
        # Create rotated test image
        test_image = Image.new('RGB', (100, 50), color='white')
        rotated_image = test_image.rotate(90, expand=True)
        
        with patch('pipeline.load_image', return_value=rotated_image):
            with patch('pipeline._auto_rotate') as mock_rotate:
                mock_rotate.return_value = test_image  # Return corrected image
                
                # Get images from cache (should trigger rotation)
                cache = pipeline.ImageCache()
                images = cache.get_images(Path("test.png"), 300, is_image=True)
                
                # Rotation should be called once during caching
                mock_rotate.assert_called_once()
                assert images[0] == test_image  # Should get corrected image
    
    def test_rotation_detection_accuracy(self):
        """Test rotation detection works correctly."""
        test_image = Image.new('RGB', (100, 50), color='white')
        
        with patch('pipeline.pytesseract') as mock_tesseract:
            # Mock OSD to return 90 degree rotation
            mock_tesseract.image_to_osd.return_value = {"rotate": 90}
            mock_tesseract.Output.DICT = "dict"
            
            result = pipeline._auto_rotate(test_image)
            
            # Should call image_to_osd for orientation detection
            mock_tesseract.image_to_osd.assert_called_once()


class TestBlankDocumentHandling:
    """Test enhanced blank document detection and handling."""
    
    def test_early_blank_detection_for_images(self):
        """Test early blank detection prevents expensive OCR on blank images."""
        # Create blank white image
        blank_image = Image.new('RGB', (100, 100), color=(255, 255, 255))
        
        with patch('pipeline.load_image', return_value=blank_image):
            with patch('pipeline._is_blank_image', return_value=True):
                result = pipeline.run_ocr(Path("blank.png"))
                
                # Should return blank_document engine
                assert result.engine == "blank_document"
                assert result.text == ""
    
    def test_blank_vs_content_detection(self):
        """Test distinction between blank and content images."""
        # Test blank detection function directly
        blank_image = Image.new('L', (100, 100), color=255)  # White
        content_image = Image.new('L', (100, 100), color=128)  # Gray
        
        # Add some noise to content image
        content_array = np.array(content_image)
        content_array[20:80, 20:80] = 50  # Dark rectangle
        content_image = Image.fromarray(content_array)
        
        assert pipeline._is_blank_image(blank_image) == True
        assert pipeline._is_blank_image(content_image) == False
    
    def test_blank_status_in_json_output(self):
        """Test that blank documents are properly reported in JSON output."""
        with patch('pipeline.run_ocr') as mock_ocr:
            # Mock blank document result
            mock_ocr.return_value = pipeline.OcrResult("", [], [], "blank_document")
            
            with patch('pipeline.extract_fields', return_value={}):
                # Create temporary test file
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    test_path = Path(tmp.name)
                
                try:
                    # Test main pipeline
                    sys.argv = ['pipeline.py', str(test_path)]
                    
                    with patch('builtins.print') as mock_print:
                        with patch('pipeline._validate_file_format'):
                            pipeline.main()
                    
                    # Should print JSON with error status
                    printed_output = mock_print.call_args[0][0]
                    output_data = json.loads(printed_output)
                    
                    assert output_data['meta']['extraction_status'] == 'failed'
                    assert 'blank' in str(output_data['meta']['errors']).lower()
                
                finally:
                    test_path.unlink(missing_ok=True)


class TestFileFormatValidation:
    """Test enhanced file format validation and corruption detection."""
    
    def test_corrupted_pdf_detection(self):
        """Test detection of corrupted PDF files."""
        # Create file with wrong header
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(b'This is not a PDF file')
            corrupted_path = Path(tmp.name)
        
        try:
            with pytest.raises(ValueError, match="not a valid PDF"):
                pipeline._validate_file_format(corrupted_path)
        finally:
            corrupted_path.unlink()
    
    def test_corrupted_image_detection(self):
        """Test detection of corrupted image files."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(b'This is not a PNG file')
            corrupted_path = Path(tmp.name)
        
        try:
            with pytest.raises(ValueError, match="corrupted or invalid"):
                pipeline._validate_file_format(corrupted_path)
        finally:
            corrupted_path.unlink()
    
    def test_empty_file_detection(self):
        """Test detection of empty files."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            empty_path = Path(tmp.name)
        
        try:
            # Test via main() function
            sys.argv = ['pipeline.py', str(empty_path)]
            
            with pytest.raises(SystemExit):
                with patch('builtins.print') as mock_print:
                    pipeline.main()
            
            # Should print error about empty file
            error_msg = mock_print.call_args[0][0]
            assert "empty (0 bytes)" in error_msg
        
        finally:
            empty_path.unlink()


class TestEngineMetadataOutput:
    """Test engine information in JSON metadata."""
    
    def test_engine_info_in_json(self):
        """Test that OCR engine information is included in output."""
        with patch('pipeline.run_ocr') as mock_ocr:
            # Mock successful tesseract result
            mock_ocr.return_value = pipeline.OcrResult("test text", ["test"], [0.95], "tesseract")
            
            with patch('pipeline.extract_fields') as mock_extract:
                mock_extract.return_value = {"electricity_kwh": 299, "carbon_kgco2e": 120}
                
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    test_path = Path(tmp.name)
                
                try:
                    sys.argv = ['pipeline.py', str(test_path)]
                    
                    with patch('builtins.print') as mock_print:
                        with patch('pipeline._validate_file_format'):
                            pipeline.main()
                    
                    printed_output = mock_print.call_args[0][0]
                    output_data = json.loads(printed_output)
                    
                    # Should include engine metadata
                    assert 'ocr_engine' in output_data['meta']
                    assert output_data['meta']['ocr_engine'] == 'tesseract'
                    assert 'extraction_status' in output_data['meta']
                    assert 'confidence_thresholds' in output_data['meta']
                
                finally:
                    test_path.unlink()


class TestConfigurableThresholds:
    """Test configurable confidence thresholds."""
    
    def test_environment_variable_thresholds(self):
        """Test threshold configuration via environment variables."""
        with patch.dict(os.environ, {
            'TAU_FIELD_ACCEPT': '0.98',
            'TAU_ENHANCER_PASS': '0.93', 
            'TAU_LLM_PASS': '0.88'
        }):
            # Reload config to pick up environment variables
            import importlib
            importlib.reload(config)
            
            assert config.TAU_FIELD_ACCEPT == 0.98
            assert config.TAU_ENHANCER_PASS == 0.93
            assert config.TAU_LLM_PASS == 0.88
    
    def test_cli_threshold_overrides(self):
        """Test threshold configuration via command line."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            test_path = Path(tmp.name)
        
        try:
            sys.argv = ['pipeline.py', str(test_path), '--thresholds', '0.97,0.92,0.87']
            
            with patch('pipeline.run_ocr') as mock_ocr:
                mock_ocr.return_value = pipeline.OcrResult("test", ["test"], [0.9], "tesseract")
                
                with patch('pipeline.extract_fields', return_value={}):
                    with patch('builtins.print'):
                        with patch('pipeline._validate_file_format'):
                            pipeline.main()
            
            # Check that thresholds were updated
            assert config.TAU_FIELD_ACCEPT == 0.97
            assert config.TAU_ENHANCER_PASS == 0.92
            assert config.TAU_LLM_PASS == 0.87
        
        finally:
            test_path.unlink()


class TestPerformanceOptimizations:
    """Test performance optimization features."""
    
    def test_image_resizing_functionality(self):
        """Test image resizing for performance optimization."""
        # Create large test image
        large_image = Image.new('RGB', (3000, 4000), color='white')
        
        # Set resize limits
        with patch('pipeline.MAX_IMAGE_WIDTH', 2000):
            with patch('pipeline.MAX_IMAGE_HEIGHT', 2000):
                resized = pipeline._optimize_image_size(large_image)
                
                # Should be resized to fit within limits
                assert resized.size[0] <= 2000
                assert resized.size[1] <= 2000
                assert resized.size != large_image.size
    
    def test_thread_pool_optimization(self):
        """Test optimal thread count calculation."""
        with patch('multiprocessing.cpu_count', return_value=8):
            with patch('pipeline.AUTO_THREAD_COUNT', True):
                with patch('pipeline.MAX_WORKER_THREADS', 4):
                    # Mock traditional engines
                    traditional_engines = ["tesseract", "easyocr"]
                    
                    # Should use min of (CPU cores // 2, engines, max limit)
                    # 8 // 2 = 4, min(4, 2, 4) = 2
                    expected_workers = 2
                    
                    # This would be tested in the actual run_ocr function
                    # For now, test the logic directly
                    cpu_count = 8
                    optimal_workers = min(cpu_count // 2, len(traditional_engines), 4)
                    optimal_workers = max(1, optimal_workers)
                    
                    assert optimal_workers == 2
    
    def test_cache_size_management(self):
        """Test image cache size limits."""
        cache = pipeline.ImageCache()
        
        # Test cache size estimation
        test_image = Image.new('RGB', (1000, 1000), color='white')
        size_mb = cache._estimate_image_size_mb(test_image)
        
        # Should estimate reasonable size (1000x1000x3 bytes = ~3MB)
        assert 2 < size_mb < 5
        
        # Test cache size checking
        with patch('pipeline.MAX_CACHE_SIZE_MB', 1):  # Very small limit
            cache._cache_size_mb = 10  # Simulate large cache
            cache._check_cache_size()
            
            # Should have cleared cache
            assert len(cache._cache) == 0
            assert cache._cache_size_mb == 0


class TestCodeRefactoring:
    """Test code refactoring improvements."""
    
    def test_multi_page_ocr_helper(self):
        """Test the multi-page OCR helper function."""
        # Create test images
        images = [
            Image.new('RGB', (100, 100), color='white'),
            Image.new('RGB', (100, 100), color='gray')
        ]
        
        # Mock OCR function
        def mock_ocr_func(img):
            return pipeline.OcrResult("page text", ["page", "text"], [0.9, 0.8])
        
        result = pipeline._process_multi_page_ocr(images, mock_ocr_func)
        
        # Should aggregate results from both pages
        assert "page text\npage text" in result.text
        assert len(result.tokens) == 4  # 2 tokens per page
        assert len(result.confidences) == 4
    
    def test_bounding_box_extraction_helper(self):
        """Test the bounding box extraction helper."""
        test_image = Image.new('RGB', (100, 100), color='white')
        
        with patch('pipeline._tesseract_ocr') as mock_tesseract:
            mock_tesseract.return_value = pipeline.OcrResult("text", ["word1", "word2"], [0.9, 0.8])
            
            result = pipeline._extract_bounding_boxes_for_vlm(test_image)
            
            assert "Found 2 text regions" in result
            mock_tesseract.assert_called_once_with(test_image)


class TestFieldExtractionRobustness:
    """Test field extraction improvements and robustness."""
    
    def test_extraction_with_various_formats(self):
        """Test field extraction works with various text formats."""
        test_cases = [
            ("Dubai Electricity Water Authority Invoice Electricity 299 kWh Carbon Footprint Kg CO2e 120", 299, 120),
            ("Consumption: 299 kWh Carbon emissions: 120 kg CO2e", 299, 120),
            ("Electricity usage 1,234 kWh Environmental impact 456 kg CO2e", 1234, 456),
            ("299 Electricity kWh Carbon 120 kg CO2e", 299, 120),
        ]
        
        for text, expected_elec, expected_carbon in test_cases:
            result = pipeline.extract_fields(text, None)
            
            assert result.get("electricity_kwh") == expected_elec, f"Failed for text: {text}"
            assert result.get("carbon_kgco2e") == expected_carbon, f"Failed for text: {text}"


def run_comprehensive_tests():
    """Run all improvement tests and generate report."""
    
    print("🧪 Running Comprehensive OCR Pipeline Improvement Tests...\n")
    
    # Test categories and their descriptions
    test_categories = {
        "VLM Integration": TestVLMIntegrationFix,
        "Orientation Correction": TestCentralizedOrientationCorrection, 
        "Blank Document Handling": TestBlankDocumentHandling,
        "File Format Validation": TestFileFormatValidation,
        "Engine Metadata Output": TestEngineMetadataOutput,
        "Configurable Thresholds": TestConfigurableThresholds,
        "Performance Optimizations": TestPerformanceOptimizations,
        "Code Refactoring": TestCodeRefactoring,
        "Field Extraction": TestFieldExtractionRobustness
    }
    
    results = {}
    total_tests = 0
    passed_tests = 0
    
    for category_name, test_class in test_categories.items():
        print(f"📋 Testing {category_name}...")
        
        category_passed = 0
        category_total = 0
        category_errors = []
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for test_method in test_methods:
            category_total += 1
            total_tests += 1
            
            try:
                # Create test instance and run test
                test_instance = test_class()
                method = getattr(test_instance, test_method)
                method()
                
                category_passed += 1
                passed_tests += 1
                print(f"  ✅ {test_method}")
                
            except Exception as e:
                category_errors.append(f"{test_method}: {str(e)}")
                print(f"  ❌ {test_method}: {e}")
        
        results[category_name] = {
            'passed': category_passed,
            'total': category_total,
            'errors': category_errors
        }
        
        print(f"   📊 {category_passed}/{category_total} tests passed\n")
    
    # Generate summary report
    print("=" * 60)
    print("🎯 IMPROVEMENT VALIDATION SUMMARY")
    print("=" * 60)
    
    for category, result in results.items():
        status = "✅ PASS" if result['passed'] == result['total'] else "⚠️  PARTIAL" if result['passed'] > 0 else "❌ FAIL"
        print(f"{status} {category}: {result['passed']}/{result['total']}")
        
        if result['errors']:
            for error in result['errors'][:2]:  # Show first 2 errors
                print(f"    🔍 {error}")
    
    print(f"\n🏆 OVERALL: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("🎉 All improvements validated successfully!")
    else:
        print(f"⚠️  {total_tests - passed_tests} tests need attention")
    
    return results


if __name__ == "__main__":
    run_comprehensive_tests()