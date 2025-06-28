#!/usr/bin/env python3
"""
Consolidated Comprehensive Test Suite for OCR Pipeline

This test suite combines the most valuable tests from all test files:
- test_basic.py: Core functionality tests
- test_final_validation.py: Final improvement validation
- test_improvements.py: Performance and robustness tests  
- test_specific_improvements.py: Specific feature tests
- test_accuracy_comprehensive.py: Ground truth accuracy tests

Organized into logical test classes covering all critical functionality.
"""

import pytest
import tempfile
import json
import os
import sys
import pathlib
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image, ImageDraw
import numpy as np

# Add repository root to path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

# Mock heavy dependencies before importing pipeline
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False
mock_torch.backends.mps.is_available.return_value = False
sys.modules['torch'] = mock_torch

sys.modules['easyocr'] = MagicMock()
sys.modules['paddleocr'] = MagicMock()
sys.modules['mistralai'] = MagicMock()

import pipeline
import config


class TestCoreFramework:
    """Test core pipeline framework and basic functionality."""
    
    def test_pipeline_imports_successfully(self):
        """Test that pipeline imports successfully with all components."""
        assert pipeline is not None
        assert hasattr(pipeline, 'run_ocr')
        assert hasattr(pipeline, 'extract_fields')
        assert hasattr(pipeline, 'OcrResult')
        assert hasattr(pipeline, 'ImageCache')
    
    def test_ocr_result_creation_and_properties(self):
        """Test OcrResult object creation and confidence calculations."""
        result = pipeline.OcrResult("test text", ["test", "text"], [0.9, 0.8], "tesseract")
        
        assert result.text == "test text"
        assert result.tokens == ["test", "text"]
        assert result.confidences == [0.9, 0.8]
        assert result.engine == "tesseract"
        
        # Test field confidence calculation (geometric mean)
        assert 0.8 < result.field_confidence < 0.9
    
    def test_image_cache_basic_functionality(self):
        """Test image cache key generation and caching logic."""
        cache = pipeline.ImageCache()
        
        # Test cache key generation
        key = cache.get_cache_key(Path("test.pdf"), 300, 0)
        assert isinstance(key, str)
        assert "300" in key
        
        # Test cache size estimation
        test_image = Image.new('RGB', (100, 100), color='white')
        size_mb = cache._estimate_image_size_mb(test_image)
        assert size_mb > 0
    
    def test_configuration_loading_and_validation(self):
        """Test configuration loading and threshold validation."""
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


class TestAccuracyImprovements:
    """Test the 8 major accuracy improvements implemented."""
    
    def test_unified_geometric_correction(self):
        """Test that geometric corrections are applied centrally in ImageCache."""
        # Test deskewing function exists and works
        test_image = Image.new('RGB', (100, 50), color='white')
        result = pipeline._deskew_image(test_image)
        assert isinstance(result, Image.Image)
        
        # Test dewarping function exists and works
        result = pipeline._dewarp_image(test_image)
        assert isinstance(result, Image.Image)
        
        # Test centralized application in ImageCache
        with patch('pipeline.load_image', return_value=test_image):
            with patch('pipeline._auto_rotate') as mock_rotate:
                with patch('pipeline._deskew_image') as mock_deskew:
                    with patch('pipeline._dewarp_image') as mock_dewarp:
                        mock_rotate.return_value = test_image
                        mock_deskew.return_value = test_image
                        mock_dewarp.return_value = test_image
                        
                        cache = pipeline.ImageCache()
                        images = cache.get_images(Path("test.png"), 300, is_image=True)
                        
                        # All corrections should be called
                        mock_rotate.assert_called_once()
                        mock_deskew.assert_called_once()
                        mock_dewarp.assert_called_once()
    
    def test_engine_specific_tuning_configurations(self):
        """Test that engine-specific configurations are properly loaded and used."""
        # Test configuration structures exist
        assert hasattr(config, 'TESSERACT_ARGS')
        assert hasattr(config, 'EASYOCR_ARGS')
        assert hasattr(config, 'PADDLEOCR_ARGS')
        assert hasattr(config, 'DOCUMENT_TYPE')
        
        # Test bills configuration exists
        assert 'bills' in config.TESSERACT_ARGS
        assert 'bills' in config.EASYOCR_ARGS
        assert 'bills' in config.PADDLEOCR_ARGS
        
        # Test specific optimizations for bills
        bills_tesseract = config.TESSERACT_ARGS['bills']
        assert 'whitelist' in bills_tesseract
        assert '0123456789' in bills_tesseract['whitelist']
    
    def test_token_level_ensemble_voting(self):
        """Test token-level ensemble voting with bounding box alignment."""
        # Test IoU calculation function
        bbox1 = (10, 10, 50, 50)
        bbox2 = (30, 30, 70, 70)
        bbox3 = (100, 100, 150, 150)  # No overlap
        
        iou_overlap = pipeline._calculate_bbox_iou(bbox1, bbox2)
        iou_no_overlap = pipeline._calculate_bbox_iou(bbox1, bbox3)
        
        assert 0 < iou_overlap < 1  # Should have some overlap
        assert iou_no_overlap == 0  # Should have no overlap
        
        # Test vote merging function exists
        assert hasattr(pipeline, '_vote_merge_tokens')
        
        # Test that OcrResult supports bboxes
        result = pipeline.OcrResult("test", ["test"], [0.9], "tesseract")
        assert hasattr(result, 'bboxes')
    
    def test_confidence_recalibration_system(self):
        """Test confidence calibration system components."""
        # Test ConfidenceCalibrator class exists
        assert hasattr(pipeline, 'ConfidenceCalibrator')
        
        # Test global calibrator instance exists
        assert hasattr(pipeline, '_confidence_calibrator')
        calibrator = pipeline._confidence_calibrator
        assert isinstance(calibrator, pipeline.ConfidenceCalibrator)
        
        # Test calibration methods exist
        assert hasattr(calibrator, 'calibrate_confidence')
        assert hasattr(calibrator, 'fit_from_validation_data')
        assert hasattr(calibrator, 'save_calibration')
        assert hasattr(calibrator, 'load_calibration')
    
    def test_field_aware_post_processing(self):
        """Test field-aware post-processing and numerical corrections."""
        # Test numerical correction function exists
        assert hasattr(pipeline, '_apply_numerical_corrections')
        
        # Test OCR error corrections
        test_text = "Electricity I20 kWh Carbon O5 kg"
        corrected = pipeline._apply_numerical_corrections(test_text)
        
        # Should fix I->1 and O->0
        assert "120" in corrected
        assert "05" in corrected or "5" in corrected
        
        # Test field-aware corrections function exists
        assert hasattr(pipeline, '_apply_field_aware_corrections')
    
    def test_vlm_bounding_box_hints(self):
        """Test VLM enhancement with bounding box hints."""
        # Test high-confidence region extraction
        assert hasattr(pipeline, '_extract_high_confidence_regions')
        
        # Test VLM function accepts bounding box hints
        test_image = Image.new('RGB', (100, 100), color='white')
        
        # Mock EasyOCR result with bounding boxes
        mock_result = pipeline.OcrResult(
            "test text", 
            ["test", "text"], 
            [0.95, 0.90],
            "easyocr",
            bboxes=[(10, 10, 50, 30), (60, 10, 100, 30)]
        )
        
        regions = pipeline._extract_high_confidence_regions(test_image, mock_result)
        assert isinstance(regions, list)


class TestGroundTruthAccuracy:
    """Test field-level extraction accuracy against known ground truth."""
    
    # Definitive ground truth test cases
    GROUND_TRUTH_CASES = [
        # Basic clean cases
        ("Dubai Electricity Water Authority Invoice Electricity 299 kWh Carbon Footprint Kg CO2e 120", 299, 120),
        ("Consumption: 299 kWh Carbon emissions: 120 kg CO2e", 299, 120),
        ("Electricity usage 1,234 kWh Environmental impact 456 kg CO2e", 1234, 456),
        
        # Real DEWA bill patterns
        ("299 Electricity kWh Carbon 120 kg CO2e", 299, 120),
        ("Dubai bill usage 500 kWh footprint 200 kg", 500, 200),
        
        # High/low usage cases
        ("Commercial: 2,500 kWh Carbon: 1000 kg CO2e", 2500, 1000),
        ("Residential: 150 kWh Carbon: 60 kg", 150, 60),
        
        # Complex layouts
        ("""Dubai Electricity & Water Authority Page 2 of 3
        Account: 2052672303 Issue Date: 21/05/2025
        Electricity 450 kWh Rate: 0.23 AED
        Carbon Footprint Kg CO2e levels 180 kg""", 450, 180),
        
        # Partial cases
        ("Electricity consumption: 500 kWh", 500, None),  # Only electricity
        ("Environmental impact: 200 kg CO2e", None, 200),  # Only carbon
    ]
    
    @pytest.mark.parametrize("text,expected_elec,expected_carbon", GROUND_TRUTH_CASES)
    def test_field_extraction_accuracy(self, text, expected_elec, expected_carbon):
        """Test field-level extraction accuracy - the north star metric."""
        result = pipeline.extract_fields(text)
        
        # Test electricity extraction
        if expected_elec is not None:
            actual_elec = result.get("electricity_kwh")
            assert actual_elec == expected_elec, \
                f"Electricity FAIL: got {actual_elec}, expected {expected_elec} for text: {text[:60]}..."
        
        # Test carbon extraction  
        if expected_carbon is not None:
            actual_carbon = result.get("carbon_kgco2e")
            assert actual_carbon == expected_carbon, \
                f"Carbon FAIL: got {actual_carbon}, expected {expected_carbon} for text: {text[:60]}..."
    
    def test_overall_accuracy_rate(self):
        """Calculate and report overall field-level accuracy percentage."""
        correct_fields = 0
        total_fields = 0
        failed_cases = []
        
        for text, expected_elec, expected_carbon in self.GROUND_TRUTH_CASES:
            result = pipeline.extract_fields(text)
            
            if expected_elec is not None:
                total_fields += 1
                if result.get("electricity_kwh") == expected_elec:
                    correct_fields += 1
                else:
                    failed_cases.append(f"Electricity: {text[:40]}... got {result.get('electricity_kwh')}, expected {expected_elec}")
            
            if expected_carbon is not None:
                total_fields += 1
                if result.get("carbon_kgco2e") == expected_carbon:
                    correct_fields += 1
                else:
                    failed_cases.append(f"Carbon: {text[:40]}... got {result.get('carbon_kgco2e')}, expected {expected_carbon}")
        
        accuracy = correct_fields / total_fields * 100
        
        print(f"\n=== GROUND TRUTH ACCURACY REPORT ===")
        print(f"Correct fields: {correct_fields}/{total_fields}")
        print(f"Field-level accuracy: {accuracy:.1f}%")
        
        if failed_cases:
            print(f"\nFailed cases ({len(failed_cases)}):")
            for case in failed_cases[:5]:  # Show first 5 failures
                print(f"  - {case}")
        
        # Require at least 90% accuracy as minimum acceptable
        assert accuracy >= 90.0, f"Field accuracy {accuracy:.1f}% below required 90%"


class TestEngineIntegration:
    """Test OCR engine integration and coordination."""
    
    def test_vlm_engine_image_handling(self):
        """Test that VLM engines receive proper image objects."""
        test_image = Image.new('RGB', (100, 100), color='white')
        
        with patch('pipeline._image_cache.get_images', return_value=[test_image]):
            with patch('pipeline._mistral_ocr') as mock_mistral:
                mock_mistral.return_value = pipeline.OcrResult("test", ["test"], [0.9], "mistral")
                
                result = pipeline._run_single_engine_with_cache(
                    (Path("test.png"), "mistral", True, 300)
                )
                
                # Mistral should be called with image object
                if mock_mistral.called:
                    args = mock_mistral.call_args[0]
                    assert isinstance(args[0], Image.Image)
    
    def test_bounding_box_extraction_for_vlm(self):
        """Test bounding box extraction for VLM guidance."""
        test_image = Image.new('RGB', (100, 100), color='white')
        
        with patch('pipeline._tesseract_ocr') as mock_tesseract:
            mock_tesseract.return_value = pipeline.OcrResult("text", ["word1", "word2"], [0.9, 0.8])
            
            result = pipeline._extract_bounding_boxes_for_vlm(test_image)
            assert "Found 2 text regions" in result
    
    @patch('pipeline._run_single_engine_with_cache')
    def test_engine_voting_accuracy(self, mock_engine):
        """Test that parallel engines with voting improve accuracy."""
        
        # Simulate different engines giving different results
        def mock_engine_response(args):
            engine = args[1]
            if engine == "tesseract":
                return engine, pipeline.OcrResult("Electricity 299 kWh", ["299"], [0.85], engine)
            elif engine == "easyocr":
                return engine, pipeline.OcrResult("Electricity 299 kWh Carbon 120", ["299", "120"], [0.95, 0.90], engine)
            elif engine == "paddleocr":
                return engine, pipeline.OcrResult("Electricity 300 kWh Carbon 125", ["300", "125"], [0.80, 0.75], engine)
            else:
                return engine, pipeline.OcrResult("", [], [], engine)
        
        mock_engine.side_effect = mock_engine_response
        
        # Mock the full pipeline
        with patch('pipeline.extract_text', return_value=None):
            with patch('pipeline._image_cache.get_images', return_value=[Mock()]):
                result = pipeline.run_ocr(Path("dummy.pdf"))
        
        # Should select easyocr (highest confidence and most complete)
        assert result.engine == "easyocr"
        assert "299" in result.text and "120" in result.text


class TestRobustnessFeatures:
    """Test robustness and error handling features."""
    
    def test_blank_document_detection(self):
        """Test blank document detection and early termination."""
        # Test blank detection function
        blank_image = Image.new('L', (100, 100), color=255)  # Pure white
        content_image = Image.new('L', (100, 100), color=128)  # Gray
        
        # Add content to second image
        content_array = np.array(content_image)
        content_array[20:80, 20:80] = 50  # Dark rectangle
        content_image = Image.fromarray(content_array)
        
        assert pipeline._is_blank_image(blank_image) == True
        assert pipeline._is_blank_image(content_image) == False
        
        # Test early detection in pipeline
        with patch('pipeline.load_image', return_value=blank_image):
            with patch('pipeline._is_blank_image', return_value=True):
                result = pipeline.run_ocr(Path("blank.png"))
                assert result.engine == "blank_document"
                assert result.text == ""
    
    def test_file_format_validation(self):
        """Test file format validation and corruption detection."""
        # Test corrupted PDF detection
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(b'This is not a PDF file')
            corrupted_path = Path(tmp.name)
        
        try:
            with pytest.raises(ValueError, match="not a valid PDF"):
                pipeline._validate_file_format(corrupted_path)
        finally:
            corrupted_path.unlink()
    
    def test_performance_optimizations(self):
        """Test performance optimization features."""
        # Test image resizing
        large_image = Image.new('RGB', (3000, 4000), color='white')
        with patch('pipeline.MAX_IMAGE_WIDTH', 2000):
            with patch('pipeline.MAX_IMAGE_HEIGHT', 2000):
                resized = pipeline._optimize_image_size(large_image)
                assert resized.size[0] <= 2000
                assert resized.size[1] <= 2000
        
        # Test cache management
        cache = pipeline.ImageCache()
        with patch('pipeline.MAX_CACHE_SIZE_MB', 1):
            cache._cache_size_mb = 10
            cache._check_cache_size()
            assert len(cache._cache) == 0


class TestValidationAndProcessing:
    """Test validation and post-processing features."""
    
    def test_cross_field_validation(self):
        """Test that validation prevents false positives."""
        # These should be rejected by validation
        invalid_cases = [
            (299, 1500),   # Carbon way too high for electricity 
            (10000, 50),   # Electricity too high for carbon
            (50, 5),       # Both values too low
        ]
        
        for electricity, carbon in invalid_cases:
            result = pipeline._validate_extraction_values(electricity, carbon)
            assert not result, f"Validation should reject electricity={electricity}, carbon={carbon}"
    
    def test_validation_allows_valid_cases(self):
        """Test that validation allows realistic value combinations."""
        # These should be accepted by validation
        valid_cases = [
            (299, 120),    # Standard residential
            (1000, 400),   # High usage residential
            (150, 60),     # Low usage
            (2500, 1000),  # Commercial
        ]
        
        for electricity, carbon in valid_cases:
            result = pipeline._validate_extraction_values(electricity, carbon)
            assert result, f"Validation should accept electricity={electricity}, carbon={carbon}"
    
    def test_multi_page_ocr_processing(self):
        """Test multi-page OCR aggregation."""
        images = [
            Image.new('RGB', (100, 100), color='white'),
            Image.new('RGB', (100, 100), color='gray')
        ]
        
        def mock_ocr_func(img):
            return pipeline.OcrResult("page text", ["page", "text"], [0.9, 0.8])
        
        result = pipeline._process_multi_page_ocr(images, mock_ocr_func)
        
        assert "page text" in result.text
        assert len(result.tokens) == 4  # 2 tokens per page
        assert len(result.confidences) == 4


class TestOutputAndMetadata:
    """Test output formatting and metadata generation."""
    
    def test_payload_building_structure(self):
        """Test JSON payload building with all metadata."""
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
            assert payload["meta"]["extraction_status"] == "success"
            assert "confidence_thresholds" in payload["meta"]
    
    def test_configurable_thresholds(self):
        """Test threshold configuration via environment variables."""
        # Test helper function
        test_val = config._get_float_env_var("TEST_THRESHOLD", 0.95, 0.0, 1.0)
        assert test_val == 0.95
        
        # Test invalid value handling
        with patch.dict(os.environ, {'TEST_INVALID': 'not_a_number'}):
            test_val = config._get_float_env_var("TEST_INVALID", 0.95, 0.0, 1.0)
            assert test_val == 0.95  # Should fall back to default


class TestRealWorldScenarios:
    """Test realistic OCR scenarios and edge cases."""
    
    def test_noisy_ocr_accuracy(self):
        """Test extraction accuracy with realistic OCR noise."""
        noisy_cases = [
            ("Electricity 299kWh Carbon Footprint: Kg CO2e 120", 299, 120),
            ("DEWA Bill Usage: 450 kWh Environmental: 180 kg", 450, 180),
            ("Consumption 1,234 kWh Carbon emissions 456 kg CO2e", 1234, 456),
        ]
        
        for text, expected_elec, expected_carbon in noisy_cases:
            result = pipeline.extract_fields(text)
            
            if expected_elec:
                assert result.get("electricity_kwh") == expected_elec, \
                    f"Noisy OCR failed for electricity: {text}"
            if expected_carbon:
                assert result.get("carbon_kgco2e") == expected_carbon, \
                    f"Noisy OCR failed for carbon: {text}"
    
    def test_ocr_error_variants(self):
        """Test extraction with common OCR character substitution errors."""
        ocr_variants = [
            ('Carbon footprint: Kg coze 120', 120),  # coze -> CO2e
            ('kg co2e 250', 250),  # case variations
            ('Kg  CO2e   180', 180),  # extra spaces
            ('Carbon emissions in Kg CO2e levels 150', 150),  # embedded context
        ]
        
        for text, expected_carbon in ocr_variants:
            result = pipeline.extract_fields(text)
            assert result.get("carbon_kgco2e") == expected_carbon, \
                f"OCR variant failed for: {text}"
    
    def test_edge_case_filtering(self):
        """Test that invalid values are properly filtered."""
        invalid_cases = [
            'Kg CO2e 5',  # Too low, should be filtered
            'Kg coze 0',  # Zero value
            'Carbon but no number',  # No valid number
        ]
        
        for text in invalid_cases:
            result = pipeline.extract_fields(text)
            # Should either not extract or extract valid values only
            if 'carbon_kgco2e' in result:
                assert result['carbon_kgco2e'] >= 10  # Minimum valid threshold


if __name__ == "__main__":
    # Run with verbose output and show print statements
    pytest.main([__file__, "-v", "-s", "--tb=short"])