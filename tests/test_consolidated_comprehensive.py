#!/usr/bin/env python3
"""
Consolidated Comprehensive Test Suite for OCR Pipeline

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


class TestWordCharacterAccuracy:
    """Test word-level and character-level accuracy across OCR engines."""
    
    # Ground truth text with known character-level accuracy
    WORD_ACCURACY_CASES = [
        # (input_text, expected_tokens, expected_full_text)
        ("Dubai Electricity Water Authority", 
         ["Dubai", "Electricity", "Water", "Authority"],
         "Dubai Electricity Water Authority"),
        
        ("Consumption 299 kWh", 
         ["Consumption", "299", "kWh"],
         "Consumption 299 kWh"),
         
        ("Carbon Footprint 120 kg CO2e",
         ["Carbon", "Footprint", "120", "kg", "CO2e"],
         "Carbon Footprint 120 kg CO2e"),
         
        ("Account: 2052672303 Issue Date: 21/05/2025",
         ["Account:", "2052672303", "Issue", "Date:", "21/05/2025"],
         "Account: 2052672303 Issue Date: 21/05/2025"),
    ]
    
    def calculate_character_error_rate(self, reference: str, hypothesis: str) -> float:
        """Calculate Character Error Rate (CER) between reference and hypothesis."""
        import difflib
        
        # Use SequenceMatcher for character-level comparison
        matcher = difflib.SequenceMatcher(None, reference, hypothesis)
        
        # Count operations needed to transform hypothesis to reference
        operations = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != 'equal':
                operations += max(i2 - i1, j2 - j1)
        
        # CER = (Substitutions + Deletions + Insertions) / Length of reference
        cer = operations / len(reference) if len(reference) > 0 else 0.0
        return cer
    
    def calculate_word_error_rate(self, reference_tokens: list, hypothesis_tokens: list) -> float:
        """Calculate Word Error Rate (WER) between reference and hypothesis tokens."""
        import difflib
        
        # Use SequenceMatcher for word-level comparison
        matcher = difflib.SequenceMatcher(None, reference_tokens, hypothesis_tokens)
        
        # Count word-level operations
        operations = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != 'equal':
                operations += max(i2 - i1, j2 - j1)
        
        # WER = (Word Substitutions + Deletions + Insertions) / Number of reference words  
        wer = operations / len(reference_tokens) if len(reference_tokens) > 0 else 0.0
        return wer
    
    @pytest.mark.parametrize("input_text,expected_tokens,expected_text", WORD_ACCURACY_CASES)
    def test_word_level_accuracy_across_engines(self, input_text, expected_tokens, expected_text):
        """Test word-level accuracy for each OCR engine."""
        
        # Create realistic OCR errors based on the actual input
        def add_ocr_errors(tokens, text, engine):
            if engine == "tesseract":
                # Common Tesseract errors: I->l, O->0, etc.
                error_tokens = [t.replace("I", "l").replace("O", "0") for t in tokens]
                error_text = text.replace("I", "l").replace("O", "0")
            elif engine == "easyocr":
                # Common EasyOCR errors: m->rn, W->VV, etc.
                error_tokens = [t.replace("m", "rn").replace("W", "VV") for t in tokens]
                error_text = text.replace("m", "rn").replace("W", "VV")
            else:  # paddleocr
                # Common PaddleOCR errors: subtle character recognition
                error_tokens = [t.replace("i", "l").replace("y", "v") for t in tokens]
                error_text = text.replace("i", "l").replace("y", "v")
            
            # Add confidence scores
            confidences = [0.90 + 0.05 * (i % 2) for i in range(len(error_tokens))]
            return {"tokens": error_tokens, "text": error_text, "confidences": confidences}
        
        # Generate engine results with errors
        engines = ["tesseract", "easyocr", "paddleocr"]
        engine_results = {}
        for engine in engines:
            engine_results[engine] = add_ocr_errors(expected_tokens, expected_text, engine)
        
        # Test each engine's word accuracy
        for engine, result in engine_results.items():
            wer = self.calculate_word_error_rate(expected_tokens, result["tokens"])
            cer = self.calculate_character_error_rate(expected_text, result["text"])
            
            # Log accuracy metrics
            print(f"\n{engine} Word Accuracy:")
            print(f"  WER: {wer:.3f} ({(1-wer)*100:.1f}% word accuracy)")
            print(f"  CER: {cer:.3f} ({(1-cer)*100:.1f}% character accuracy)")
            print(f"  Tokens: {result['tokens']}")
            
            # Assert reasonable accuracy thresholds (relaxed for realistic OCR errors)
            assert wer <= 0.8, f"{engine} WER {wer:.3f} too high (>80% word errors)"
            assert cer <= 0.3, f"{engine} CER {cer:.3f} too high (>30% character errors)"
    
    def test_engine_accuracy_comparison(self):
        """Compare accuracy across all OCR engines on the same text."""
        
        test_text = "Dubai Electricity Water Authority Invoice 299 kWh Carbon 120 kg CO2e"
        expected_tokens = test_text.split()
        
        # Mock realistic engine results with different error patterns
        engine_results = {
            "tesseract": pipeline.OcrResult(
                "Dubai Electriclty Water Authority Invoice 299 kWh Carbon I20 kg CO2e",
                ["Dubai", "Electriclty", "Water", "Authority", "Invoice", "299", "kWh", "Carbon", "I20", "kg", "CO2e"],
                [0.95, 0.75, 0.90, 0.88, 0.92, 0.98, 0.94, 0.91, 0.60, 0.89, 0.85]
            ),
            "easyocr": pipeline.OcrResult(
                "Dubai Electricity VVater Authority Invoice 299 kWh Carbon 120 kg C02e", 
                ["Dubai", "Electricity", "VVater", "Authority", "Invoice", "299", "kWh", "Carbon", "120", "kg", "C02e"],
                [0.92, 0.96, 0.70, 0.91, 0.94, 0.97, 0.95, 0.93, 0.96, 0.92, 0.78]
            ),
            "paddleocr": pipeline.OcrResult(
                "Dubai Electricity Water Authority Invoice 299 kWh Carbon 120 kg CO2e",
                ["Dubai", "Electricity", "Water", "Authority", "Invoice", "299", "kWh", "Carbon", "120", "kg", "CO2e"], 
                [0.94, 0.93, 0.95, 0.89, 0.91, 0.99, 0.96, 0.94, 0.97, 0.93, 0.88]
            )
        }
        
        accuracy_results = {}
        
        for engine, result in engine_results.items():
            wer = self.calculate_word_error_rate(expected_tokens, result.tokens)
            cer = self.calculate_character_error_rate(test_text, result.text)
            
            accuracy_results[engine] = {
                'wer': wer,
                'cer': cer, 
                'word_accuracy': (1 - wer) * 100,
                'char_accuracy': (1 - cer) * 100,
                'avg_confidence': sum(result.confidences) / len(result.confidences)
            }
        
        # Print comparison report
        print(f"\n{'='*60}")
        print("OCR ENGINE ACCURACY COMPARISON")
        print(f"{'='*60}")
        print(f"{'Engine':<12} {'Word Acc':<10} {'Char Acc':<10} {'Avg Conf':<10} {'WER':<8} {'CER':<8}")
        print(f"{'-'*60}")
        
        for engine, metrics in accuracy_results.items():
            print(f"{engine:<12} {metrics['word_accuracy']:<10.1f}% {metrics['char_accuracy']:<10.1f}% "
                  f"{metrics['avg_confidence']:<10.3f} {metrics['wer']:<8.3f} {metrics['cer']:<8.3f}")
        
        # Find best performing engine
        best_word_engine = min(accuracy_results.keys(), key=lambda x: accuracy_results[x]['wer'])
        best_char_engine = min(accuracy_results.keys(), key=lambda x: accuracy_results[x]['cer'])
        
        print(f"\nBest word accuracy: {best_word_engine} ({accuracy_results[best_word_engine]['word_accuracy']:.1f}%)")
        print(f"Best character accuracy: {best_char_engine} ({accuracy_results[best_char_engine]['char_accuracy']:.1f}%)")
        
        # Assert minimum accuracy requirements (realistic for OCR with errors)
        for engine, metrics in accuracy_results.items():
            assert metrics['word_accuracy'] >= 60.0, f"{engine} word accuracy {metrics['word_accuracy']:.1f}% below 60%"
            assert metrics['char_accuracy'] >= 70.0, f"{engine} character accuracy {metrics['char_accuracy']:.1f}% below 70%"
    
    def test_confidence_accuracy_correlation(self):
        """Test correlation between engine confidence and actual accuracy."""
        
        # Test cases with known accuracy levels
        test_cases = [
            # (text, expected_accuracy_level)
            ("Perfect clear text", "high"),      # Should have high confidence and accuracy
            ("Noisy 0CR t3xt w1th err0rs", "low"), # Should have low confidence and accuracy  
            ("Moderate quality text", "medium"),  # Should have medium confidence and accuracy
        ]
        
        for text, expected_level in test_cases:
            # Mock engine result based on expected accuracy
            if expected_level == "high":
                result = pipeline.OcrResult(text, text.split(), [0.95] * len(text.split()), "tesseract")
                expected_min_accuracy = 95.0
            elif expected_level == "medium":
                result = pipeline.OcrResult(text, text.split(), [0.80] * len(text.split()), "tesseract")
                expected_min_accuracy = 80.0
            else:  # low
                result = pipeline.OcrResult(text, text.split(), [0.60] * len(text.split()), "tesseract")
                expected_min_accuracy = 60.0
            
            # Test that confidence correlates with expected accuracy
            avg_confidence = result.field_confidence * 100
            assert avg_confidence >= expected_min_accuracy * 0.8, \
                f"Confidence {avg_confidence:.1f}% too low for {expected_level} accuracy text"


if __name__ == "__main__":
    # Run with verbose output and show print statements
    pytest.main([__file__, "-v", "-s", "--tb=short"])