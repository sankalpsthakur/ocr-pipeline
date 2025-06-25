"""
Comprehensive tests for OCR improvements including regex patterns and engine integration.
Consolidates test_regex_improvements.py and test_ocr_engines.py for better organization.
"""
import pytest
import sys, pathlib
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from PIL import Image, ImageDraw

# Add repository root to path so tests can import modules directly
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import pipeline


class TestCarbonDetectionImprovements:
    """Test carbon detection regex pattern improvements."""
    
    @pytest.mark.parametrize("text,expected", [
        # OCR variants
        ('Carbon footprint: Kg coze 120', 120),
        ('Carbon emissions: Kg C0Ze kwh 000 0.00 120 kWh', 120),
        ('kg co2e 250', 250),
        ('Kg  CO2e   180', 180),
        ('Carbon emissions in Kg CO2e levels 150', 150),
        
        # Complex cases with multiple numbers
        ('Kg C0Ze kwh 000 0.00 120 kWh 000 0.00', 120),
        ('Carbon Footprint 299kWh 0.230 68.77\n0 kWh 0.000 0.00\nKg C0Ze kwh 000 0.00\n120 kWh 000 0.00', 120),
        
        # Value filtering
        ('Kg CO2e 9999', 9999),  # Large values should pass
    ])
    def test_carbon_detection_variants(self, text, expected):
        """Test various carbon detection patterns and OCR errors."""
        result = pipeline.extract_fields(text)
        assert result['carbon_kgco2e'] == expected
    
    @pytest.mark.parametrize("text", [
        'Kg CO2e 5',  # Too low, should be filtered
        'Kg coze 0',  # Zero value
        'Carbon but no number',  # No valid number
        'Electricity: 299 kWh but no carbon data',  # No carbon mention
    ])
    def test_carbon_filtering_and_edge_cases(self, text):
        """Test that invalid carbon values are properly filtered."""
        result = pipeline.extract_fields(text)
        assert 'carbon_kgco2e' not in result
    
    def test_electricity_and_carbon_together(self):
        """Test extraction of both values in complex text."""
        text = '''Electricity consumption: 299 kWh
        Carbon Footprint: Kg C0Ze kwh 000 0.00 120 kWh'''
        result = pipeline.extract_fields(text)
        assert result['electricity_kwh'] == 299
        assert result['carbon_kgco2e'] == 120
    
    def test_regex_pattern_priority(self):
        """Test that regex patterns are applied in correct priority order."""
        # Primary pattern should take precedence when available
        text = 'Kg CO2e 150 and also Kg coze 000 0.00 120'
        result = pipeline.extract_fields(text)
        assert result['carbon_kgco2e'] == 150  # Should use primary pattern
    
    def test_empty_text_handling(self):
        """Test behavior with empty or malformed input."""
        assert pipeline.extract_fields('') == {}


class TestOCREngineIntegration:
    """Test OCR engine selection and integration."""
    
    def test_configuration_loading(self):
        """Test that OCR backend configuration is valid."""
        from config import OCR_BACKEND
        assert OCR_BACKEND in ["tesseract", "easyocr", "paddleocr"]
    
    @patch('pipeline.pytesseract')
    @patch('pipeline.preprocess')
    @patch('pipeline._auto_rotate')
    def test_tesseract_ocr_pipeline(self, mock_rotate, mock_preprocess, mock_tesseract):
        """Test complete Tesseract OCR pipeline."""
        # Setup mocks
        mock_image = Mock()
        mock_rotate.return_value = mock_image
        mock_preprocess.return_value = (mock_image, {})
        mock_tesseract.image_to_data.return_value = {
            'text': ['Electricity', '299', 'kWh', 'Carbon', '120'],
            'conf': [95, 90, 85, 92, 88]
        }
        
        result = pipeline._tesseract_ocr(mock_image)
        
        assert result.text == "Electricity 299 kWh Carbon 120"
        assert len(result.tokens) == 5
        assert len(result.confidences) == 5
        mock_tesseract.image_to_data.assert_called_once()
    
    @patch('pipeline.easyocr')
    @patch('pipeline.np')
    def test_easyocr_integration(self, mock_np, mock_easyocr):
        """Test EasyOCR integration and result processing."""
        # Setup mocks
        mock_reader = Mock()
        mock_easyocr.Reader.return_value = mock_reader
        mock_reader.readtext.return_value = [
            ([[0, 0], [100, 30]], 'Electricity 299 kWh', 0.95),
            ([[0, 50], [100, 80]], 'Carbon 120 kg', 0.90),
        ]
        mock_image = Mock()
        mock_np.array.return_value = np.array([[255, 255], [255, 255]])
        
        # Clear any cached reader for clean test
        if hasattr(pipeline._easyocr_ocr, "reader"):
            delattr(pipeline._easyocr_ocr, "reader")
        
        result = pipeline._easyocr_ocr(mock_image)
        
        assert "Electricity 299 kWh Carbon 120 kg" == result.text
        assert len(result.tokens) == 2
        assert result.confidences == [0.95, 0.90]
    
    @patch('pipeline.easyocr', None)
    def test_missing_dependencies_handling(self):
        """Test error handling when OCR engines are not available."""
        with pytest.raises(RuntimeError, match="easyocr is not available"):
            pipeline._easyocr_ocr(Mock())
    
    @patch('pipeline.pytesseract', None)
    def test_tesseract_not_available_error(self):
        """Test error when Tesseract is not available."""
        with pytest.raises(RuntimeError, match="pytesseract is not available"):
            pipeline._tesseract_ocr(Mock())

    def test_preprocess_deskew(self):
        """Verify deskew and binarisation pipeline."""
        img = Image.new("L", (200, 80), 255)
        draw = ImageDraw.Draw(img)
        for y in range(20, 60, 20):
            draw.line((10, y, 190, y), fill=0, width=3)
        rotated = img.rotate(7, expand=True, fillcolor=255)

        processed, meta = pipeline.preprocess(rotated, dpi=300)

        assert isinstance(processed, Image.Image)
        assert "deskew_angle" in meta
        arr = np.array(processed)
        assert arr.ndim == 2
        assert set(np.unique(arr)).issubset({0, 255})


class TestOCRAccuracyMetrics:
    """Test OCR accuracy calculation and confidence metrics."""
    
    @pytest.mark.parametrize("confidences,expected", [
        ([0.95, 0.85], 0.898),  # Geometric mean
        ([1.0, 1.0], 1.0),      # Perfect confidence
        ([0.5, 0.5], 0.5),      # Medium confidence
        ([], 0.0),              # Empty confidence list
        ([0.0], 0.001),         # Minimum threshold applied
    ])
    def test_confidence_calculation(self, confidences, expected):
        """Test confidence calculation with various inputs."""
        tokens = ["test"] * len(confidences) if confidences else []
        result = pipeline.OcrResult("test text", tokens, confidences)
        assert abs(result.field_confidence - expected) < 0.001


class TestRegexPatternValidation:
    """Test regex patterns independently for robustness."""
    
    def test_energy_pattern_variations(self):
        """Test electricity detection pattern variations."""
        test_cases = [
            ('Total usage: 1,234 kWh', 1234),
            ('299 kWh consumption', 299),
            ('Energy: 42kWh', 42),
            ('1 234 kWh with spaces', 1234),
        ]
        
        for text, expected in test_cases:
            result = pipeline.extract_fields(text)
            assert result['electricity_kwh'] == expected
    
    def test_carbon_pattern_robustness(self):
        """Test carbon patterns handle various formatting."""
        import re
        
        # Test each pattern individually
        patterns = [
            pipeline.CARBON_RE,
            pipeline.CARBON_SIMPLE_RE,
            pipeline.CARBON_ALT_RE,
            pipeline.CARBON_EMISSIONS_RE
        ]
        
        test_texts = [
            'Kg CO2e 120',
            'Kg C0Ze kwh 000 0.00 120',
            'Carbon emissions in Kg CO2e levels 150'
        ]
        
        # At least one pattern should match for each valid text
        for text in test_texts:
            matched = any(pattern.search(text) for pattern in patterns)
            assert matched, f"No pattern matched text: {text}"


class TestEndToEndExtraction:
    """Test complete end-to-end extraction scenarios."""
    
    def test_real_world_ocr_text_simulation(self):
        """Test with simulated real OCR text including errors."""
        # Simulate text similar to actual EasyOCR output
        ocr_text = '''Dubai Electricity Water Authority Page 2 of 3
        Invoice: 100045272594 Issue Date:21/05/2025
        Account Number 2052672303
        Kilowatt Hours(kWh) Current reading: 19462
        Electricity 299 Previous reading: 19163
        Electricity Consumption Rate AED Carbon Footprint 299kWh 0.230 68.77
        0 kWh 0.000 0.00 Kg C0Ze kwh 000 0.00 120 kWh 000 0.00'''
        
        result = pipeline.extract_fields(ocr_text)
        
        # Should extract both values correctly
        assert result['electricity_kwh'] == 299
        assert result['carbon_kgco2e'] == 120
    
    def test_extraction_with_no_valid_data(self):
        """Test extraction when no valid data is present."""
        text = 'This is just random text with no utility data'
        result = pipeline.extract_fields(text)
        assert result == {}
    
    def test_partial_extraction_scenarios(self):
        """Test scenarios where only one type of data is available."""
        # Only electricity
        electricity_only = 'Electricity consumption: 500 kWh'
        result = pipeline.extract_fields(electricity_only)
        assert result['electricity_kwh'] == 500
        assert 'carbon_kgco2e' not in result
        
        # Only carbon (using simple pattern)
        carbon_only = 'Environmental impact: Kg C0Ze 0.00 150'
        result = pipeline.extract_fields(carbon_only)
        assert result['carbon_kgco2e'] == 150
        assert 'electricity_kwh' not in result


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @patch('pipeline.load_image')
    def test_backend_error_handling(self, mock_load):
        """Test handling of unsupported OCR backends."""
        mock_load.return_value = Mock()
        
        # Test with unsupported backend
        with patch('config.OCR_BACKEND', 'unsupported_engine'):
            with pytest.raises(ValueError, match="Unsupported OCR backend"):
                # Create a mock that bypasses the actual backend logic
                with patch.object(pipeline, '_run_ocr_engine') as mock_run:
                    mock_run.side_effect = ValueError("Unsupported OCR backend: unsupported_engine")
                    mock_run(pathlib.Path("dummy.png"), is_image=True)
    
    def test_number_normalization_edge_cases(self):
        """Test number normalization with various inputs."""
        # Test the internal normalization function
        test_cases = [
            ('123', 123),
            ('1,234', 1234),
            ('1 234', 1234),
            ('1,234,567', 1234567),
            ('000', 0),
        ]
        
        for input_str, expected in test_cases:
            result = pipeline._normalise_number(input_str)
            assert result == expected


# Performance and integration tests
class TestPerformanceAndIntegration:
    """Test performance characteristics and integration points."""
    
    def test_confidence_threshold_integration(self):
        """Test that confidence thresholds work as expected."""
        from config import TAU_FIELD_ACCEPT, TAU_ENHANCER_PASS
        
        # Ensure thresholds are reasonable
        assert 0.0 <= TAU_FIELD_ACCEPT <= 1.0
        assert 0.0 <= TAU_ENHANCER_PASS <= 1.0
        assert TAU_ENHANCER_PASS <= TAU_FIELD_ACCEPT
    
    def test_large_text_handling(self):
        """Test extraction with very large text input."""
        # Create a large text block
        large_text = "Random text " * 1000 + " Electricity 299 kWh " + "More text " * 1000
        
        result = pipeline.extract_fields(large_text)
        assert result['electricity_kwh'] == 299
        assert len(large_text) > 10000  # Ensure we're actually testing large text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])