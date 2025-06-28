"""
Comprehensive ground truth accuracy testing across OCR engines and extraction methods.
This is the north star test - field and value level accuracy is the key metric.
"""
import pytest
import sys
import pathlib
from unittest.mock import Mock, patch

# Add repository root to path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import pipeline


class TestGroundTruthAccuracy:
    """Core ground truth accuracy test - this is what matters most."""
    
    # Definitive ground truth test cases covering all scenarios
    GROUND_TRUTH_CASES = [
        # Basic clean cases
        ("Dubai Electricity Water Authority Invoice Electricity 299 kWh Carbon Footprint Kg CO2e 120", 299, 120),
        ("Consumption: 299 kWh Carbon emissions: 120 kg CO2e", 299, 120),
        ("Electricity usage 1,234 kWh Environmental impact 456 kg CO2e", 1234, 456),
        
        # Real DEWA bill patterns
        ("Carbon Footprint 299kWh 0.230 68.77 0 kWh 0.000 0.00 Kg C0Ze kwh 000 0.00 120 kWh 000 0.00", 299, 120),
        ("299 Electricity kWh Carbon 120 kg CO2e", 299, 120),
        
        # OCR noise cases (these should work with KIE fallback)
        ("Electricity 299 kWh Carbon 120 kg CO2e", 299, 120),  # Clean baseline
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
            for case in failed_cases:
                print(f"  - {case}")
        
        # Require at least 90% accuracy as minimum acceptable
        assert accuracy >= 90.0, f"Field accuracy {accuracy:.1f}% below required 90%"


class TestEngineParallelization:
    """Test parallel OCR engine processing and voting accuracy."""
    
    @patch('pipeline._run_single_engine_with_cache')
    def test_engine_voting_accuracy(self, mock_engine):
        """Test that parallel engines with voting improve accuracy."""
        
        # Simulate different engines giving different results
        def mock_engine_response(args):
            engine = args[1]
            if engine == "tesseract":
                # Tesseract gets electricity right but misses carbon
                return engine, pipeline.OcrResult("Electricity 299 kWh", ["299"], [0.85], engine)
            elif engine == "easyocr":
                # EasyOCR gets both fields right with high confidence
                return engine, pipeline.OcrResult("Electricity 299 kWh Carbon 120", ["299", "120"], [0.95, 0.90], engine)
            elif engine == "paddleocr":
                # PaddleOCR gets numbers slightly wrong
                return engine, pipeline.OcrResult("Electricity 300 kWh Carbon 125", ["300", "125"], [0.80, 0.75], engine)
            else:
                return engine, pipeline.OcrResult("", [], [], engine)
        
        mock_engine.side_effect = mock_engine_response
        
        # Mock the full pipeline
        with patch('pipeline.extract_text', return_value=None):
            with patch('pipeline._image_cache.get_images', return_value=[Mock()]):
                result = pipeline.run_ocr(pathlib.Path("dummy.pdf"))
        
        # Should select easyocr (highest confidence and most complete)
        assert result.engine == "easyocr"
        assert "299" in result.text and "120" in result.text
    
    @patch('pipeline._run_single_engine_with_cache')
    def test_parallel_vs_sequential_selection(self, mock_engine):
        """Test that parallel processing selects best engine result."""
        
        # Setup engines with different accuracy profiles
        def mock_varied_results(args):
            engine = args[1]
            confidence_map = {
                "tesseract": (pipeline.OcrResult("Text with errors", ["text"], [0.70], engine), 0.70),
                "easyocr": (pipeline.OcrResult("Clean accurate text", ["clean"], [0.95], engine), 0.95),
                "paddleocr": (pipeline.OcrResult("Moderate quality", ["moderate"], [0.80], engine), 0.80),
            }
            result, conf = confidence_map.get(engine, (pipeline.OcrResult("", [], [], engine), 0.0))
            return engine, result
        
        mock_engine.side_effect = mock_varied_results
        
        with patch('pipeline.extract_text', return_value=None):
            with patch('pipeline._image_cache.get_images', return_value=[Mock()]):
                result = pipeline.run_ocr(pathlib.Path("dummy.pdf"))
        
        # Should select the highest confidence engine
        assert result.engine == "easyocr"
        assert result.field_confidence == 0.95


class TestValidationAccuracy:
    """Test that validation prevents false positives and improves accuracy."""
    
    def test_cross_field_validation_prevents_errors(self):
        """Test that validation catches impossible value combinations."""
        
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


class TestRealWorldAccuracy:
    """Test accuracy on realistic OCR scenarios with noise."""
    
    def test_noisy_ocr_accuracy(self):
        """Test extraction accuracy with realistic OCR noise."""
        
        # Real-world OCR noise scenarios
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
    
    def test_ocr_variant_accuracy(self):
        """Test extraction of OCR error variants."""
        
        ocr_variants = [
            # OCR character substitution errors
            ('Carbon footprint: Kg coze 120', 120),  # coze -> CO2e
            ('kg co2e 250', 250),  # case variations
            ('Kg  CO2e   180', 180),  # extra spaces
            ('Carbon emissions in Kg CO2e levels 150', 150),  # embedded context
            
            # Complex DEWA patterns
            ('Kg C0Ze kwh 000 0.00 120 kWh 000 0.00', 120),  # C0Ze variant
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
            'Electricity: 299 kWh but no carbon data',  # No carbon mention
        ]
        
        for text in invalid_cases:
            result = pipeline.extract_fields(text)
            assert 'carbon_kgco2e' not in result, f"Should filter invalid case: {text}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to show print statements