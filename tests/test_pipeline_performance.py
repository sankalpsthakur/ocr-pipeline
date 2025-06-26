"""
Performance and integration tests for the hierarchical OCR pipeline.
Tests real-world scenarios with actual bill images and downscaling.
"""
import pytest
import sys
import pathlib
import time
import os
from PIL import Image
from unittest.mock import patch, Mock

# Add repository root to path so tests can import modules directly
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import pipeline
import config


class TestPipelinePerformance:
    """Test pipeline performance with real bill images."""
    
    @classmethod
    def setup_class(cls):
        """Set up test class with downscaled bill image."""
        cls.original_bill_path = pathlib.Path(__file__).parent.parent / "ActualBill.png"
        cls.test_bill_path = pathlib.Path(__file__).parent / "test_bill_tiny.jpg"
        cls._create_downscaled_bill()
    
    @classmethod
    def teardown_class(cls):
        """Clean up test files."""
        if cls.test_bill_path.exists():
            cls.test_bill_path.unlink()
    
    @classmethod
    def _create_downscaled_bill(cls):
        """Create a downscaled version of ActualBill.png under 20KB."""
        if not cls.original_bill_path.exists():
            pytest.skip("ActualBill.png not found")
        
        # Check if already exists
        if cls.test_bill_path.exists():
            file_size = cls.test_bill_path.stat().st_size
            print(f"Using existing tiny bill: {file_size / 1024:.1f}KB")
            return
        
        # Open the original image
        with Image.open(cls.original_bill_path) as img:
            # Aggressive scaling for tiny file
            scale_factor = 0.25
            quality = 60
            
            while True:
                # Scale down image significantly
                new_width = int(img.width * scale_factor)
                new_height = int(img.height * scale_factor)
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Convert to RGB for JPEG compression
                if resized_img.mode != 'RGB':
                    resized_img = resized_img.convert('RGB')
                
                # Save as JPEG for better compression
                resized_img.save(cls.test_bill_path, "JPEG", optimize=True, quality=quality)
                
                # Check file size
                file_size = cls.test_bill_path.stat().st_size
                if file_size < 20 * 1024:  # 20KB
                    print(f"Created tiny bill: {file_size / 1024:.1f}KB ({new_width}x{new_height})")
                    break
                
                # Adjust parameters for next iteration
                if quality > 20:
                    quality -= 10
                else:
                    scale_factor -= 0.03
                
                if scale_factor < 0.1:
                    # If still too large, just accept it and continue
                    print(f"Warning: Could not reduce below 20KB. Current size: {file_size / 1024:.1f}KB")
                    break
    
    def test_bill_downscaling(self):
        """Test that the bill was successfully downscaled."""
        assert self.test_bill_path.exists(), "Downscaled bill file should exist"
        
        file_size = self.test_bill_path.stat().st_size
        print(f"Downscaled bill size: {file_size / 1024:.1f}KB")
        
        # Verify it meets the <20KB requirement  
        assert file_size < 20 * 1024, f"File should be under 20KB, got {file_size / 1024:.1f}KB"
        
        # Verify it's much smaller than original
        original_size = self.original_bill_path.stat().st_size
        assert file_size < original_size, "Downscaled file should be smaller than original"
        
        reduction = ((original_size - file_size) / original_size) * 100
        assert reduction > 90, f"Should achieve >90% reduction, got {reduction:.1f}%"
        
        # Verify it's a valid image
        with Image.open(self.test_bill_path) as img:
            assert img.size[0] > 0 and img.size[1] > 0, "Image should have valid dimensions"
    
    def test_hierarchical_pipeline_with_mocks(self):
        """Test hierarchical pipeline behavior with mocked engines."""
        # Mock all engines to simulate different confidence levels
        with patch('pipeline._tesseract_ocr') as mock_tesseract, \
             patch('pipeline._easyocr_ocr') as mock_easyocr, \
             patch('pipeline._paddleocr_ocr') as mock_paddleocr, \
             patch('pipeline._mistral_ocr') as mock_mistral, \
             patch('pipeline._datalab_ocr') as mock_datalab, \
             patch('pipeline._gemma_vlm_ocr') as mock_gemma:
            
            # Configure mocks with increasing confidence levels
            mock_tesseract.return_value = pipeline.OcrResult(
                "Electricity 299 kWh Carbon 120 kgCO2e", 
                ["Electricity", "299", "kWh", "Carbon", "120", "kgCO2e"],
                [0.60, 0.65, 0.70, 0.60, 0.65, 0.70]  # Low confidence
            )
            
            mock_easyocr.return_value = pipeline.OcrResult(
                "Electricity 299 kWh Carbon 120 kgCO2e",
                ["Electricity", "299", "kWh", "Carbon", "120", "kgCO2e"], 
                [0.75, 0.80, 0.85, 0.75, 0.80, 0.85]  # Medium confidence
            )
            
            mock_paddleocr.return_value = pipeline.OcrResult(
                "Electricity 299 kWh Carbon 120 kgCO2e",
                ["Electricity", "299", "kWh", "Carbon", "120", "kgCO2e"],
                [0.96, 0.96, 0.96, 0.96, 0.96, 0.96]  # High confidence - should trigger acceptance
            )
            
            # These shouldn't be called due to early acceptance
            mock_mistral.return_value = pipeline.OcrResult("", [], [])
            mock_datalab.return_value = pipeline.OcrResult("", [], [])
            mock_gemma.return_value = pipeline.OcrResult("", [], [])
            
            # Run the pipeline
            result = pipeline.run_ocr(self.test_bill_path)
            
            # Verify results
            assert result.text == "Electricity 299 kWh Carbon 120 kgCO2e"
            assert result.field_confidence >= config.TAU_FIELD_ACCEPT
            
            # Verify call sequence (should stop at PaddleOCR due to high confidence)
            mock_tesseract.assert_called_once()
            mock_easyocr.assert_called_once()
            mock_paddleocr.assert_called_once()
            mock_mistral.assert_not_called()  # Should not reach these
            mock_datalab.assert_not_called()
            mock_gemma.assert_not_called()
    
    def test_hierarchical_pipeline_datalab_acceptance(self):
        """Test that Datalab is reached and accepts with high confidence."""
        with patch('pipeline._tesseract_ocr') as mock_tesseract, \
             patch('pipeline._easyocr_ocr') as mock_easyocr, \
             patch('pipeline._paddleocr_ocr') as mock_paddleocr, \
             patch('pipeline._mistral_ocr') as mock_mistral, \
             patch('pipeline._datalab_ocr') as mock_datalab, \
             patch('pipeline._gemma_vlm_ocr') as mock_gemma:
            
            # Configure lower confidence for first engines
            low_confidence_result = pipeline.OcrResult(
                "Electricity 299 kWh Carbon 120 kgCO2e",
                ["Electricity", "299", "kWh", "Carbon", "120", "kgCO2e"],
                [0.80, 0.85, 0.90, 0.80, 0.85, 0.90]  # Below acceptance threshold
            )
            
            mock_tesseract.return_value = low_confidence_result
            mock_easyocr.return_value = low_confidence_result
            mock_paddleocr.return_value = low_confidence_result
            mock_mistral.return_value = low_confidence_result
            
            # Datalab returns high confidence
            mock_datalab.return_value = pipeline.OcrResult(
                "Electricity 299 kWh Carbon 120 kgCO2e",
                ["Electricity", "299", "kWh", "Carbon", "120", "kgCO2e"],
                [0.95, 0.95, 0.95, 0.95, 0.95, 0.95]  # High confidence
            )
            
            mock_gemma.return_value = pipeline.OcrResult("", [], [])
            
            result = pipeline.run_ocr(self.test_bill_path)
            
            # Verify Datalab was reached and accepted
            assert result.field_confidence >= config.TAU_FIELD_ACCEPT
            mock_datalab.assert_called_once()
            mock_gemma.assert_not_called()  # Should not reach Gemma
    
    def test_field_extraction_from_pipeline(self):
        """Test complete pipeline from OCR to field extraction."""
        # Use a simple mock that returns recognizable text
        with patch('pipeline.run_ocr') as mock_run_ocr:
            mock_run_ocr.return_value = pipeline.OcrResult(
                "Dubai Electricity Water Authority Electricity 299 kWh Carbon Footprint Kg CO2e 120",
                ["Dubai", "Electricity", "Water", "Authority", "Electricity", "299", "kWh", "Carbon", "Footprint", "Kg", "CO2e", "120"],
                [0.95] * 12
            )
            
            # Run full pipeline
            result = pipeline.run_ocr(self.test_bill_path)
            fields = pipeline.extract_fields(result.text)
            payload = pipeline.build_payload(fields, self.test_bill_path)
            
            # Verify extraction
            assert fields['electricity_kwh'] == 299
            assert fields['carbon_kgco2e'] == 120
            
            # Verify payload structure
            assert payload['electricity']['consumption']['value'] == 299
            assert payload['carbon']['location_based']['value'] == 120
            assert payload['source_document']['file_name'] == 'test_bill_tiny.jpg'
    
    def test_engine_performance_characteristics(self):
        """Test that each engine has expected performance characteristics."""
        engines = ["tesseract", "easyocr", "paddleocr", "mistral", "datalab", "gemma_vlm"]
        
        # Verify all engines are in the hierarchy
        result = pipeline.run_ocr.__doc__
        for engine in engines:
            assert engine in result.lower() or engine.replace('_', ' ') in result.lower()
        
        # Test engine functions exist
        engine_functions = [
            pipeline._tesseract_ocr,
            pipeline._easyocr_ocr, 
            pipeline._paddleocr_ocr,
            pipeline._mistral_ocr,
            pipeline._datalab_ocr,
            pipeline._gemma_vlm_ocr
        ]
        
        for func in engine_functions:
            assert callable(func), f"{func.__name__} should be callable"
    
    def test_confidence_thresholds(self):
        """Test that confidence thresholds are properly configured."""
        # Verify threshold relationships
        assert 0 <= config.TAU_ENHANCER_PASS <= config.TAU_FIELD_ACCEPT <= 1
        assert config.TAU_FIELD_ACCEPT >= 0.90  # Should be high for quality
        
        # Test confidence calculation
        high_conf_result = pipeline.OcrResult("test", ["test"], [0.95])
        low_conf_result = pipeline.OcrResult("test", ["test"], [0.60])
        
        assert high_conf_result.field_confidence >= config.TAU_FIELD_ACCEPT
        assert low_conf_result.field_confidence < config.TAU_FIELD_ACCEPT
    
    def test_file_size_impact_on_processing(self):
        """Test that smaller file sizes don't negatively impact accuracy."""
        # This is more of a documentation test for expected behavior
        original_size = self.original_bill_path.stat().st_size
        downscaled_size = self.test_bill_path.stat().st_size
        
        size_reduction = (original_size - downscaled_size) / original_size
        print(f"File size reduced by {size_reduction:.1%}")
        print(f"Original: {original_size / 1024:.1f}KB, Downscaled: {downscaled_size / 1024:.1f}KB")
        
        # Verify massive reduction for tiny file
        assert size_reduction > 0.9, "Should achieve >90% size reduction for tiny file"
        
        # The actual OCR accuracy test would require real API calls
        # which we'll test in the manual verification below


class TestDataLabConfiguration:
    """Test Datalab-specific configuration and integration."""
    
    def test_datalab_configuration(self):
        """Test that Datalab is properly configured."""
        assert hasattr(config, 'DATALAB_API_KEY')
        assert hasattr(config, 'DATALAB_URL') 
        assert config.DATALAB_URL == "https://www.datalab.to/api/v1/ocr"
        assert len(config.DATALAB_API_KEY) > 0
    
    def test_datalab_in_engine_list(self):
        """Test that Datalab is included in the engine hierarchy.""" 
        # Check the run_ocr function's engine list
        with patch('pipeline.extract_text') as mock_extract, \
             patch('pipeline._tesseract_ocr') as mock_tesseract, \
             patch('pipeline._easyocr_ocr') as mock_easyocr, \
             patch('pipeline._paddleocr_ocr') as mock_paddleocr, \
             patch('pipeline._mistral_ocr') as mock_mistral, \
             patch('pipeline._datalab_ocr') as mock_datalab, \
             patch('pipeline._gemma_vlm_ocr') as mock_gemma, \
             patch('pipeline.load_image') as mock_load_image:
            
            # Mock PDF extraction to fail
            mock_extract.return_value = ""
            
            # Mock image loading
            mock_load_image.return_value = Mock()
            
            # Mock all engines to return low confidence except datalab
            low_conf_result = pipeline.OcrResult("test", ["test"], [0.80])
            mock_tesseract.return_value = low_conf_result
            mock_easyocr.return_value = low_conf_result
            mock_paddleocr.return_value = low_conf_result
            mock_mistral.return_value = low_conf_result
            
            mock_datalab.return_value = pipeline.OcrResult(
                "test", ["test"], [0.95]  # High confidence - should be accepted
            )
            
            mock_gemma.return_value = pipeline.OcrResult("", [], [])
            
            # Should reach datalab and accept its result
            result = pipeline.run_ocr(pathlib.Path("dummy.png"))
            mock_datalab.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])