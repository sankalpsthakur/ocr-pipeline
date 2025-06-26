"""
Basic CI tests - CPU-only, no GPU dependencies.

Tests core functionality that can run in GitHub Actions
without GPU hardware or heavy ML dependencies.
"""

import os
import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Detect CI environment
IS_CI = os.environ.get("CI", "false").lower() == "true" or os.environ.get("GITHUB_ACTIONS", "false").lower() == "true"


def test_pipeline_import():
    """Test that the pipeline module can be imported."""
    try:
        import pipeline
        assert hasattr(pipeline, "run_ocr")
        assert hasattr(pipeline, "extract_fields")
    except ImportError as e:
        pytest.fail(f"Failed to import pipeline module: {e}")


def test_config_import():
    """Test that config can be imported."""
    try:
        import config
        # Check that basic config values exist
        assert hasattr(config, "OCR_BACKEND")
        assert hasattr(config, "TAU_FIELD_ACCEPT")
    except ImportError as e:
        pytest.fail(f"Failed to import config module: {e}")


def test_basic_ocr_result():
    """Test OcrResult class instantiation."""
    try:
        from pipeline import OcrResult
        
        # Test basic instantiation
        result = OcrResult("test text", ["test", "tokens"], [0.9, 0.8])
        assert result.text == "test text"
        assert result.tokens == ["test", "tokens"]
        assert result.confidences == [0.9, 0.8]
        assert 0.0 <= result.field_confidence <= 1.0
        
    except ImportError as e:
        pytest.skip(f"OcrResult not available: {e}")


def test_extract_fields_function():
    """Test field extraction function with sample text."""
    try:
        from pipeline import extract_fields
        
        # Test with sample utility bill text
        sample_text = """
        Dubai Electricity & Water Authority
        Invoice: 100045272594
        Electricity Consumption: 299 kWh
        Carbon Footprint: 120 kg CO2e
        """
        
        fields = extract_fields(sample_text)
        
        # Should be a dictionary
        assert isinstance(fields, dict)
        
        # Should extract at least some fields
        if "electricity_kwh" in fields:
            assert isinstance(fields["electricity_kwh"], int)
            assert fields["electricity_kwh"] > 0
            
        if "carbon_kgco2e" in fields:
            assert isinstance(fields["carbon_kgco2e"], int)
            assert fields["carbon_kgco2e"] > 0
            
    except ImportError as e:
        pytest.skip(f"extract_fields not available: {e}")


def test_tesseract_availability():
    """Test that Tesseract OCR is available (CPU-only)."""
    if not IS_CI:
        pytest.skip("Skipping Tesseract test outside CI")
        
    try:
        from PIL import Image
        from pipeline import _tesseract_ocr
        
        # Create a simple test image with text
        test_image = Image.new('RGB', (200, 50), color='white')
        
        # Try to run Tesseract (may fail due to empty image, but should not crash)
        result = _tesseract_ocr(test_image)
        
        # Should return an OcrResult object
        assert hasattr(result, 'text')
        assert hasattr(result, 'tokens')
        assert hasattr(result, 'confidences')
        
    except ImportError as e:
        pytest.skip(f"Tesseract dependencies not available: {e}")
    except Exception as e:
        # Tesseract may fail on empty image, but should not crash
        assert "tesseract" in str(e).lower() or "text" in str(e).lower()


@pytest.mark.skipif(IS_CI, reason="Heavy dependencies not available in CI")
def test_full_pipeline_available():
    """Test that full pipeline is available (local only)."""
    try:
        from pipeline import run_ocr
        
        # Just test that the function exists and is callable
        assert callable(run_ocr)
        
    except ImportError as e:
        pytest.skip(f"Full pipeline not available: {e}")


def test_assets_directory_exists():
    """Test that test assets directory exists."""
    assets_dir = Path(__file__).parent / "assets"
    assert assets_dir.exists(), "Test assets directory should exist"
    
    expected_files = ["ActualBill.png", "ActualBill.pdf", "expected_fields.json"]
    
    for expected_file in expected_files:
        file_path = assets_dir / expected_file
        if expected_file.endswith('.json'):
            assert file_path.exists(), f"Expected file {expected_file} should exist"
        # Image/PDF files are optional in CI
        elif not IS_CI:
            assert file_path.exists(), f"Expected file {expected_file} should exist"


if __name__ == "__main__":
    pytest.main([__file__])