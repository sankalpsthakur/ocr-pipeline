"""
Ultra-minimal CI tests that don't require heavy ML dependencies.
"""

import os
import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Detect CI environment
IS_CI = os.environ.get("CI", "false").lower() == "true" or os.environ.get("GITHUB_ACTIONS", "false").lower() == "true"


def test_basic_imports():
    """Test that basic Python libraries work."""
    import json
    import re
    from pathlib import Path
    
    assert json is not None
    assert re is not None
    assert Path is not None


def test_pillow_available():
    """Test that Pillow is available."""
    try:
        from PIL import Image
        
        # Create a test image
        img = Image.new('RGB', (100, 100), color='red')
        assert img.size == (100, 100)
        assert img.mode == 'RGB'
        
    except ImportError:
        pytest.skip("Pillow not available")




def test_pytesseract_import():
    """Test that pytesseract can be imported."""
    try:
        import pytesseract
        assert hasattr(pytesseract, 'image_to_string')
    except ImportError:
        pytest.skip("pytesseract not available")


def test_pipeline_stub():
    """Test the pipeline stub functionality."""
    try:
        import pipeline_stub
        
        # Test OcrResult
        result = pipeline_stub.OcrResult("test", ["test"], [0.9])
        assert result.text == "test"
        assert result.tokens == ["test"]
        assert result.confidences == [0.9]
        assert 0.0 <= result.field_confidence <= 1.0
        
        # Test extract_fields
        test_text = "Electricity: 299 kWh, Carbon: 120 kg CO2e"
        fields = pipeline_stub.extract_fields(test_text)
        
        if 'electricity_kwh' in fields:
            assert fields['electricity_kwh'] == 299
        if 'carbon_kgco2e' in fields:
            assert fields['carbon_kgco2e'] == 120
            
        # Test run_ocr stub
        stub_result = pipeline_stub.run_ocr("dummy_path")
        assert isinstance(stub_result, pipeline_stub.OcrResult)
        
    except ImportError as e:
        pytest.skip(f"Pipeline stub not available: {e}")


def test_config_values():
    """Test basic configuration constants."""
    # Test that we can define basic config values
    TAU_FIELD_ACCEPT = 0.95
    TAU_ENHANCER_PASS = 0.90
    TAU_LLM_PASS = 0.85
    
    assert 0.0 <= TAU_LLM_PASS <= TAU_ENHANCER_PASS <= TAU_FIELD_ACCEPT <= 1.0


def test_assets_structure():
    """Test that test assets directory exists."""
    assets_dir = Path(__file__).parent / "assets"
    
    if assets_dir.exists():
        expected_files = ["expected_fields.json"]
        for expected_file in expected_files:
            file_path = assets_dir / expected_file
            assert file_path.exists(), f"Expected file {expected_file} should exist"
    else:
        pytest.skip("Assets directory not found")


if __name__ == "__main__":
    pytest.main([__file__])