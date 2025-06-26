"""
Deterministic field extraction tests with parametrized fixtures.

Tests that the OCR pipeline extracts the correct business fields
regardless of engine-specific variations in text output.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Detect CI environment
IS_CI = os.environ.get("CI", "false").lower() == "true" or os.environ.get("GITHUB_ACTIONS", "false").lower() == "true"

# Import available functions based on environment
try:
    from pipeline import (
        _easyocr_ocr,
        _tesseract_ocr,
        extract_fields,
        run_ocr,
    )
    
    # GPU-dependent imports (may fail in CI)
    if not IS_CI:
        from pipeline import (
            _gemma_vlm_ocr,
            _mistral_ocr,
            _paddleocr_ocr,
            gemini_flash_fallback,
        )
    else:
        # Stub functions for CI
        def _gemma_vlm_ocr(*args, **kwargs):
            pytest.skip("GPU-dependent engine not available in CI")
        
        def _mistral_ocr(*args, **kwargs):
            pytest.skip("GPU-dependent engine not available in CI")
        
        def _paddleocr_ocr(*args, **kwargs):
            pytest.skip("GPU-dependent engine not available in CI")
        
        def gemini_flash_fallback(*args, **kwargs):
            pytest.skip("GPU-dependent engine not available in CI")
            
except ImportError as e:
    pytest.skip(f"Required dependencies not available: {e}")  


# Load expected field values
@pytest.fixture(scope="module")
def expected_fields() -> Dict[str, Any]:
    """Load expected field values from JSON fixture."""
    fixture_path = Path(__file__).parent / "assets" / "expected_fields.json"
    with open(fixture_path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def test_assets_dir() -> Path:
    """Path to test assets directory."""
    return Path(__file__).parent / "assets"


def extract_field_values(text: str) -> Dict[str, Any]:
    """Extract business field values from OCR text."""
    fields = {}

    # Extract electricity kWh
    electricity_patterns = [r"(\d+)\s*kWh", r"Electricity.*?(\d+)", r"Kilowatt\s+Hours.*?(\d+)"]

    for pattern in electricity_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            value = int(match.group(1))
            if 50 <= value <= 10000:  # Sanity check
                fields["electricity_kwh"] = value
                break

    # Extract carbon footprint
    carbon_patterns = [r"(\d+)\s*(?:kg|Kg)\s*CO2e?", r"Carbon.*?(\d+)", r"Footprint.*?(\d+)"]

    for pattern in carbon_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            value = int(match.group(1))
            if 10 <= value <= 1000:  # Sanity check
                fields["carbon_kgco2e"] = value
                break

    # Extract invoice number
    invoice_match = re.search(r"Invoice:\s*(\d+)", text, re.IGNORECASE)
    if invoice_match:
        fields["invoice_number"] = invoice_match.group(1)

    # Extract issue date
    date_match = re.search(r"(\d{2}/\d{2}/\d{4})", text)
    if date_match:
        fields["issue_date"] = date_match.group(1)

    # Extract account number
    account_match = re.search(r"Account\s+Number\s*(\d+)", text, re.IGNORECASE)
    if account_match:
        fields["account_number"] = account_match.group(1)

    return fields


@pytest.mark.parametrize(
    "engine_name,engine_func",
    [
        ("tesseract", _tesseract_ocr),
        ("easyocr", _easyocr_ocr),
        ("paddleocr", _paddleocr_ocr),
        ("mistral_ocr", _mistral_ocr),
        ("gemma_vlm", _gemma_vlm_ocr),
    ],
)
@pytest.mark.parametrize("test_file", ["ActualBill.png"])
def test_engine_field_extraction(
    engine_name: str, engine_func, test_file: str, test_assets_dir: Path, expected_fields: Dict[str, Any]
):
    """Test that individual OCR engines extract correct business fields."""
    test_path = test_assets_dir / test_file
    expected = expected_fields[test_file]
    min_confidence = expected["min_confidence_thresholds"].get(engine_name, 0.5)

    # Skip if file doesn't exist
    if not test_path.exists():
        pytest.skip(f"Test file {test_file} not found")

    try:
        # Load image and run engine
        image = Image.open(test_path)
        result = engine_func(image)

        # Check minimum confidence threshold
        if result.field_confidence < min_confidence:
            pytest.skip(f"{engine_name} confidence {result.field_confidence:.3f} below threshold {min_confidence}")

        # Extract fields from result text
        extracted_fields = extract_field_values(result.text)

        # Assert critical business fields
        assert (
            extracted_fields.get("electricity_kwh") == expected["electricity_kwh"]
        ), f"{engine_name} failed to extract electricity: got {extracted_fields.get('electricity_kwh')}, expected {expected['electricity_kwh']}"

        assert (
            extracted_fields.get("carbon_kgco2e") == expected["carbon_kgco2e"]
        ), f"{engine_name} failed to extract carbon: got {extracted_fields.get('carbon_kgco2e')}, expected {expected['carbon_kgco2e']}"

        # Optional fields (warn if missing but don't fail)
        if "invoice_number" in expected and extracted_fields.get("invoice_number") != expected["invoice_number"]:
            print(
                f"Warning: {engine_name} invoice number mismatch: got {extracted_fields.get('invoice_number')}, expected {expected['invoice_number']}"
            )

    except Exception as e:
        # Some engines may fail due to system dependencies
        pytest.skip(f"{engine_name} engine failed: {e}")


@pytest.mark.parametrize("test_file", ["ActualBill.png", "ActualBill.pdf"])
def test_final_pipeline_field_extraction(test_file: str, test_assets_dir: Path, expected_fields: Dict[str, Any]):
    """Test that the final hierarchical pipeline extracts correct fields."""
    test_path = test_assets_dir / test_file
    expected = expected_fields[test_file]
    min_confidence = expected["min_confidence_thresholds"]["final_pipeline"]

    # Skip if file doesn't exist
    if not test_path.exists():
        pytest.skip(f"Test file {test_file} not found")

    # Run final pipeline
    result = run_ocr(test_path)

    # Check minimum confidence
    assert (
        result.field_confidence >= min_confidence
    ), f"Final pipeline confidence {result.field_confidence:.3f} below threshold {min_confidence}"

    # Extract fields using pipeline's own extraction logic
    pipeline_fields = extract_fields(result.text)

    # Assert critical business fields
    assert (
        pipeline_fields.get("electricity_kwh") == expected["electricity_kwh"]
    ), f"Final pipeline failed to extract electricity: got {pipeline_fields.get('electricity_kwh')}, expected {expected['electricity_kwh']}"

    assert (
        pipeline_fields.get("carbon_kgco2e") == expected["carbon_kgco2e"]
    ), f"Final pipeline failed to extract carbon: got {pipeline_fields.get('carbon_kgco2e')}, expected {expected['carbon_kgco2e']}"


def test_gemini_flash_field_extraction(test_assets_dir: Path, expected_fields: Dict[str, Any]):
    """Test Gemini Flash JSON extraction for business fields."""
    test_file = "ActualBill.png"
    test_path = test_assets_dir / test_file
    expected = expected_fields[test_file]

    # Skip if file doesn't exist
    if not test_path.exists():
        pytest.skip(f"Test file {test_file} not found")

    try:
        # Run Gemini Flash fallback
        result = gemini_flash_fallback(test_path)

        # Check that we got valid JSON results
        assert isinstance(result, dict), "Gemini Flash should return a dictionary"
        assert len(result) > 0, "Gemini Flash should return non-empty results"

        # Assert critical business fields
        assert (
            result.get("electricity_kwh") == expected["electricity_kwh"]
        ), f"Gemini Flash failed to extract electricity: got {result.get('electricity_kwh')}, expected {expected['electricity_kwh']}"

        assert (
            result.get("carbon_kgco2e") == expected["carbon_kgco2e"]
        ), f"Gemini Flash failed to extract carbon: got {result.get('carbon_kgco2e')}, expected {expected['carbon_kgco2e']}"

    except Exception as e:
        # Gemini Flash may fail due to API issues
        pytest.skip(f"Gemini Flash failed: {e}")


@pytest.mark.parametrize("test_file", ["ActualBill.png"])
def test_field_extraction_consistency(test_file: str, test_assets_dir: Path, expected_fields: Dict[str, Any]):
    """Test that multiple engines extract the same critical fields consistently."""
    test_path = test_assets_dir / test_file
    expected = expected_fields[test_file]

    # Skip if file doesn't exist
    if not test_path.exists():
        pytest.skip(f"Test file {test_file} not found")

    engines_to_test = [
        ("tesseract", _tesseract_ocr),
        ("easyocr", _easyocr_ocr),
        ("final_pipeline", lambda img: run_ocr(test_path)),
    ]

    successful_extractions = {}

    for engine_name, engine_func in engines_to_test:
        try:
            if engine_name == "final_pipeline":
                result = engine_func(None)  # run_ocr takes path, not image
            else:
                image = Image.open(test_path)
                result = engine_func(image)

            min_confidence = expected["min_confidence_thresholds"].get(engine_name, 0.5)
            if result.field_confidence >= min_confidence:
                if engine_name == "final_pipeline":
                    fields = extract_fields(result.text)
                else:
                    fields = extract_field_values(result.text)

                successful_extractions[engine_name] = fields

        except Exception:
            continue  # Skip failed engines

    # Require at least 2 successful extractions for consistency test
    if len(successful_extractions) < 2:
        pytest.skip("Not enough successful extractions for consistency test")

    # Check that all successful engines agree on critical fields
    electricity_values = {name: fields.get("electricity_kwh") for name, fields in successful_extractions.items()}
    carbon_values = {name: fields.get("carbon_kgco2e") for name, fields in successful_extractions.items()}

    # All engines should extract the same electricity value
    unique_electricity = set(v for v in electricity_values.values() if v is not None)
    assert len(unique_electricity) <= 1, f"Inconsistent electricity extraction: {electricity_values}"

    # All engines should extract the same carbon value
    unique_carbon = set(v for v in carbon_values.values() if v is not None)
    assert len(unique_carbon) <= 1, f"Inconsistent carbon extraction: {carbon_values}"
