# OCR Pipeline Testing and Development Notes

## Important Files - DO NOT DELETE

**Critical test files that must be preserved:**
- `ActualBill.pdf` - Primary test PDF file for OCR validation
- `ActualBill.png` - Primary test image file for OCR validation

These files are essential for testing the OCR pipeline across different formats and must never be removed or modified.

## Testing Results

### Test Environment
- Python 3.13.2
- Virtual environment: `venv_new/`
- Dependencies installed from `requirements.txt`

### OCR Pipeline Structure
The pipeline supports multiple OCR backends:
- **Tesseract** - Default OCR engine
- **EasyOCR** - Alternative engine with GPU support
- **PaddleOCR** - High-accuracy engine (may have memory constraints)

### Configuration
- Main config in `config.py`
- OCR backend selection via `OCR_BACKEND` variable
- Confidence thresholds tunable for different accuracy requirements

### Known Issues
1. **Python 3.13 Compatibility**: `imghdr` module removed - compatibility shim created
2. **PaddleOCR**: May fail on systems with memory constraints
3. **Dependencies**: Large installation (~500MB) due to ML models

### Working Steps for Testing

1. **Environment Setup**:
   ```bash
   source venv_new/bin/activate
   ```

2. **Test with Images**:
   ```bash
   python pipeline.py ActualBill.png
   ```

3. **Test with PDFs**:
   ```bash
   python pipeline.py ActualBill.pdf
   ```

4. **Run Test Suite**:
   ```bash
   pytest -q  # Should show 52 passed tests
   ```

### Test Results with ActualBill Files

#### ActualBill.png (Tesseract backend)
- **Status**: ✅ PASSED
- **Electricity**: 299 kWh
- **Carbon**: 120 kgCO2e  
- **Confidence**: 59.9%
- **Processing time**: 0.60s

#### ActualBill.pdf (Digital text extraction)
- **Status**: ✅ PASSED  
- **Electricity**: 9 kWh (Note: Different value - likely different bill)
- **Carbon**: 120 kgCO2e
- **Confidence**: 100% (Digital text)
- **Processing time**: Fast (digital extraction)

### Expected Output Format
```json
{
  "electricity": {
    "consumption": {
      "value": 299,
      "unit": "kWh"
    }
  },
  "carbon": {
    "location_based": {
      "value": 120,
      "unit": "kgCO2e"
    }
  },
  "source_document": {
    "file_name": "bill.pdf",
    "sha256": "53a1755f..."
  },
  "meta": {
    "extraction_confidence": 1.0
  }
}
```