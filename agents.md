# OCR Pipeline Testing and Development Notes

## Important Files - DO NOT DELETE

**Critical test files that must be preserved:**
- `ActualBill.pdf` - Primary test PDF file for OCR validation
- `ActualBill.png` - Primary test image file for OCR validation

These files are essential for testing the OCR pipeline across different formats and must never be removed or modified.

## Testing Results

### Test Environment
- Python 3.13.2
- Virtual environment: `venv/`
- Dependencies installed from `requirements.txt`
- System packages `tesseract-ocr` and `poppler-utils` installed via `apt`

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
   sudo apt-get update && sudo apt-get install -y tesseract-ocr poppler-utils
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Test with Images**:
   ```bash
   python pipeline.py ActualBill.png --lang eng
   ```

3. **Test with PDFs**:
   ```bash
   python pipeline.py ActualBill.pdf --lang eng+deu
   ```

4. **Run Test Suite**:
   ```bash
   pytest -q  # Should show 54 passed tests
   ```

If `--lang` is omitted, the first 200 characters are analysed with a fastText
model to guess the best language pack. Detected font attributes (monospace or
handwritten) are logged for troubleshooting.

### Test Results with ActualBill Files

#### ActualBill.png (Tesseract backend with GPT-4o Fallback)
- **Status**: ✅ PASSED
- **Electricity**: 299 kWh
- **Carbon**: 120 kgCO2e  
- **OCR Confidence**: 60% (below 70% threshold)
- **Fallback Triggered**: ✅ GPT-4o Vision API called
- **Final Confidence**: 100% (LLM extraction)
- **Processing flow**: 
  1. Tesseract OCR extracted text with 60% confidence
  2. Confidence below 70% threshold triggered LLM fallback
  3. GPT-4o Vision API processed original image
  4. LLM returned accurate JSON: `{"electricity_kwh": 299, "carbon_kgco2e": 120}`
  5. Final extraction confidence: 100%

#### ActualBill.pdf (Digital text extraction)
- **Status**: ✅ PASSED
- **Electricity**: 299 kWh
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