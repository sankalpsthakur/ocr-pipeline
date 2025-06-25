# OCR Bill Parsing Pipeline


This reference implementation ingests a utility bill (PDF/PNG/JPG/JPEG), performs high‑accuracy text
extraction with a *cascaded OCR* strategy, and returns a canonical JSON object containing:

* Electricity consumption (kWh)
* Carbon footprint (kg CO₂e)

The design follows the architecture outlined in our discussion on June 22 2025
and has been validated on the sample DEWA bill.  

## Project layout

```
ocr_pipeline/
├── config.py        # centralised secrets & thresholds
├── pipeline.py      # end‑to‑end orchestration
├── requirements.txt # pip dependencies
├── tests/           # unit tests
└── README.md        # this file
```

## Installation & Quick‑start

### System packages
Install `tesseract-ocr` and `poppler-utils` via `apt` before installing Python dependencies:

```bash
$ sudo apt-get update && sudo apt-get install -y tesseract-ocr poppler-utils
```

### 1. Set up the `venv` virtual environment

**Requirements:** Python 3.8+ (tested with Python 3.13)

```bash
$ python3 -m venv venv
$ source venv/bin/activate  # On Windows: venv\Scripts\activate
$ pip install --upgrade pip setuptools wheel
$ pip install -r requirements.txt
```

**Note:** 
- The requirements.txt includes all OCR engines (Tesseract, EasyOCR, PaddleOCR) with dependencies
- PaddlePaddle framework (96MB) is automatically installed for PaddleOCR support
- **PaddleOCR optimized for 8GB Macs**: Uses minimal resolution (320px) and single-threaded processing
- If you encounter issues with Pillow on Python 3.13, the installation process will automatically use a compatible version (Pillow 11.2.1+)
- Total installation size: ~500MB including all ML models
- The pipeline relies on `pdf2image` and `pytesseract` (installed via `pip`)

### 2. Run the pipeline
From within the project directory:
```bash
$ python pipeline.py ActualBill.png
$ python pipeline.py ActualBill.pdf
```

The script prints the JSON payload to `stdout`. Add `--save output.json`
to persist to disk.

### 3. Example output
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
    "file_name": "ActualBill.pdf",
    "sha256": "53a1755f..."
  },
  "meta": {
    "extraction_confidence": 1.0
  }
}
```

### 4. Run tests

Run the full unit test suite with **pytest**. A successful run executes 54 tests:

```bash
$ pytest -q
54 passed
```

## OCR Strategy & Supported Engines

1. **Digital text pass** – `pdfminer.six` (vector PDFs only).
2. **Bitmap pass** – `pytesseract` at 300 dpi via `pdf2image`.
3. **Orientation check** – pages are auto‑rotated using Tesseract OSD.
4. **Enhancer** – if *field‑level* confidence < 95 %, re‑run step 2 at 600 dpi
   **or** switch to an alternate engine (`OCR_BACKEND="easyocr"` or `"paddleocr"`).
5. **LLM fallback** – optional: set `USE_LLM_FALLBACK=True` in `config.py`.

Confidence is computed as the geometric mean of token confidences reported
by each OCR engine.

**Test Results (DEWA Bill Sample):**

| Method | Input Type | Confidence | Electricity | Carbon | Notes |
|--------|------------|------------|-------------|---------|-------|
| **OCR Engines (PNG Image)** |
| Tesseract | PNG Image | 60.0% | ✅ 299 kWh | ✅ 120 kgCO2e | Low confidence, triggers next engine |
| EasyOCR | PNG Image | 74.1% | ✅ 299 kWh | ✅ 120 kgCO2e | Good accuracy with enhanced patterns |
| PaddleOCR | PNG Image | 94.2% | ✅ 299 kWh | ✅ 120 kgCO2e | Highest Confidence Correct Extraction |
| **VLM/LLM Engines (PNG Image)** |
| Mistral OCR | PNG Image | 97.0% | ✅ 299 kWh | ✅ 120 kgCO2e | Specialized OCR model, high accuracy |
| Gemma VLM | PNG Image | 96.0% | ✅ 299 kWh | ✅ 120 kgCO2e | Vision-language model, excellent extraction |
| **LLM Fallback (PNG Image)** |
| Gemini Flash | PNG Image | 100% | ✅ 299 kWh | ✅ 120 kgCO2e | Final fallback for field-specific JSON extraction |
| **Digital Text Extraction (PDF)** |
| pdfminer.six | PDF | 100% | ✅ 299 kWh | ✅ 120 kgCO2e | Perfect digital text extraction |

**Confidence Metrics:**
- **Traditional OCR**: Geometric mean of token-level confidence scores from engines
- **VLM/LLM**: Fixed high confidence (96-97%) based on model specialization
- **Comparison**: Accuracy (correct extraction) + Confidence + Robustness across formats

**Configuration:**
Set `OCR_BACKEND` in `config.py` to choose engine ("tesseract", "easyocr", "paddleocr").
Tesseract language, OEM and PSM settings can be adjusted in `config.py` to match document type.
EasyOCR uses `EASYOCR_LANG` (e.g. `['en', 'fr']`), while PaddleOCR uses
`PADDLEOCR_LANG` (e.g. `'en'` or `'ch'`). Set `OCR_LANG` to a language code to
override both engines with a single value.

## Hierarchical OCR Pipeline Architecture

**Cascade order inside run_ocr():**

```
┌──── PDF? ──► try pdfminer (digital text pass)
│              │
│              └── success → done
│
└─ otherwise → hierarchical OCR engines:

1.  Tesseract        # fast, local baseline
2.  EasyOCR          # open-source CNN/LSTM reader  
3.  PaddleOCR        # high-accuracy CRNN/CLS pipeline
4.  Mistral OCR      # cloud, vision-LLM specialised
5.  Gemma VLM OCR    # Gemini 2.0 Flash vision model
───────────────────────────────────────────────
6.  Gemini Flash     # final LLM fallback (field-specific JSON)
```

### How the hierarchy works

| Step | What the code does | When it moves on |
|------|-------------------|------------------|
| **Pre-flight** | • If file is PDF it first calls `extract_text()` (pdfminer). | No text or empty string. |
| **Engine loop** | Iterates through the list `["tesseract", "easyocr", "paddleocr", "mistral", "gemma_vlm"]`. Each engine is run via `_run_ocr_engine()`. | If `field_confidence < TAU_FIELD_ACCEPT` (strict) and `< TAU_ENHANCER_PASS` after any DPI boost, the loop continues. |
| **DPI enhancement** | For PDF pages whose first pass is mediocre (`>= TAU_ENHANCER_PASS` but `< TAU_FIELD_ACCEPT`), the same engine is rerun at `DPI_ENHANCED` (typically 600 dpi vs. 300 dpi). | Enhanced confidence still too low. |
| **Accept / return** | As soon as an engine's geometric-mean confidence `field_confidence ≥ TAU_FIELD_ACCEPT`, that result is returned. | n/a |
| **LLM fallback** | If every engine is rejected, `gemini_flash_fallback()` asks Gemini 2.0 Flash to pull the electricity kWh and carbon kgCO₂e directly from the raw image/PDF. | Only if even this fails does the pipeline return an empty payload. |

### Confidence math
`OcrResult.field_confidence = geometric mean of token confidences` (each bounded below by 1 × 10⁻³).

## Customising field extraction

Regex patterns for electricity and carbon values live in `pipeline.py`.
Modify `ENERGY_RE` and `CARBON_RE` or extend `extract_fields()` to
handle additional metrics.
