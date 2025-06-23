# OCR Bill Parsing Pipeline

**Version:** 1.2.0  
**Last updated:** 2025-06-22 14:00:00 UTC

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

### 1. Set up virtual environment

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

### 2. Run the pipeline
From within the project directory:
```bash
$ python pipeline.py utility_bill.pdf
$ python pipeline.py Actualbill.png
$ python pipeline.py bill_image.jpg
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
    "file_name": "bill.pdf",
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
| Tesseract | PNG Image | 37.4% | ✅ 299 kWh | ✅ 120 kgCO2e | Complete extraction |
| EasyOCR | PNG Image | 75.2% | ✅ 299 kWh | ✅ 120 kgCO2e | Complete extraction with enhanced patterns |
| PaddleOCR | PNG Image | 94.2% | ✅ 299 kWh | ✅ 120 kgCO2e | Highest confidence, complete extraction |
| **Digital Text Extraction (PDF)** |
| pdfminer.six | PDF | 100% | ✅ 299 kWh | ✅ 120 kgCO2e | Perfect digital text extraction |

**Configuration:**
Set `OCR_BACKEND` in `config.py` to choose engine ("tesseract", "easyocr", "paddleocr").
Tesseract language, OEM and PSM settings can be adjusted in `config.py` to match document type.
EasyOCR uses `EASYOCR_LANG` (e.g. `['en', 'fr']`), while PaddleOCR uses
`PADDLEOCR_LANG` (e.g. `'en'` or `'ch'`). Set `OCR_LANG` to a language code to
override both engines with a single value.

## Hard‑coded API keys

`config.py` contains an example key for OpenAI integration:

* **OpenAI GPT‑4o** – `OPENAI_API_KEY`

When processing image files, the pipeline will automatically call GPT‑4o as a
fallback if OCR confidence is low. The key is defined directly in `config.py` as
`OPENAI_API_KEY`.

> **Important**: Keys are fake placeholders. Replace them with real credentials
> before first run. Hard‑coding is **not** recommended in production.

## Customising field extraction

Regex patterns for electricity and carbon values live in `pipeline.py`.
Modify `ENERGY_RE` and `CARBON_RE` or extend `extract_fields()` to
handle additional metrics.

## License

MIT – Feel free to fork, adapt and deploy.
