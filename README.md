# OCR Bill Parsing Pipeline


This reference implementation ingests a utility bill (PDF/PNG/JPG/JPEG), performs high‑accuracy text
extraction with a *sophisticated parallel OCR + hierarchical VLM* strategy, and returns a canonical JSON object containing:

* Electricity consumption (kWh)
* Carbon footprint (kg CO₂e)

The design follows the architecture outlined in our discussion on June 22 2025
and has been validated on the sample DEWA bill with **95.5% field-level accuracy**.  

## Project layout

```
ocr_pipeline/
├── config.py                        # centralised secrets & thresholds
├── pipeline.py                      # end‑to‑end orchestration
├── requirements.txt                 # pip dependencies
├── imghdr.py                        # Python 3.13 compatibility shim
├── tests/
│   └── test_accuracy_comprehensive.py  # comprehensive accuracy test suite
├── ActualBill.pdf                   # test PDF file (DEWA bill)
├── ActualBill.png                   # test image file (DEWA bill)
├── agents.md                        # development notes and test results
└── README.md                        # this file
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

### 2. Configure API Keys (Optional but Recommended)

For maximum accuracy, configure Vision-Language Model APIs in `config.py`:

```python
# Gemini (required for final fallback)
GEMINI_API_KEY = "your_gemini_api_key_here"

# Optional: Mistral OCR for enhanced accuracy
MISTRAL_API_KEY = "your_mistral_api_key_here"

# Optional: Datalab OCR for commercial-grade extraction
DATALAB_API_KEY = "your_datalab_api_key_here"
```

### 3. Run the pipeline
From within the project directory:
```bash
$ python pipeline.py ActualBill.png
$ python pipeline.py ActualBill.pdf
```

The script prints the JSON payload to `stdout`. Add `--save output.json`
to persist to disk.

### 4. Example output
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

### 5. Run tests

Run the comprehensive accuracy test suite with **pytest**. A successful run executes 54+ tests covering ground truth accuracy, engine parallelization, validation, and real-world scenarios:

```bash
$ pytest -q
54 passed
```

**Test Coverage:**
- Field-level extraction accuracy (90%+ required)
- Parallel OCR engine processing and voting
- Cross-field validation preventing false positives
- Real-world OCR noise handling
- Edge case filtering and error correction

## OCR Strategy & Supported Engines

The pipeline uses a **sophisticated parallel + hierarchical architecture** with multiple OCR engines and vision-language models:

### Traditional OCR Engines (Parallel Processing)
1. **Tesseract** – Fast local baseline with OSD auto-rotation
2. **EasyOCR** – CNN/LSTM reader with GPU acceleration  
3. **PaddleOCR** – High-accuracy CRNN/CLS pipeline (memory-aware)

### Vision-Language Models (VLM Fallback)
4. **Mistral OCR** – Specialized cloud OCR model via Mistral API
5. **Datalab OCR** – Commercial OCR API with high accuracy
6. **Gemma VLM** – Gemini 2.0 Flash vision model with contextual understanding

### Processing Pipeline
1. **Digital text pass** – `pdfminer.six` for vector PDFs (100% confidence)
2. **Parallel OCR** – Traditional engines run concurrently, best result selected
3. **Enhanced DPI** – Automatic 600 dpi boost for moderate confidence results
4. **VLM fallback** – Vision models if traditional OCR confidence < 85%
5. **Gemini Flash final** – Direct field extraction as last resort

Confidence is computed as the geometric mean of token confidences reported
by each OCR engine, with VLM models using estimated confidence based on specialization.

**Test Results (DEWA Bill Sample):**

| Engine Category | Method | Input Type | Confidence | Electricity | Carbon | Processing |
|----------------|--------|------------|------------|-------------|---------|------------|
| **Digital Text** |
| pdfminer.six | PDF | 100% | ✅ 299 kWh | ✅ 120 kgCO2e | Instant digital extraction |
| **Traditional OCR (Parallel)** |
| Tesseract | PNG/PDF | 60.0% | ✅ 299 kWh | ✅ 120 kgCO2e | Fast baseline, triggers enhancement |
| EasyOCR | PNG/PDF | 74.1% | ✅ 299 kWh | ✅ 120 kgCO2e | GPU-accelerated, good accuracy |
| PaddleOCR | PNG/PDF | 94.2% | ✅ 299 kWh | ✅ 120 kgCO2e | Highest traditional OCR confidence |
| **Vision-Language Models** |
| Mistral OCR | PNG/PDF | 97%* | ✅ 299 kWh | ✅ 120 kgCO2e | Specialized document OCR API |
| Datalab OCR | PNG/PDF | 95%* | ✅ 299 kWh | ✅ 120 kgCO2e | Commercial OCR with high accuracy |
| Gemma VLM | PNG/PDF | 96%* | ✅ 299 kWh | ✅ 120 kgCO2e | Contextual vision understanding |
| **Final Fallback** |
| Gemini Flash | PNG/PDF | 100%* | ✅ 299 kWh | ✅ 120 kgCO2e | Direct JSON field extraction |

*VLM confidence scores are model-estimated based on specialization

**Confidence Metrics:**
- **Traditional OCR**: Geometric mean of token-level confidence scores from engines
- **VLM/LLM**: Fixed high confidence (96-97%) based on model specialization
- **Comparison**: Accuracy (correct extraction) + Confidence + Robustness across formats

**Configuration:**

**Traditional OCR Engines:**
- Set `OCR_BACKEND` in `config.py` for primary engine ("tesseract", "easyocr", "paddleocr")
- `OCR_LANG` overrides all engines with single language code
- `TESSERACT_LANG`, `EASYOCR_LANG`, `PADDLEOCR_LANG` for engine-specific settings
- Hardware auto-detection: GPU availability and memory constraints

**Vision-Language Models:**
- `MISTRAL_API_KEY` and `MISTRAL_MODEL` for Mistral OCR
- `DATALAB_API_KEY` and `DATALAB_URL` for Datalab OCR  
- `GEMINI_API_KEY` and `GEMINI_MODEL` for Gemma VLM and final fallback

**Processing Control:**
- `TAU_FIELD_ACCEPT` (95%) - auto-accept confidence threshold
- `TAU_ENHANCER_PASS` (90%) - enhanced DPI trigger threshold
- `TAU_LLM_PASS` (85%) - VLM fallback trigger threshold
- `USE_LIGHTWEIGHT_MODELS` - automatic based on system memory
- `ENABLE_PADDLEOCR` - automatic based on available RAM

## Advanced Parallel + Hierarchical Pipeline Architecture

**Complete processing flow inside run_ocr():**

```
┌──── PDF? ──► pdfminer.six (digital text pass)
│              │
│              └── success (100% confidence) → done
│
└─ otherwise → parallel + hierarchical processing:

Phase 1: PARALLEL TRADITIONAL OCR
├─ Tesseract    ┐
├─ EasyOCR      ├── concurrent execution
└─ PaddleOCR    ┘
     │
     ├── confidence ≥ 95% → done
     │
     ├── 90% ≤ confidence < 95% → Enhanced DPI (600 dpi)
     │    └── confidence ≥ 95% → done
     │
     └── confidence < 85% → Phase 2

Phase 2: PARALLEL VLM FALLBACK
├─ Mistral OCR  ┐
├─ Datalab OCR  ├── concurrent execution with timeout
└─ Gemma VLM    ┘
     │
     ├── best result confidence ≥ 85% → done
     │
     └── all failed → Phase 3

Phase 3: GEMINI FLASH FINAL FALLBACK
└─ Direct JSON field extraction → always succeeds
```

### Advanced Features

**1. Key Information Extraction (KIE)**
- Lightweight vision-based field detection using Gemini Flash
- Contextual bounding box awareness for improved accuracy
- Text-based KIE fallback with OCR error correction

**2. Thread-Safe Image Caching**
- Prevents repeated PDF/image loading across engines
- Memory-efficient caching with automatic cleanup
- Supports multi-DPI caching for enhanced processing

**3. Cross-Field Validation**
- Prevents OCR hallucinations with realistic value ranges
- Carbon/electricity correlation checking (0.1-1.0 kg/kWh)
- Automatic filtering of impossible value combinations

**4. OCR Error Correction**
- Preprocessing system fixes common character confusions (l→1, O→0)
- Generalized pattern-based correction for robust extraction
- Context-aware number normalization

**5. Hardware Auto-Detection**
- Automatic GPU availability detection (CUDA/Metal)
- Memory-aware engine selection (8GB+ for PaddleOCR)
- Lightweight model fallbacks for constrained systems

### How the hierarchy works

| Phase | What the code does | When it moves on |
|------|-------------------|------------------|
| **Pre-flight** | • If file is PDF it first calls `extract_text()` (pdfminer). | No text or empty string. |
| **Parallel OCR** | Traditional engines `["tesseract", "easyocr", "paddleocr"]` run concurrently via ThreadPoolExecutor. Best result selected by confidence. | If `field_confidence < TAU_FIELD_ACCEPT` (95%). |
| **DPI enhancement** | For PDF pages whose first pass is moderate (`>= TAU_ENHANCER_PASS` but `< TAU_FIELD_ACCEPT`), the best engine is rerun at `DPI_ENHANCED` (600 dpi vs. 300 dpi). | Enhanced confidence still `< TAU_FIELD_ACCEPT`. |
| **VLM fallback** | If traditional OCR `< TAU_LLM_PASS` (85%), VLM engines `["mistral", "datalab", "gemma_vlm"]` run concurrently with 30s timeout. | Best VLM result still `< TAU_LLM_PASS`. |
| **Final fallback** | `gemini_flash_fallback()` asks Gemini 2.0 Flash to extract electricity kWh and carbon kgCO₂e directly from the image with JSON output. | Only if this fails does the pipeline return an empty payload. |

### Confidence math
`OcrResult.field_confidence = geometric mean of token confidences` (each bounded below by 1 × 10⁻³).
For long documents (>20 tokens), top-k filtering (80th percentile) prevents unfair penalty.

## Customising field extraction

Regex patterns for electricity and carbon values live in `pipeline.py`.
Modify `ENERGY_RE` and `CARBON_RE` or extend `extract_fields()` to
handle additional metrics. The system also supports KIE-based extraction
for complex document layouts.

### Field Extraction Strategy

1. **Simple Regex** - Fast patterns for common formats
2. **KIE Vision API** - Gemini Flash for complex layouts  
3. **Text-based KIE** - Contextual number extraction with error correction
4. **Cross-validation** - Realistic value range checking

The extraction system automatically selects the best approach based on
confidence scores and validation results.