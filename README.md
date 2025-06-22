# OCR Bill Parsing Pipeline

**Version:** 1.1.0
**Last updated:** 2025-06-22 13:00:00 UTC

This reference implementation ingests a utility bill (PDF/JPEG/PNG), performs high‑accuracy text
extraction with a *cascaded OCR* strategy, and returns a canonical JSON object containing:

* Electricity consumption (kWh)
* Carbon footprint (kg CO₂e)

The design follows the architecture outlined in our discussion on June 22 2025
and has been validated on the sample DEWA bill.  

## Project layout

```
robust_ocr_pipeline/
├── config.py        # centralised secrets & thresholds
├── pipeline.py      # end‑to‑end orchestration
├── requirements.txt # pip dependencies
└── README.md        # this file
```

## Quick‑start

```bash
$ python -m venv venv && source venv/bin/activate
$ pip install -r requirements.txt
$ python pipeline.py /path/to/utility_bill.pdf
```

The script prints the JSON payload to `stdout`. Add `--save output.json`
to persist to disk.

## OCR strategy

1. **Digital text pass** – `pdfminer.six` (vector PDFs only).
2. **Bitmap pass** – `pytesseract` at 300 dpi via `pdf2image`.
3. **Orientation check** – pages are auto‑rotated using Tesseract OSD.
4. **Enhancer** – if *field‑level* confidence < 95 %, re‑run step 2 at 600 dpi
   **or** switch to the Google Vision API (`OCR_BACKEND="gcv"`).
5. **LLM fallback** – optional: set `USE_LLM_FALLBACK=True` in `config.py`.

Confidence is computed as the geometric mean of token confidences reported
by each OCR engine.

Tesseract language, OEM and PSM settings can be adjusted in `config.py`
to match the document type.

## Hard‑coded API keys

`config.py` contains example keys for:

* **Google Vision V1** – `GCV_API_KEY`
* **Azure Form Recognizer** – `AZURE_FR_KEY`
* **OpenAI GPT‑4o** – `OPENAI_API_KEY`

> **Important**: Keys are fake placeholders. Replace them with real credentials
> before first run. Hard‑coding is **not** recommended in production.

## Customising field extraction

Regex patterns for electricity and carbon values live in `pipeline.py`.
Modify `ENERGY_RE` and `CARBON_RE` or extend `extract_fields()` to
handle additional metrics.

## License

MIT – Feel free to fork, adapt and deploy.
