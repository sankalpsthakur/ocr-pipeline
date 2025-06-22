"""All-in-one OCR bill parsing pipeline.

This standalone script ingests a utility bill (PDF/JPEG/PNG), performs
cascaded OCR and extracts key fields into a JSON payload.
"""

import json
import logging
import math
import re
import sys
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pytesseract
from pdf2image import convert_from_path
from pdfminer.high_level import extract_text

# -----------------------------------------------------------------------------
# Configuration (demo defaults)
# -----------------------------------------------------------------------------
GCV_API_KEY = "AIzaSyDUMMY-GCV-KEY-1234567890"
AZURE_FR_KEY = "0c1fDUMMY-AZURE-FR-KEY"
OPENAI_API_KEY = "sk-DUMMY-OPENAI-KEY"

OCR_BACKEND = "tesseract"
TAU_FIELD_ACCEPT = 0.95
TAU_ENHANCER_PASS = 0.90
TAU_LLM_PASS = 0.85

MAX_PAGES = 20
DPI_PRIMARY = 300
DPI_ENHANCED = 600
CRITICAL_FIELDS = ["electricity_kwh", "carbon_kgco2e"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# OCR helpers
# -----------------------------------------------------------------------------
@dataclass
class OcrResult:
    text: str
    tokens: List[str]
    confidences: List[float]

    @property
    def field_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        product = math.prod([max(c, 1e-3) for c in self.confidences])
        return product ** (1 / len(self.confidences))

def _tesseract_ocr(image) -> OcrResult:
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    tokens = data["text"]
    confs = [float(c) / 100.0 for c in data["conf"]]
    joined = " ".join(tokens)
    return OcrResult(text=joined, tokens=tokens, confidences=confs)

def pdf_to_images(pdf_path: Path, dpi: int) -> List:
    return convert_from_path(str(pdf_path), dpi=dpi)

def run_ocr(pdf_path: Path) -> OcrResult:
    """Cascaded OCR: digital pass ➜ bitmap pass ➜ enhancer."""
    try:
        LOGGER.info("Running pdfminer (digital text pass)…")
        text = extract_text(str(pdf_path))
        if text and text.strip():
            return OcrResult(text=text, tokens=text.split(), confidences=[1.0] * len(text.split()))
    except Exception as exc:
        LOGGER.warning("pdfminer failed: %s", exc)

    images = pdf_to_images(pdf_path, dpi=DPI_PRIMARY)
    joined, tok, conf = "", [], []
    for img in images:
        res = _tesseract_ocr(img)
        joined += res.text + "\n"
        tok.extend(res.tokens)
        conf.extend(res.confidences)

    result = OcrResult(joined, tok, conf)
    if result.field_confidence >= TAU_FIELD_ACCEPT:
        LOGGER.info("tesseract pass accepted (%.2f)", result.field_confidence)
        return result

    LOGGER.info("Enhancer triggered – running 600 dpi…")
    images = pdf_to_images(pdf_path, dpi=DPI_ENHANCED)
    joined, tok, conf = "", [], []
    for img in images:
        res = _tesseract_ocr(img)
        joined += res.text + "\n"
        tok.extend(res.tokens)
        conf.extend(res.confidences)
    enhanced = OcrResult(joined, tok, conf)

    if enhanced.field_confidence >= TAU_ENHANCER_PASS:
        LOGGER.info("enhancer pass accepted (%.2f)", enhanced.field_confidence)
        return enhanced

    LOGGER.warning("Confidence still low (%.2f) – returning best effort", enhanced.field_confidence)
    return enhanced

# -----------------------------------------------------------------------------
# Field extraction
# -----------------------------------------------------------------------------
ENERGY_RE = re.compile(r"(\d[\d\s]{0,10})\s*k\s*W\s*h", re.I)
CARBON_RE = re.compile(r"Kg\s*CO2e\s*(\d[\d\s]{0,10})", re.I)

def _normalise_number(num_txt: str) -> int:
    digits = re.sub(r"\s+", "", num_txt)
    return int(digits)

def extract_fields(text: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    m = ENERGY_RE.search(text)
    if m:
        out["electricity_kwh"] = _normalise_number(m.group(1))

    m = CARBON_RE.search(text)
    if m:
        out["carbon_kgco2e"] = _normalise_number(m.group(1))
    return out

# -----------------------------------------------------------------------------
# Payload builder
# -----------------------------------------------------------------------------
def sha256(fp: Path) -> str:
    h = hashlib.sha256()
    with fp.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def build_payload(fields: Dict[str, Any], source_path: Path) -> Dict[str, Any]:
    return {
        "electricity": {
            "consumption": {
                "value": fields.get("electricity_kwh"),
                "unit": "kWh",
            }
        } if "electricity_kwh" in fields else {},
        "carbon": {
            "location_based": {
                "value": fields.get("carbon_kgco2e"),
                "unit": "kgCO2e",
            }
        } if "carbon_kgco2e" in fields else {},
        "source_document": {
            "file_name": source_path.name,
            "sha256": sha256(source_path),
        },
        "meta": {
            "extraction_confidence": fields.get("_confidence"),
        },
    }

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py path/to/bill.pdf [--save outfile.json]")
        sys.exit(1)

    bill_path = Path(sys.argv[1])
    save_path = None
    if "--save" in sys.argv:
        save_path = Path(sys.argv[sys.argv.index("--save") + 1])

    ocr_res = run_ocr(bill_path)
    fields = extract_fields(ocr_res.text)
    fields["_confidence"] = ocr_res.field_confidence

    payload = build_payload(fields, bill_path)

    if save_path:
        save_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {save_path}")

    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
