
import io, os, subprocess, tempfile, json, math, logging
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import pytesseract
from pdf2image import convert_from_path
from pdfminer.high_level import extract_text

from . import config

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

@dataclass
class OcrResult:
    text: str
    tokens: List[str]
    confidences: List[float]

    @property
    def field_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        # geometric mean to penalise low tokens
        product = math.prod([max(c, 1e-3) for c in self.confidences])
        return product ** (1 / len(self.confidences))

# --------------------------------------------------------------------------- #
#                               Helper utils                                  #
# --------------------------------------------------------------------------- #
def _tesseract_ocr(image) -> OcrResult:
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    tokens = data["text"]
    confs  = [float(c) / 100.0 for c in data["conf"]]
    joined = " ".join(tokens)
    return OcrResult(text=joined, tokens=tokens, confidences=confs)

def pdf_to_images(pdf_path: Path, dpi: int) -> List:
    return convert_from_path(str(pdf_path), dpi=dpi)

# --------------------------------------------------------------------------- #
#                               Main entry                                    #
# --------------------------------------------------------------------------- #
def run_ocr(pdf_path: Path) -> OcrResult:
    """Cascaded OCR: digital pass ➜ bitmap pass ➜ enhancer."""
    # 1. Try digital text extraction
    try:
        LOGGER.info("Running pdfminer (digital text pass)…")
        text = extract_text(str(pdf_path))
        if text and text.strip():
            # pdfminer has no confidences; assume perfect
            return OcrResult(text=text, tokens=text.split(), confidences=[1.0] * len(text.split()))
    except Exception as exc:
        LOGGER.warning("pdfminer failed: %s", exc)

    # 2. Tesseract 300‑dpi
    images = pdf_to_images(pdf_path, dpi=config.DPI_PRIMARY)
    joined, tok, conf = "", [], []
    for img in images:
        res = _tesseract_ocr(img)
        joined += res.text + "\n"
        tok.extend(res.tokens); conf.extend(res.confidences)

    result = OcrResult(joined, tok, conf)
    if result.field_confidence >= config.TAU_FIELD_ACCEPT:
        LOGGER.info("tesseract pass accepted (%.2f)", result.field_confidence)
        return result

    # 3. Enhancer: 600‑dpi
    LOGGER.info("Enhancer triggered – running 600 dpi…")
    images = pdf_to_images(pdf_path, dpi=config.DPI_ENHANCED)
    joined, tok, conf = "", [], []
    for img in images:
        res = _tesseract_ocr(img)
        joined += res.text + "\n"
        tok.extend(res.tokens); conf.extend(res.confidences)
    enhanced = OcrResult(joined, tok, conf)

    if enhanced.field_confidence >= config.TAU_ENHANCER_PASS:
        LOGGER.info("enhancer pass accepted (%.2f)", enhanced.field_confidence)
        return enhanced

    # 4. Placeholder for Google Vision or LLM call
    LOGGER.warning("Confidence still low (%.2f) – returning best effort", enhanced.field_confidence)
    return enhanced
