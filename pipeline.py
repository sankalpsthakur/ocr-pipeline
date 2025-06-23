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

# Direct execution: python pipeline.py
import config
from config import *

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None
try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None
try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None

try:
    import pytesseract
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None
try:
    from pdf2image import convert_from_path
except Exception:  # pragma: no cover - optional dependency
    convert_from_path = None
try:
    from pdfminer.high_level import extract_text
except Exception:  # pragma: no cover - optional dependency
    extract_text = None
try:
    import easyocr
except Exception:  # pragma: no cover - optional dependency
    easyocr = None

# Configuration imported from config.py

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

def _auto_rotate(img):
    """Rotate image according to Tesseract's OSD output."""
    try:
        osd = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
        rot = int(osd.get("rotate", 0))
        if rot:
            return img.rotate(-rot, expand=True)
    except Exception as exc:
        LOGGER.debug("orientation detection failed: %s", exc)
    return img


def preprocess_image(image):
    """Convert Pillow image to cleaned grayscale numpy array."""
    if np is None:
        return None  # noqa: E701 - preprocessing skipped

    arr = np.array(image)
    if cv2:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    # Fallback using Pillow only
    return np.array(image.convert("L"))


def _tesseract_ocr(image) -> OcrResult:
    if pytesseract is None:
        raise RuntimeError("pytesseract is not available")
    image = _auto_rotate(image)
    img = preprocess_image(image)
    if img is None:
        img = image
    config = f"--oem {TESSERACT_OEM} --psm {TESSERACT_PSM}"
    data = pytesseract.image_to_data(
        img,
        lang=TESSERACT_LANG,
        config=config,
        output_type=pytesseract.Output.DICT,
    )
    
    # Filter tokens: remove empty text and negative confidence
    filtered_tokens = []
    filtered_confs = []
    for token, conf in zip(data["text"], data["conf"]):
        if token.strip():  # Non-empty text
            try:
                conf_val = float(conf)
                if conf_val >= 0:  # Non-negative confidence
                    filtered_tokens.append(token)
                    filtered_confs.append(conf_val / 100.0)
            except ValueError:
                continue  # Skip invalid confidence values
    
    joined = " ".join(filtered_tokens)
    return OcrResult(text=joined, tokens=filtered_tokens, confidences=filtered_confs)

def _easyocr_ocr(image) -> OcrResult:
    if easyocr is None:
        raise RuntimeError("easyocr is not available")
    
    # Initialize EasyOCR reader (cached after first use)
    if not hasattr(_easyocr_ocr, "reader"):
        _easyocr_ocr.reader = easyocr.Reader(['en'], gpu=False)
    
    # Convert PIL image to numpy array
    if np is None:
        raise RuntimeError("numpy is required for EasyOCR")
    img_array = np.array(image)
    
    # Run EasyOCR
    results = _easyocr_ocr.reader.readtext(img_array, detail=1)
    
    tokens = []
    confidences = []
    text_parts = []
    
    for (bbox, text, conf) in results:
        if text.strip():
            tokens.append(text)
            confidences.append(float(conf))
            text_parts.append(text)
    
    joined = " ".join(text_parts)
    return OcrResult(text=joined, tokens=tokens, confidences=confidences)

def pdf_to_images(pdf_path: Path, dpi: int) -> List:
    """Convert PDF pages to images with a page cap for efficiency."""
    if convert_from_path is None:
        raise RuntimeError("pdf2image is not available")
    return convert_from_path(
        str(pdf_path),
        dpi=dpi,
        first_page=1,
        last_page=MAX_PAGES,
    )

def load_image(image_path: Path):
    """Load an image file using PIL."""
    if Image is None:
        raise RuntimeError("PIL is not available")
    return Image.open(str(image_path))

def _run_ocr_engine(file_path: Path, dpi: int = None, is_image: bool = False) -> OcrResult:
    """Run OCR using the configured backend engine."""
    if OCR_BACKEND == "tesseract":
        if is_image:
            # Process single image file
            img = load_image(file_path)
            return _tesseract_ocr(img)
        else:
            # Process PDF (existing logic)
            images = pdf_to_images(file_path, dpi=dpi)
            joined, tok, conf = "", [], []
            for img in images:
                res = _tesseract_ocr(img)
                joined += res.text + "\n"
                tok.extend(res.tokens)
                conf.extend(res.confidences)
            return OcrResult(joined, tok, conf)
    elif OCR_BACKEND == "easyocr":
        if is_image:
            # Process single image file
            img = load_image(file_path)
            return _easyocr_ocr(img)
        else:
            # Process PDF
            images = pdf_to_images(file_path, dpi=dpi)
            joined, tok, conf = "", [], []
            for img in images:
                res = _easyocr_ocr(img)
                joined += res.text + "\n"
                tok.extend(res.tokens)
                conf.extend(res.confidences)
            return OcrResult(joined, tok, conf)
    elif OCR_BACKEND == "gcv":
        # Google Cloud Vision implementation would go here
        raise NotImplementedError("Google Cloud Vision backend not implemented")
    elif OCR_BACKEND == "azure":
        # Azure Form Recognizer implementation would go here
        raise NotImplementedError("Azure Form Recognizer backend not implemented")
    else:
        raise ValueError(f"Unsupported OCR backend: {OCR_BACKEND}")

def run_ocr(file_path: Path) -> OcrResult:
    """Cascaded OCR: digital pass ➜ bitmap pass ➜ enhancer."""
    # Check if file is an image
    is_image = file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    
    if not is_image and extract_text:
        try:
            LOGGER.info("Running pdfminer (digital text pass)…")
            text = extract_text(str(file_path))
            if text and text.strip():
                return OcrResult(text=text, tokens=text.split(), confidences=[1.0] * len(text.split()))
        except Exception as exc:
            LOGGER.warning("pdfminer failed: %s", exc)

    # Primary OCR pass
    if is_image:
        LOGGER.info("Running primary OCR pass on image with %s backend…", OCR_BACKEND)
        result = _run_ocr_engine(file_path, is_image=True)
    else:
        LOGGER.info("Running primary OCR pass at %d dpi with %s backend…", DPI_PRIMARY, OCR_BACKEND)
        result = _run_ocr_engine(file_path, dpi=DPI_PRIMARY)
    
    if result.field_confidence >= TAU_FIELD_ACCEPT:
        LOGGER.info("Primary pass accepted (%.2f)", result.field_confidence)
        return result

    # Enhancement pass (for PDFs only)
    if not is_image:
        LOGGER.info("Enhancement triggered – running at %d dpi…", DPI_ENHANCED)
        enhanced = _run_ocr_engine(file_path, dpi=DPI_ENHANCED)
        
        if enhanced.field_confidence >= TAU_ENHANCER_PASS:
            LOGGER.info("Enhancement pass accepted (%.2f)", enhanced.field_confidence)
            return enhanced
        
        # LLM fallback if confidence is still too low
        if enhanced.field_confidence < TAU_LLM_PASS:
            LOGGER.warning("OCR confidence (%.2f) below LLM threshold (%.2f)", 
                          enhanced.field_confidence, TAU_LLM_PASS)
            LOGGER.info("LLM fallback could be implemented here")
        
        LOGGER.warning("Confidence still low (%.2f) – returning best effort", enhanced.field_confidence)
        return enhanced
    else:
        # For images, just return the result as enhancement is not applicable
        LOGGER.info("Image processing complete (%.2f)", result.field_confidence)
        return result

# -----------------------------------------------------------------------------
# Field extraction
# -----------------------------------------------------------------------------
ENERGY_RE = re.compile(r"(\d[\d\s,]{0,10})\s*k\s*W\s*h", re.I)
# Improved carbon regex to handle OCR errors like "coze", "C0Ze" instead of "CO2e" 
CARBON_RE = re.compile(r"Kg\s*(?:CO(?:2|\u2082)e|co(?:2|\u2082)e|coze|C0Ze)\s+(\d[\d\s,]{0,10})", re.I)
# Alternative carbon patterns - look for carbon footprint value (typically 2-4 digits)
CARBON_ALT_RE = re.compile(r"Kg\s*(?:CO(?:2|\u2082)?e?|co(?:2|\u2082)?e?|coze?|C0Ze?).*?(\d{2,4})(?=\s|$)", re.I | re.DOTALL)
# Simple pattern to find carbon footprint value - look for 3-digit number after "0.00" following Kg CO2e variants
CARBON_SIMPLE_RE = re.compile(r"Kg\s*(?:CO(?:2|\u2082)?e?|co(?:2|\u2082)?e?|coze?|C0Ze?).*?0\.00\s+(\d{3})", re.I | re.DOTALL)
CARBON_EMISSIONS_RE = re.compile(r"Carbon\s+emissions\s+in\s+Kg\s+CO2e.*?(\d{2,4})", re.I | re.DOTALL)


def _normalise_number(num_txt: str) -> int:
    digits = re.sub(r"[\s,]+", "", num_txt)
    return int(digits)


def extract_fields(text: str) -> Dict[str, int]:
    """Extracts electricity and carbon values from OCR text."""
    out: Dict[str, int] = {}
    m = ENERGY_RE.search(text)
    if m:
        out["electricity_kwh"] = _normalise_number(m.group(1))

    # Try carbon patterns in order of specificity
    patterns = [
        (CARBON_RE, "primary"),
        (CARBON_SIMPLE_RE, "simple"),
        (CARBON_ALT_RE, "alternative"), 
        (CARBON_EMISSIONS_RE, "emissions")
    ]
    
    for pattern, name in patterns:
        m = pattern.search(text)
        if m:
            carbon_value = _normalise_number(m.group(1))
            # Skip obviously wrong values like 0, 000, etc.
            if carbon_value > 10:  # Carbon footprint should be > 10 kg for typical usage
                out["carbon_kgco2e"] = carbon_value
                break

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
        print("Usage: python pipeline.py bill.[pdf|png] [--save outfile.json]")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    save_path = None
    if "--save" in sys.argv:
        save_path = Path(sys.argv[sys.argv.index("--save") + 1])

    # Check if file exists and has supported extension
    if not file_path.exists():
        print(f"Error: File '{file_path}' does not exist")
        sys.exit(1)
    
    supported_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    if file_path.suffix.lower() not in supported_extensions:
        print(f"Error: Unsupported file type '{file_path.suffix}'. Supported: {', '.join(supported_extensions)}")
        sys.exit(1)

    ocr_res = run_ocr(file_path)
    fields = extract_fields(ocr_res.text)
    fields["_confidence"] = ocr_res.field_confidence

    payload = build_payload(fields, file_path)

    if save_path:
        save_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {save_path}")

    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
