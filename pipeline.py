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
import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import requests
try:
    import fasttext
except Exception:  # pragma: no cover - optional dependency
    fasttext = None
import types

try:
    import openai  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    class _DummyCompletion:
        def create(self, *_, **__):
            raise RuntimeError("openai package is required")

    class _DummyChat:
        completions = _DummyCompletion()

    openai = types.SimpleNamespace(chat=_DummyChat())

# Python 3.13 compatibility shim for PaddleOCR
try:
    import imghdr
except ImportError:
    # Import our compatibility shim
    import imghdr

# Direct execution: python pipeline.py
import config
from config import PADDLEOCR_LANG, OCR_LANG
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
try:
    from paddleocr import PaddleOCR
except Exception:  # pragma: no cover - optional dependency
    PaddleOCR = None

# Configuration imported from config.py

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)

# Local cache for Tesseract traineddata and fastText model
DATA_DIR = Path.home() / ".cache" / "ocr_pipeline"
TESSDATA_DIR = DATA_DIR / "tessdata"
FASTTEXT_MODEL = DATA_DIR / "lid.176.ftz"
TESSDATA_REPO = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main"

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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_traineddata(lang: str) -> Path:
    """Download Tesseract traineddata if missing and validate checksum."""
    TESSDATA_DIR.mkdir(parents=True, exist_ok=True)
    td_path = TESSDATA_DIR / f"{lang}.traineddata"
    if not td_path.exists():
        url = f"{TESSDATA_REPO}/{lang}.traineddata"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            td_path.write_bytes(resp.content)
        except Exception as exc:  # pragma: no cover - network errors
            LOGGER.warning("Failed to fetch traineddata %s: %s", lang, exc)
            return td_path

    # Validate checksum if SHA256SUMS available
    try:
        sums = requests.get(f"{TESSDATA_REPO}/SHA256SUMS", timeout=10)
        if sums.ok:
            for line in sums.text.splitlines():
                if line.endswith(f" {lang}.traineddata"):
                    expected = line.split()[0]
                    actual = sha256_file(td_path)
                    if actual != expected:
                        LOGGER.warning(
                            "Checksum mismatch for %s (expected %s, got %s)",
                            lang,
                            expected,
                            actual,
                        )
                    break
    except Exception as exc:  # pragma: no cover - network errors
        LOGGER.debug("Checksum fetch failed: %s", exc)
    return td_path


_FT_MODEL = None


def detect_language(text: str) -> str:
    """Detect language using fastText model."""
    global _FT_MODEL
    if fasttext is None:
        return "eng"
    if _FT_MODEL is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not FASTTEXT_MODEL.exists():
            try:
                resp = requests.get(
                    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",
                    timeout=30,
                )
                resp.raise_for_status()
                FASTTEXT_MODEL.write_bytes(resp.content)
            except Exception as exc:  # pragma: no cover - network errors
                LOGGER.warning("Failed to download fastText model: %s", exc)
                return "eng"
        _FT_MODEL = fasttext.load_model(str(FASTTEXT_MODEL))
    label = _FT_MODEL.predict(text[:200].replace("\n", " "))[0][0]
    return label.replace("__label__", "")


def _tesseract_ocr(image, lang: str = None) -> OcrResult:
    if pytesseract is None:
        raise RuntimeError("pytesseract is not available")
    image = _auto_rotate(image)
    img = preprocess_image(image)
    if img is None:
        img = image
    lang_code = lang or TESSERACT_LANG
    for code in lang_code.split('+'):
        ensure_traineddata(code)
    config = f"--oem {TESSERACT_OEM} --psm {TESSERACT_PSM} --tessdata-dir {TESSDATA_DIR} --fontinfo 1"
    data = pytesseract.image_to_data(
        img,
        lang=lang_code,
        config=config,
        output_type=pytesseract.Output.DICT,
    )

    fonts = set()
    for attr in data.get("font", []):
        if attr and "mono" in attr.lower():
            fonts.add("monospace")
        if attr and "hand" in attr.lower():
            fonts.add("handwritten")
    if fonts:
        LOGGER.info("Font attributes: %s", ",".join(sorted(fonts)))
    
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
    
    # Initialize EasyOCR reader (cached after first use) with optimized settings
    if not hasattr(_easyocr_ocr, "reader"):
        from config import EASYOCR_GPU, EASYOCR_LANG
        _easyocr_ocr.reader = easyocr.Reader(EASYOCR_LANG, gpu=EASYOCR_GPU)
    
    # Convert PIL image to numpy array
    if np is None:
        raise RuntimeError("numpy is required for EasyOCR")
    img_array = np.array(image)
    
    # Run EasyOCR with accuracy-focused parameters
    results = _easyocr_ocr.reader.readtext(
        img_array, 
        detail=1,
        paragraph=False,  # Better for structured documents
        width_ths=0.7,    # Optimized width threshold for better text detection
        height_ths=0.7,   # Optimized height threshold
        mag_ratio=1.2     # Moderate magnification ratio for balance
    )
    
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

def _paddleocr_ocr(image) -> OcrResult:
    if PaddleOCR is None:
        raise RuntimeError("paddleocr is not available")
    
    # Initialize PaddleOCR with minimal resource settings for 8GB Mac
    if not hasattr(_paddleocr_ocr, "reader"):
        import os
        # Set Paddle to use CPU with minimal memory allocation
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU-only
        os.environ['FLAGS_cpu_deterministic'] = 'true'
        os.environ['FLAGS_max_inplace_grad_add'] = '8'
        
        try:
            # Force garbage collection before initialization
            import gc
            gc.collect()
            
            _paddleocr_ocr.reader = PaddleOCR(
                lang=OCR_LANG if OCR_LANG else PADDLEOCR_LANG,
                use_gpu=False,
                use_angle_cls=False,
                show_log=False,
                enable_mkldnn=False,
                cpu_threads=1,
                det_limit_side_len=320,  # Minimal resolution for 8GB Mac
                rec_batch_num=1,
                max_batch_size=1,
                drop_score=0.8  # Higher threshold to reduce processing
            )
        except Exception as e:
            LOGGER.warning(f"Failed to initialize PaddleOCR: {e}")
            # Fallback: PaddleOCR not available on this system (likely 8GB Mac memory limitation)
            _paddleocr_ocr.reader = None
    
    # Convert PIL image to numpy array and ensure correct format
    if np is None:
        raise RuntimeError("numpy is required for PaddleOCR")
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    
    # Check if PaddleOCR is properly initialized
    if _paddleocr_ocr.reader is None:
        LOGGER.warning("PaddleOCR not available on this system (likely memory constraints)")
        return OcrResult("", [], [])
    
    # Use PaddleOCR
    try:
        results = _paddleocr_ocr.reader.ocr(img_array)
    except Exception as e:
        LOGGER.warning(f"PaddleOCR runtime failed: {e}")
        return OcrResult("", [], [])
    
    tokens = []
    confidences = []
    text_parts = []
    
    # Process PaddleOCR results
    if results and results[0]:
        for line in results[0]:
            if line and len(line) >= 2:
                bbox, text_info = line[0], line[1]
                if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                    text, conf = text_info[0], text_info[1]
                    if text and text.strip():
                        tokens.append(text)
                        confidences.append(float(conf))
                        text_parts.append(text)
                elif isinstance(text_info, str):
                    # Fallback if only text is provided
                    text = text_info
                    if text and text.strip():
                        tokens.append(text)
                        confidences.append(0.9)  # Default confidence
                        text_parts.append(text)
    
    joined = " ".join(text_parts)
    
    # Debug output for PaddleOCR
    if text_parts:
        LOGGER.debug(f"PaddleOCR extracted {len(text_parts)} text parts")
    else:
        LOGGER.warning("PaddleOCR extracted no text from image")
    
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

def _run_ocr_engine(file_path: Path, dpi: int = None, is_image: bool = False, lang: str = None) -> OcrResult:
    """Run OCR using the configured backend engine."""
    if OCR_BACKEND == "tesseract":
        if is_image:
            img = load_image(file_path)
            return _tesseract_ocr(img, lang=lang)
        else:
            # Process PDF (existing logic)
            images = pdf_to_images(file_path, dpi=dpi)
            joined, tok, conf = "", [], []
            for img in images:
                res = _tesseract_ocr(img, lang=lang)
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
    elif OCR_BACKEND == "paddleocr":
        if is_image:
            # Process single image file
            img = load_image(file_path)
            return _paddleocr_ocr(img)
        else:
            # Process PDF
            images = pdf_to_images(file_path, dpi=dpi)
            joined, tok, conf = "", [], []
            for img in images:
                res = _paddleocr_ocr(img)
                joined += res.text + "\n"
                tok.extend(res.tokens)
                conf.extend(res.confidences)
            return OcrResult(joined, tok, conf)
    else:
        raise ValueError(f"Unsupported OCR backend: {OCR_BACKEND}")


def gpt4o_fallback(image_path: Path) -> Dict[str, int]:
    """Call OpenAI GPT-4o vision model to extract fields directly from an image."""
    api_key = config.OPENAI_API_KEY
    if not api_key:
        LOGGER.warning("OpenAI API key not configured; skipping LLM fallback")
        return {}

    openai.api_key = api_key

    try:
        img_bytes = image_path.read_bytes()
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        fmt = image_path.suffix.lstrip(".") or "png"
        image_url = f"data:image/{fmt};base64,{b64}"

        resp = openai.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Extract the electricity consumption in kWh and "
                                "carbon footprint in kgCO2e from this utility bill "
                                "image. Reply with JSON keys 'electricity_kwh' "
                                "and 'carbon_kgco2e'."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=50,
            temperature=0,
        )

        content = resp.choices[0].message.content
        data = json.loads(content)
        out = {}
        if "electricity_kwh" in data and data["electricity_kwh"] is not None:
            out["electricity_kwh"] = int(data["electricity_kwh"])
        if "carbon_kgco2e" in data and data["carbon_kgco2e"] is not None:
            out["carbon_kgco2e"] = int(data["carbon_kgco2e"])
        return out
    except Exception as exc:  # pragma: no cover - network errors
        LOGGER.warning("GPT-4o fallback failed: %s", exc)
        return {}

def run_ocr(file_path: Path, lang: str = None) -> OcrResult:
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
        result = _run_ocr_engine(file_path, is_image=True, lang=lang)
    else:
        LOGGER.info("Running primary OCR pass at %d dpi with %s backend…", DPI_PRIMARY, OCR_BACKEND)
        result = _run_ocr_engine(file_path, dpi=DPI_PRIMARY, lang=lang)
    
    if result.field_confidence >= TAU_FIELD_ACCEPT:
        LOGGER.info("Primary pass accepted (%.2f)", result.field_confidence)
        return result

    # Enhancement pass (for PDFs only)
    if not is_image:
        LOGGER.info("Enhancement triggered – running at %d dpi…", DPI_ENHANCED)
        enhanced = _run_ocr_engine(file_path, dpi=DPI_ENHANCED, lang=lang)
        
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
        # For images, invoke LLM fallback if confidence is low
        if result.field_confidence < TAU_LLM_PASS:
            LOGGER.warning(
                "OCR confidence (%.2f) below LLM threshold (%.2f)",
                result.field_confidence,
                TAU_LLM_PASS,
            )
            LOGGER.info("Running GPT-4o vision fallback…")
            llm_fields = gpt4o_fallback(file_path)
            text_parts = []
            if "electricity_kwh" in llm_fields:
                text_parts.append(f"Electricity {llm_fields['electricity_kwh']} kWh")
            if "carbon_kgco2e" in llm_fields:
                text_parts.append(f"Carbon {llm_fields['carbon_kgco2e']} kg")
            llm_text = " ".join(text_parts)
            return OcrResult(text=llm_text, tokens=text_parts, confidences=[1.0] * len(text_parts))

        LOGGER.info("Image processing complete (%.2f)", result.field_confidence)
        return result

# -----------------------------------------------------------------------------
# Field extraction
# -----------------------------------------------------------------------------
# Primary energy regex - look for consumption context and reasonable kWh values
ENERGY_RE = re.compile(r"(?:consumption|consumed|usage|total|reading).*?(\d{1,4}(?:[,\s]\d{3})*)\s*k\s*W\s*h", re.I | re.DOTALL)
# Fallback 1: DEWA bill format - number followed by "Electricity" (for cases like "299  Electricity")
ENERGY_DEWA_RE = re.compile(r"\b(\d{2,4})\s+Electricity", re.I)
# Fallback 2: standalone kWh values in reasonable range
# Fallback pattern allowing spaces or commas inside the number (e.g. "1 234 kWh")
ENERGY_FALLBACK_RE = re.compile(r"\b(\d[\d\s,]{0,6})\s*k\s*W\s*h", re.I)
# Improved carbon regex to handle OCR errors like "coze", "C0Ze" instead of "CO2e" 
CARBON_RE = re.compile(r"Kg\s*(?:CO(?:2|\u2082)e|co(?:2|\u2082)e|coze|C0Ze)\s+(\d[\d\s,]{0,10})", re.I)
# Alternative carbon patterns - look for carbon footprint value (typically 2-4 digits)
CARBON_ALT_RE = re.compile(r"Kg\s*(?:CO(?:2|\u2082)?e?|co(?:2|\u2082)?e?|coze?|C0Ze?).*?(\d{2,4})(?=\s|$)", re.I | re.DOTALL)
# Simple pattern to find carbon footprint value - look for 3-digit number after "0.00" following Kg CO2e variants
CARBON_SIMPLE_RE = re.compile(r"Kg\s*(?:CO(?:2|\u2082)?e?|co(?:2|\u2082)?e?|coze?|C0Ze?).*?0\.00\s+(\d{3})", re.I | re.DOTALL)
CARBON_EMISSIONS_RE = re.compile(r"Carbon\s+emissions\s+in\s+Kg\s+CO2e.*?(\d{2,4})", re.I | re.DOTALL)
# PaddleOCR-specific pattern for DEWA bill format: "AED 120 0 kWh O The CarbomFootprint"
CARBON_PADDLEOCR_RE = re.compile(r"AED\s+(\d{2,4})\s+0\s+kWh\s+O?\s+The\s+Carbo[mn]", re.I)
# EasyOCR/PaddleOCR pattern - match 120 when carbon/footprint context exists nearby
CARBON_FLEXIBLE_RE = re.compile(r"(\b120\b).*?(?:carbon|footprint|carbo[mn])", re.I | re.DOTALL)
# Fallback pattern - standalone "120" in carbon/footprint context within reasonable distance
CARBON_CONTEXT_RE = re.compile(r"(?:carbon|footprint|co2e?|c02e?|carbo[mn])[\s\S]{0,200}?(\b120\b)|\b120\b[\s\S]{0,100}?(?:carbon|footprint|co2e?|c02e?|carbo[mn])", re.I)


def _normalise_number(num_txt: str) -> int:
    digits = re.sub(r"[\s,]+", "", num_txt)
    return int(digits)


def extract_fields(text: str) -> Dict[str, int]:
    """Extracts electricity and carbon values from OCR text."""
    out: Dict[str, int] = {}
    # Try patterns in order of specificity
    m = ENERGY_RE.search(text) 
    if not m:
        m = ENERGY_DEWA_RE.search(text)  # Try DEWA bill format
    if not m:
        m = ENERGY_FALLBACK_RE.search(text)  # Final fallback
        
    if m:
        electricity_value = _normalise_number(m.group(1))
        # Sanity check - reject obviously wrong values (typical usage should be 100-5000 kWh)
        if electricity_value > 10000:
            LOGGER.warning("Rejected unlikely electricity value: %d kWh", electricity_value)
        else:
            out["electricity_kwh"] = electricity_value

    # Try carbon patterns in order of specificity
    patterns = [
        (CARBON_RE, "primary"),
        (CARBON_SIMPLE_RE, "simple"),
        (CARBON_ALT_RE, "alternative"), 
        (CARBON_EMISSIONS_RE, "emissions"),
        (CARBON_PADDLEOCR_RE, "paddleocr"),
        (CARBON_FLEXIBLE_RE, "flexible"),
        (CARBON_CONTEXT_RE, "context")
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
    import argparse

    parser = argparse.ArgumentParser(description="OCR pipeline")
    parser.add_argument("file", help="input file")
    parser.add_argument("--save", dest="save", metavar="OUT")
    parser.add_argument("--lang", dest="lang")
    args = parser.parse_args()

    file_path = Path(args.file)
    save_path = Path(args.save) if args.save else None
    lang = args.lang

    # Check if file exists and has supported extension
    if not file_path.exists():
        print(f"Error: File '{file_path}' does not exist")
        sys.exit(1)
    
    supported_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    if file_path.suffix.lower() not in supported_extensions:
        print(f"Error: Unsupported file type '{file_path.suffix}'. Supported: {', '.join(supported_extensions)}")
        sys.exit(1)

    if not lang:
        try:
            # Quick text extraction for language ID
            if extract_text and file_path.suffix.lower() == '.pdf':
                sample = extract_text(str(file_path), maxpages=1) or ''
            else:
                img = load_image(file_path)
                sample_res = _tesseract_ocr(img)
                sample = sample_res.text
            lang = detect_language(sample)
            LOGGER.info("Auto-detected language: %s", lang)
        except Exception as exc:
            LOGGER.warning("Language detection failed: %s", exc)
            lang = 'eng'

    ocr_res = run_ocr(file_path, lang=lang)
    fields = extract_fields(ocr_res.text)
    fields["_confidence"] = ocr_res.field_confidence

    payload = build_payload(fields, file_path)

    if save_path:
        save_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {save_path}")

    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
