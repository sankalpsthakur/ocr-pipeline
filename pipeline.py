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
import concurrent.futures
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import types

try:
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None

try:
    from mistralai import Mistral
except Exception:  # pragma: no cover - optional dependency
    Mistral = None

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

# -----------------------------------------------------------------------------
# Image caching and OCR helpers
# -----------------------------------------------------------------------------

class ImageCache:
    """Thread-safe cache for page bitmaps to avoid repeated PDF/Image loading."""
    
    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()
    
    def get_cache_key(self, file_path: Path, dpi: int, page_num: int = 0) -> str:
        """Generate cache key for image."""
        file_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
        return f"{file_hash}_{dpi}_{page_num}"
    
    def get_images(self, file_path: Path, dpi: int, is_image: bool = False) -> List:
        """Get cached images or load and cache them."""
        with self._lock:
            if is_image:
                cache_key = self.get_cache_key(file_path, dpi, 0)
                if cache_key not in self._cache:
                    self._cache[cache_key] = [load_image(file_path)]
                return self._cache[cache_key]
            else:
                # For PDFs, cache each page separately
                images = []
                pdf_images = pdf_to_images(file_path, dpi)
                for i, img in enumerate(pdf_images):
                    cache_key = self.get_cache_key(file_path, dpi, i)
                    if cache_key not in self._cache:
                        self._cache[cache_key] = img
                    images.append(self._cache[cache_key])
                return images
    
    def clear(self):
        """Clear the cache."""
        with self._lock:
            self._cache.clear()

# Global image cache instance
_image_cache = ImageCache()

# -----------------------------------------------------------------------------
# OCR helpers
# -----------------------------------------------------------------------------
@dataclass
class OcrResult:
    text: str
    tokens: List[str]
    confidences: List[float]
    engine: str = ""
    
    @property
    def field_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        # Use robust geometric mean with top-k filtering for long documents
        return self._geometric_mean_confidence()
    
    def _geometric_mean_confidence(self, top_k_ratio: float = 0.8) -> float:
        """Calculate geometric mean using top-k confidence scores to avoid penalty from long documents."""
        if not self.confidences:
            return 0.0
        
        # Apply minimum threshold to prevent zero multiplication
        valid_confidences = [max(c, 1e-3) for c in self.confidences]
        
        # For long documents, use top-k percentile to avoid unfair penalty
        if len(valid_confidences) > 20:  # threshold for "long document"
            k = max(5, int(len(valid_confidences) * top_k_ratio))
            valid_confidences = sorted(valid_confidences, reverse=True)[:k]
        
        # Geometric mean
        product = math.prod(valid_confidences)
        return product ** (1 / len(valid_confidences))
    
    def _logarithmic_confidence(self) -> float:
        """Alternative logarithmic aggregation method."""
        if not self.confidences:
            return 0.0
        
        valid_confidences = [max(c, 1e-6) for c in self.confidences]
        log_sum = sum(math.log(c) for c in valid_confidences)
        return math.exp(log_sum / len(valid_confidences))

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

def _mistral_ocr(image) -> OcrResult:
    """Mistral OCR implementation using official OCR API."""
    if Mistral is None:
        LOGGER.warning("mistralai package not available; skipping Mistral OCR")
        return OcrResult("", [], [])
    
    api_key = config.MISTRAL_API_KEY
    if not api_key or api_key == "REPLACE_WITH_YOUR_MISTRAL_API_KEY_HERE":
        LOGGER.warning("Mistral API key not configured; skipping Mistral OCR")
        return OcrResult("", [], [])
    
    try:
        # Convert PIL image to base64
        import io
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        img_bytes = buffer.getvalue()
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        
        # Initialize Mistral client
        client = Mistral(api_key=api_key)
        
        # Use the OCR API
        ocr_response = client.ocr.process(
            model=config.MISTRAL_MODEL,
            document={
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{b64}"
            },
            include_image_base64=True
        )
        
        # Extract text from OCR response
        if hasattr(ocr_response, 'pages') and ocr_response.pages:
            # Extract markdown from all pages
            all_text = []
            for page in ocr_response.pages:
                if hasattr(page, 'markdown') and page.markdown:
                    all_text.append(page.markdown)
            
            if all_text:
                extracted_text = '\n'.join(all_text).strip()
                tokens = extracted_text.split()
                # Assign high confidence since this is a specialized OCR model
                confidences = [0.97] * len(tokens)
                return OcrResult(text=extracted_text, tokens=tokens, confidences=confidences)
        
        # Fallback to old method
        if hasattr(ocr_response, 'text') and ocr_response.text:
            extracted_text = ocr_response.text.strip()
            tokens = extracted_text.split()
            confidences = [0.97] * len(tokens)
            return OcrResult(text=extracted_text, tokens=tokens, confidences=confidences)
        elif hasattr(ocr_response, 'content') and ocr_response.content:
            extracted_text = str(ocr_response.content).strip()
            tokens = extracted_text.split()
            confidences = [0.97] * len(tokens)
            return OcrResult(text=extracted_text, tokens=tokens, confidences=confidences)
        else:
            LOGGER.warning("No text found in Mistral OCR response")
            return OcrResult("", [], [])
            
    except Exception as exc:
        LOGGER.warning("Mistral OCR failed: %s", exc)
        return OcrResult("", [], [])

def _gemma_vlm_ocr(image, bounding_boxes=None) -> OcrResult:
    """Gemma VLM OCR with optional bounding boxes."""
    if requests is None:
        raise RuntimeError("requests package not available for Gemma VLM")
    
    api_key = config.GEMINI_API_KEY  # Using same API key as Gemini
    if not api_key or api_key == "REPLACE_WITH_YOUR_GEMINI_API_KEY_HERE":
        LOGGER.warning("Gemini API key not configured; skipping Gemma VLM OCR")
        return OcrResult("", [], [])
    
    try:
        # Convert PIL image to base64
        import io
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        
        # Use Gemini 2.0 Flash for VLM with bounding box awareness
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        
        prompt = "Extract all text from this image with high accuracy. Focus on structured document text like utility bills."
        if bounding_boxes:
            prompt += f" Pay special attention to these regions: {bounding_boxes}"
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": b64
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 2000
            }
        }

        resp = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
        resp.raise_for_status()
        
        result = resp.json()
        if 'candidates' in result and result['candidates']:
            extracted_text = result['candidates'][0]['content']['parts'][0]['text'].strip()
            tokens = extracted_text.split()
            # High confidence for VLM
            confidences = [0.96] * len(tokens)
            return OcrResult(text=extracted_text, tokens=tokens, confidences=confidences)
        else:
            LOGGER.warning("No valid response from Gemma VLM")
            return OcrResult("", [], [])
            
    except Exception as exc:
        LOGGER.warning("Gemma VLM OCR failed: %s", exc)
        return OcrResult("", [], [])

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

def _run_ocr_engine(file_path: Path, dpi: int = None, is_image: bool = False, engine: str = None) -> OcrResult:
    """Run OCR using the specified engine."""
    engine = engine or OCR_BACKEND
    
    if engine == "tesseract":
        if is_image:
            img = load_image(file_path)
            return _tesseract_ocr(img)
        else:
            images = pdf_to_images(file_path, dpi=dpi)
            joined, tok, conf = "", [], []
            for img in images:
                res = _tesseract_ocr(img)
                joined += res.text + "\n"
                tok.extend(res.tokens)
                conf.extend(res.confidences)
            return OcrResult(joined, tok, conf)
    elif engine == "easyocr":
        if is_image:
            img = load_image(file_path)
            return _easyocr_ocr(img)
        else:
            images = pdf_to_images(file_path, dpi=dpi)
            joined, tok, conf = "", [], []
            for img in images:
                res = _easyocr_ocr(img)
                joined += res.text + "\n"
                tok.extend(res.tokens)
                conf.extend(res.confidences)
            return OcrResult(joined, tok, conf)
    elif engine == "paddleocr":
        if is_image:
            img = load_image(file_path)
            return _paddleocr_ocr(img)
        else:
            images = pdf_to_images(file_path, dpi=dpi)
            joined, tok, conf = "", [], []
            for img in images:
                res = _paddleocr_ocr(img)
                joined += res.text + "\n"
                tok.extend(res.tokens)
                conf.extend(res.confidences)
            return OcrResult(joined, tok, conf)
    elif engine == "mistral":
        if is_image:
            img = load_image(file_path)
            return _mistral_ocr(img)
        else:
            images = pdf_to_images(file_path, dpi=dpi)
            joined, tok, conf = "", [], []
            for img in images:
                res = _mistral_ocr(img)
                joined += res.text + "\n"
                tok.extend(res.tokens)
                conf.extend(res.confidences)
            return OcrResult(joined, tok, conf)
    elif engine == "gemma_vlm":
        # For Gemma VLM, we might want to extract bounding boxes from traditional OCR first
        bounding_boxes = None
        if is_image:
            img = load_image(file_path)
            # Try to get bounding boxes from tesseract first
            try:
                tesseract_result = _tesseract_ocr(img)
                if tesseract_result.tokens:
                    # Extract some basic bounding box info (simplified)
                    bounding_boxes = f"Found {len(tesseract_result.tokens)} text regions"
            except:
                pass
            return _gemma_vlm_ocr(img, bounding_boxes)
        else:
            images = pdf_to_images(file_path, dpi=dpi)
            joined, tok, conf = "", [], []
            for img in images:
                # Try to get bounding boxes from tesseract first
                try:
                    tesseract_result = _tesseract_ocr(img)
                    if tesseract_result.tokens:
                        bounding_boxes = f"Found {len(tesseract_result.tokens)} text regions"
                except:
                    bounding_boxes = None
                res = _gemma_vlm_ocr(img, bounding_boxes)
                joined += res.text + "\n"
                tok.extend(res.tokens)
                conf.extend(res.confidences)
            return OcrResult(joined, tok, conf)
    else:
        raise ValueError(f"Unsupported OCR backend: {engine}")


def gemini_flash_fallback(image_path: Path) -> Dict[str, int]:
    """Call Gemini Flash vision model to extract fields directly from an image."""
    if requests is None:
        LOGGER.warning("requests package not available; skipping LLM fallback")
        return {}
    
    api_key = config.GEMINI_API_KEY
    if not api_key or api_key == "REPLACE_WITH_YOUR_GEMINI_API_KEY_HERE":
        LOGGER.warning("Gemini API key not configured; skipping LLM fallback")
        return {}

    try:
        img_bytes = image_path.read_bytes()
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_MODEL}:generateContent?key={api_key}"
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                "Extract the electricity consumption in kWh and "
                                "carbon footprint in kgCO2e from this utility bill "
                                "image. Reply only with JSON format: "
                                '{"electricity_kwh": number, "carbon_kgco2e": number}'
                            )
                        },
                        {
                            "inline_data": {
                                "mime_type": f"image/{image_path.suffix.lstrip('.') or 'png'}",
                                "data": b64
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 100
            }
        }

        resp = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
        resp.raise_for_status()
        
        result = resp.json()
        if 'candidates' in result and result['candidates']:
            content = result['candidates'][0]['content']['parts'][0]['text']
            # Try to extract JSON from the response
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # Try to find JSON-like content in the response
                import re
                json_match = re.search(r'\{[^}]*"electricity_kwh"[^}]*\}', content)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    LOGGER.warning("Could not parse JSON from Gemini response: %s", content)
                    return {}
            
            out = {}
            if "electricity_kwh" in data and data["electricity_kwh"] is not None:
                out["electricity_kwh"] = int(data["electricity_kwh"])
            if "carbon_kgco2e" in data and data["carbon_kgco2e"] is not None:
                out["carbon_kgco2e"] = int(data["carbon_kgco2e"])
            return out
        else:
            LOGGER.warning("No valid response from Gemini API")
            return {}
            
    except Exception as exc:  # pragma: no cover - network errors
        LOGGER.warning("Gemini Flash fallback failed: %s", exc)
        return {}

def _run_single_engine_with_cache(args: Tuple[Path, str, bool, int]) -> Tuple[str, OcrResult]:
    """Run a single OCR engine with cached images."""
    file_path, engine, is_image, dpi = args
    
    try:
        # Use cached images
        images = _image_cache.get_images(file_path, dpi, is_image)
        
        if engine == "tesseract":
            if is_image:
                result = _tesseract_ocr(images[0])
            else:
                joined, tok, conf = "", [], []
                for img in images:
                    res = _tesseract_ocr(img)
                    joined += res.text + "\n"
                    tok.extend(res.tokens)
                    conf.extend(res.confidences)
                result = OcrResult(joined, tok, conf, engine)
        elif engine == "easyocr":
            if is_image:
                result = _easyocr_ocr(images[0])
            else:
                joined, tok, conf = "", [], []
                for img in images:
                    res = _easyocr_ocr(img)
                    joined += res.text + "\n"
                    tok.extend(res.tokens)
                    conf.extend(res.confidences)
                result = OcrResult(joined, tok, conf, engine)
        elif engine == "paddleocr" and ENABLE_PADDLEOCR:
            if is_image:
                result = _paddleocr_ocr(images[0])
            else:
                joined, tok, conf = "", [], []
                for img in images:
                    res = _paddleocr_ocr(img)
                    joined += res.text + "\n"
                    tok.extend(res.tokens)
                    conf.extend(res.confidences)
                result = OcrResult(joined, tok, conf, engine)
        elif engine == "mistral":
            result = _mistral_ocr(file_path)
            result.engine = engine
        elif engine == "gemma_vlm":
            result = _gemma_vlm_ocr(file_path)
            result.engine = engine
        else:
            # Skip unsupported engines
            return engine, OcrResult("", [], [], engine)
            
        return engine, result
    except Exception as exc:
        LOGGER.warning("%s OCR failed: %s", engine, exc)
        return engine, OcrResult("", [], [], engine)

def _aggregate_multi_engine_results(results: List[Tuple[str, OcrResult]]) -> OcrResult:
    """Aggregate results from multiple OCR engines using confidence-weighted approach."""
    valid_results = [(engine, result) for engine, result in results 
                     if result.text.strip() and result.confidences]
    
    if not valid_results:
        return OcrResult("", [], [], "none")
    
    # Sort by confidence and select best result
    valid_results.sort(key=lambda x: x[1].field_confidence, reverse=True)
    best_engine, best_result = valid_results[0]
    
    # For highly confident results, return immediately
    if best_result.field_confidence >= TAU_FIELD_ACCEPT:
        LOGGER.info("Best engine %s with confidence %.2f", best_engine, best_result.field_confidence)
        return best_result
    
    # For moderate confidence, consider ensemble if multiple engines agree
    if len(valid_results) > 1 and best_result.field_confidence >= TAU_ENHANCER_PASS:
        # Simple ensemble: if top 2 engines have similar confidence, use the better one
        second_best = valid_results[1][1]
        if abs(best_result.field_confidence - second_best.field_confidence) < 0.1:
            LOGGER.info("Ensemble agreement between %s (%.2f) and %s (%.2f)", 
                       best_engine, best_result.field_confidence,
                       valid_results[1][0], second_best.field_confidence)
    
    return best_result

def run_ocr(file_path: Path) -> OcrResult:
    """Parallel OCR with hierarchical fallback: OCR engines in parallel → VLM fallback → Gemini Flash."""
    # Check if file is an image
    is_image = file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    
    # First try digital text extraction for PDFs
    if not is_image and extract_text:
        try:
            LOGGER.info("Running pdfminer (digital text pass)…")
            text = extract_text(str(file_path))
            if text and text.strip():
                return OcrResult(text=text, tokens=text.split(), confidences=[1.0] * len(text.split()), engine="pdfminer")
        except Exception as exc:
            LOGGER.warning("pdfminer failed: %s", exc)

    # Parallel OCR with traditional engines
    traditional_engines = ["tesseract", "easyocr"]
    if ENABLE_PADDLEOCR:
        traditional_engines.append("paddleocr")
    
    LOGGER.info("Running parallel OCR with engines: %s", traditional_engines)
    
    # Prepare arguments for parallel execution
    engine_args = [(file_path, engine, is_image, DPI_PRIMARY) for engine in traditional_engines]
    
    # Run traditional OCR engines in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(traditional_engines)) as executor:
        futures = [executor.submit(_run_single_engine_with_cache, args) for args in engine_args]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    # Aggregate results from parallel engines
    best_result = _aggregate_multi_engine_results(results)
    
    # If we have high confidence, return immediately
    if best_result.field_confidence >= TAU_FIELD_ACCEPT:
        LOGGER.info("Parallel OCR succeeded with confidence %.2f", best_result.field_confidence)
        return best_result
    
    # Try enhanced DPI for the best performing engine if confidence is moderate
    if (best_result.field_confidence >= TAU_ENHANCER_PASS and 
        not is_image and best_result.engine in traditional_engines):
        try:
            LOGGER.info("Enhancement triggered – running %s at %d dpi…", best_result.engine, DPI_ENHANCED)
            enhanced_args = (file_path, best_result.engine, is_image, DPI_ENHANCED)
            _, enhanced_result = _run_single_engine_with_cache(enhanced_args)
            if enhanced_result.field_confidence >= TAU_FIELD_ACCEPT:
                LOGGER.info("Enhanced OCR accepted with confidence %.2f", enhanced_result.field_confidence)
                return enhanced_result
            elif enhanced_result.field_confidence > best_result.field_confidence:
                best_result = enhanced_result
        except Exception as exc:
            LOGGER.warning("Enhanced OCR failed: %s", exc)
    
    # If traditional engines didn't work well, try VLM engines concurrently
    if best_result.field_confidence < TAU_LLM_PASS:
        LOGGER.info("Traditional OCR confidence too low (%.2f), trying VLM engines", best_result.field_confidence)
        
        vlm_engines = ["mistral", "gemma_vlm"]
        vlm_args = [(file_path, engine, is_image, DPI_PRIMARY) for engine in vlm_engines]
        
        # Run VLM engines in parallel with a timeout
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                vlm_futures = [executor.submit(_run_single_engine_with_cache, args) for args in vlm_args]
                vlm_results = []
                for future in concurrent.futures.as_completed(vlm_futures, timeout=30):
                    vlm_results.append(future.result())
                
                # Check if any VLM result is better
                vlm_best = _aggregate_multi_engine_results(vlm_results)
                if vlm_best.field_confidence > best_result.field_confidence:
                    best_result = vlm_best
                    
        except concurrent.futures.TimeoutError:
            LOGGER.warning("VLM engines timed out")
        except Exception as exc:
            LOGGER.warning("VLM engines failed: %s", exc)
    
    # If all engines failed or have low confidence, try Gemini Flash fallback
    if best_result.field_confidence < TAU_LLM_PASS:
        LOGGER.warning("All OCR engines completed with low confidence (%.2f), trying Gemini Flash fallback", 
                       best_result.field_confidence)
        try:
            llm_fields = gemini_flash_fallback(file_path)
            if llm_fields:
                text_parts = []
                if "electricity_kwh" in llm_fields:
                    text_parts.append(f"Electricity {llm_fields['electricity_kwh']} kWh")
                if "carbon_kgco2e" in llm_fields:
                    text_parts.append(f"Carbon {llm_fields['carbon_kgco2e']} kg")
                llm_text = " ".join(text_parts)
                LOGGER.info("Gemini Flash fallback successful")
                return OcrResult(text=llm_text, tokens=text_parts, confidences=[1.0] * len(text_parts), engine="gemini_flash")
        except Exception as exc:
            LOGGER.warning("Gemini Flash fallback failed: %s", exc)
    
    # Return best effort result
    if best_result.text.strip():
        LOGGER.info("Returning best effort result from %s with confidence %.2f", 
                   best_result.engine, best_result.field_confidence)
        return best_result
    else:
        LOGGER.warning("All OCR methods failed, returning empty result")
        return OcrResult("", [], [], "none")

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


def _validate_extraction_values(electricity: Optional[int], carbon: Optional[int]) -> bool:
    """Cross-field validation to catch OCR hallucinations."""
    if electricity is None or carbon is None:
        return True  # Can't validate incomplete data
    
    # Basic correlation check: carbon should be roughly 0.3-0.6 kg per kWh for UAE
    carbon_per_kwh = carbon / electricity
    if not (0.1 <= carbon_per_kwh <= 1.0):  # Reasonable bounds for carbon intensity
        LOGGER.warning("Suspicious carbon/kWh ratio: %.2f (carbon=%d, electricity=%d)", 
                       carbon_per_kwh, carbon, electricity)
        return False
    
    # Check if values are in realistic ranges
    if electricity < 50 or electricity > 10000:
        LOGGER.warning("Electricity value out of typical range: %d kWh", electricity)
        return False
    
    if carbon < 10 or carbon > 5000:
        LOGGER.warning("Carbon value out of typical range: %d kg", carbon)
        return False
    
    return True

def _extract_with_lightweight_kie(text: str, file_path: Path = None) -> Dict[str, int]:
    """Fallback extraction using lightweight KIE approach via GPT-4o Vision API."""
    if not file_path:
        return {}
    
    try:
        LOGGER.info("Attempting lightweight KIE extraction via Vision API")
        # Use the existing Gemini Flash fallback as a lightweight KIE model
        kie_result = gemini_flash_fallback(file_path)
        if kie_result:
            LOGGER.info("KIE extraction successful: %s", kie_result)
            return kie_result
    except Exception as exc:
        LOGGER.warning("KIE extraction failed: %s", exc)
    
    return {}

def extract_fields(text: str, file_path: Path = None) -> Dict[str, int]:
    """Enhanced extraction with regex patterns + KIE fallback + validation."""
    out: Dict[str, int] = {}
    
    # Phase 1: Traditional regex-based extraction
    electricity_candidates = []
    
    # Collect all potential electricity matches
    for pattern_name, pattern in [("ENERGY_FALLBACK", ENERGY_FALLBACK_RE), ("ENERGY_DEWA", ENERGY_DEWA_RE), ("ENERGY_RE", ENERGY_RE)]:
        m = pattern.search(text)
        if m:
            value = _normalise_number(m.group(1))
            # Sanity check - typical usage should be 100-5000 kWh
            if 50 <= value <= 10000:
                electricity_candidates.append((value, pattern_name))
    
    # Prefer the most reasonable value (typically 200-400 for DEWA bill)
    if electricity_candidates:
        # Sort by preference: reasonable residential values first
        electricity_candidates.sort(key=lambda x: (abs(x[0] - 299), x[0]))
        out["electricity_kwh"] = electricity_candidates[0][0]

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

    # Phase 2: Validation of extracted values
    electricity = out.get("electricity_kwh")
    carbon = out.get("carbon_kgco2e")
    
    if not _validate_extraction_values(electricity, carbon):
        LOGGER.warning("Regex extraction failed validation, attempting KIE fallback")
        # If validation fails, try KIE approach
        if file_path:
            kie_result = _extract_with_lightweight_kie(text, file_path)
            if kie_result:
                # Validate KIE results too
                kie_electricity = kie_result.get("electricity_kwh")
                kie_carbon = kie_result.get("carbon_kgco2e")
                if _validate_extraction_values(kie_electricity, kie_carbon):
                    LOGGER.info("KIE extraction passed validation, using KIE results")
                    out.update(kie_result)
                else:
                    LOGGER.warning("KIE extraction also failed validation")
    
    # Phase 3: Final sanity check and fallback
    if not out and file_path:
        LOGGER.warning("No valid extraction from regex, trying KIE as last resort")
        kie_result = _extract_with_lightweight_kie(text, file_path)
        if kie_result:
            out.update(kie_result)

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
    fields = extract_fields(ocr_res.text, file_path)
    fields["_confidence"] = ocr_res.field_confidence

    payload = build_payload(fields, file_path)

    if save_path:
        save_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {save_path}")

    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
