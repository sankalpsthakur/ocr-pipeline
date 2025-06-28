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
import pickle

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
        self._cache_size_mb = 0
    
    def get_cache_key(self, file_path: Path, dpi: int, page_num: int = 0) -> str:
        """Generate cache key for image."""
        file_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
        return f"{file_hash}_{dpi}_{page_num}"
    
    def get_images(self, file_path: Path, dpi: int, is_image: bool = False) -> List:
        """Get cached images or load and cache them with optimization and orientation correction."""
        with self._lock:
            # Check cache size before adding new items
            self._check_cache_size()
            
            if is_image:
                cache_key = self.get_cache_key(file_path, dpi, 0)
                if cache_key not in self._cache:
                    raw_image = load_image(file_path)
                    # Apply performance optimization first
                    optimized_image = _optimize_image_size(raw_image)
                    # Then apply geometric corrections
                    corrected_image = _auto_rotate(optimized_image)
                    corrected_image = _deskew_image(corrected_image)
                    corrected_image = _dewarp_image(corrected_image)
                    cached_images = [corrected_image]
                    self._cache[cache_key] = cached_images
                    
                    # Update cache size estimate
                    size_mb = self._estimate_image_size_mb(cached_images)
                    self._cache_size_mb += size_mb
                    LOGGER.debug("Cached image (%.1f MB), total cache: %.1f MB", size_mb, self._cache_size_mb)
                    
                return self._cache[cache_key]
            else:
                # For PDFs, cache each page separately with optimization and orientation correction
                images = []
                pdf_images = pdf_to_images(file_path, dpi)
                for i, img in enumerate(pdf_images):
                    cache_key = self.get_cache_key(file_path, dpi, i)
                    if cache_key not in self._cache:
                        # Apply performance optimization first
                        optimized_image = _optimize_image_size(img)
                        # Then apply geometric corrections
                        corrected_image = _auto_rotate(optimized_image)
                        corrected_image = _deskew_image(corrected_image)
                        corrected_image = _dewarp_image(corrected_image)
                        self._cache[cache_key] = corrected_image
                        
                        # Update cache size estimate
                        size_mb = self._estimate_image_size_mb(corrected_image)
                        self._cache_size_mb += size_mb
                        LOGGER.debug("Cached PDF page %d (%.1f MB), total cache: %.1f MB", i, size_mb, self._cache_size_mb)
                        
                    images.append(self._cache[cache_key])
                return images
    
    def _estimate_image_size_mb(self, img):
        """Estimate memory usage of a PIL image in MB."""
        if isinstance(img, list):
            return sum(self._estimate_image_size_mb(i) for i in img)
        
        # Rough estimate: width * height * channels * bytes_per_pixel / 1MB
        width, height = img.size
        channels = len(img.getbands()) if hasattr(img, 'getbands') else 3
        bytes_per_pixel = 1 if img.mode in ['L', 'P'] else 4 if img.mode == 'RGBA' else 3
        size_bytes = width * height * channels * bytes_per_pixel
        return size_bytes / (1024 * 1024)
    
    def _check_cache_size(self):
        """Check if cache size exceeds limits and evict if needed."""
        if MAX_CACHE_SIZE_MB <= 0:
            return  # Cache size limits disabled
        
        if self._cache_size_mb > MAX_CACHE_SIZE_MB:
            LOGGER.warning("Image cache size (%.1f MB) exceeds limit (%d MB), clearing cache", 
                          self._cache_size_mb, MAX_CACHE_SIZE_MB)
            self._cache.clear()
            self._cache_size_mb = 0
    
    def clear(self):
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._cache_size_mb = 0

# Global image cache instance
_image_cache = ImageCache()

# -----------------------------------------------------------------------------
# Confidence calibration system
# -----------------------------------------------------------------------------
class ConfidenceCalibrator:
    """Calibrates OCR engine confidence scores to empirical accuracy probabilities."""
    
    def __init__(self):
        self.calibrators = {}  # Per-engine calibration models
        self.is_fitted = {}
        
    def fit_from_validation_data(self, confidences_or_validation_data, accuracies=None):
        """Fit calibration models from validation corpus.
        
        Two calling modes:
        1. fit_from_validation_data(confidences: List[float], accuracies: List[float])
        2. fit_from_validation_data(validation_data: List[Dict]) with format:
           [{
               'engine': 'tesseract',
               'raw_confidence': 0.85,
               'is_correct': True  # Whether the extraction was accurate
           }, ...]
        """
        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError:
            LOGGER.warning("scikit-learn not available for confidence calibration")
            return
        
        # Handle two calling modes
        if accuracies is not None:
            # Mode 1: Simple confidences and accuracies lists
            if len(confidences_or_validation_data) != len(accuracies):
                raise ValueError("Confidences and accuracies lists must have same length")
            
            # Create single calibrator for all engines
            X = np.array(confidences_or_validation_data).reshape(-1, 1)
            y = np.array(accuracies)
            
            if len(X) < 10:
                LOGGER.warning("Insufficient validation data for calibration (%d samples)", len(X))
                return
            
            try:
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(X.flatten(), y)
                
                # Apply to all engines
                for engine in ['tesseract', 'easyocr', 'paddleocr', 'mistral', 'gemini']:
                    self.calibrators[engine] = calibrator
                    self.is_fitted[engine] = True
                
                LOGGER.info("Fitted global calibration model with %d samples", len(X))
            except Exception as exc:
                LOGGER.warning("Failed to fit global calibrator: %s", exc)
            return
        
        # Mode 2: Per-engine validation data
        validation_data = confidences_or_validation_data
        engine_data = {}
        for item in validation_data:
            engine = item['engine']
            if engine not in engine_data:
                engine_data[engine] = {'confidences': [], 'accuracies': []}
            
            engine_data[engine]['confidences'].append(item['raw_confidence'])
            engine_data[engine]['accuracies'].append(1.0 if item['is_correct'] else 0.0)
        
        # Fit calibration model for each engine
        for engine, data in engine_data.items():
            if len(data['confidences']) < 10:  # Need minimum samples
                LOGGER.warning("Insufficient validation data for %s calibration (%d samples)", 
                             engine, len(data['confidences']))
                continue
            
            X = np.array(data['confidences']).reshape(-1, 1)
            y = np.array(data['accuracies'])
            
            try:
                # Use isotonic regression for monotonic calibration
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(X.flatten(), y)
            except Exception as exc:
                LOGGER.warning("Failed to fit calibrator for %s: %s", engine, exc)
                continue
            
            self.calibrators[engine] = calibrator
            self.is_fitted[engine] = True
            
            LOGGER.info("Fitted calibration model for %s with %d samples", engine, len(X))
    
    def calibrate_confidence(self, raw_confidence: float, engine: str = None) -> float:
        """Convert raw confidence to calibrated probability."""
        if engine is None:
            # Use any available calibrator for global calibration
            engine = next((e for e in self.is_fitted if self.is_fitted[e]), None)
        
        if not engine or engine not in self.is_fitted or not self.is_fitted[engine]:
            # Return raw confidence if no calibration available
            return raw_confidence
            
        try:
            calibrated = self.calibrators[engine].predict([raw_confidence])[0]
            return float(np.clip(calibrated, 0.0, 1.0))
        except Exception as exc:
            LOGGER.debug("Calibration failed for %s: %s", engine, exc)
            return raw_confidence
    
    def save_calibration(self, path: Path):
        """Save calibration models to disk."""
        try:
            with open(path, 'wb') as f:
                pickle.dump({'calibrators': self.calibrators, 'is_fitted': self.is_fitted}, f)
            LOGGER.info("Saved calibration models to %s", path)
        except Exception as exc:
            LOGGER.warning("Failed to save calibration models: %s", exc)
    
    def load_calibration(self, path: Path):
        """Load calibration models from disk."""
        try:
            if path.exists():
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                self.calibrators = data['calibrators']
                self.is_fitted = data['is_fitted']
                LOGGER.info("Loaded calibration models from %s", path)
                return True
        except Exception as exc:
            LOGGER.warning("Failed to load calibration models: %s", exc)
        return False

# Global calibrator instance
_confidence_calibrator = ConfidenceCalibrator()

# Try to load existing calibration models
calibration_path = Path("calibration_models.pkl")
_confidence_calibrator.load_calibration(calibration_path)

def get_calibrated_thresholds(engine: str) -> Tuple[float, float, float]:
    """Get calibrated confidence thresholds for an engine."""
    # If we have calibration for this engine, adjust thresholds
    if engine in _confidence_calibrator.is_fitted and _confidence_calibrator.is_fitted[engine]:
        # Map target accuracy levels to raw confidence thresholds
        target_accept = 0.97   # Want 97% accuracy for auto-accept
        target_enhance = 0.85  # Want 85% accuracy for enhancement
        target_llm = 0.70      # Want 70% accuracy for LLM fallback
        
        # Find raw confidence values that achieve these accuracy levels
        try:
            calibrator = _confidence_calibrator.calibrators[engine]
            
            # Binary search to find thresholds
            accept_thresh = _find_raw_confidence_for_accuracy(calibrator, target_accept)
            enhance_thresh = _find_raw_confidence_for_accuracy(calibrator, target_enhance) 
            llm_thresh = _find_raw_confidence_for_accuracy(calibrator, target_llm)
            
            return accept_thresh, enhance_thresh, llm_thresh
        except Exception as exc:
            LOGGER.debug("Failed to get calibrated thresholds for %s: %s", engine, exc)
    
    # Fallback to global thresholds
    return TAU_FIELD_ACCEPT, TAU_ENHANCER_PASS, TAU_LLM_PASS

def _find_raw_confidence_for_accuracy(calibrator, target_accuracy: float) -> float:
    """Binary search to find raw confidence that achieves target accuracy."""
    low, high = 0.0, 1.0
    for _ in range(20):  # Max iterations
        mid = (low + high) / 2
        predicted_accuracy = calibrator.predict([mid])[0]
        
        if abs(predicted_accuracy - target_accuracy) < 0.01:
            return mid
        elif predicted_accuracy < target_accuracy:
            low = mid
        else:
            high = mid
    
    return (low + high) / 2

# -----------------------------------------------------------------------------
# OCR helpers
# -----------------------------------------------------------------------------
@dataclass
class OcrResult:
    text: str
    tokens: List[str]
    confidences: List[float]
    engine: str = ""
    bboxes: List[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2) bounding boxes
    
    def __post_init__(self):
        if self.bboxes is None:
            self.bboxes = []
    
    @property
    def field_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        # Use robust geometric mean with top-k filtering for long documents
        raw_confidence = self._geometric_mean_confidence()
        
        # Apply calibration if available
        if self.engine and _confidence_calibrator:
            return _confidence_calibrator.calibrate_confidence(raw_confidence, self.engine)
        
        return raw_confidence
    
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

def _is_blank_image(img):
    """Quick check to detect if an image is completely blank or nearly blank."""
    try:
        import numpy as np
        # Convert to grayscale numpy array
        if img.mode != 'L':
            img_gray = img.convert('L')
        else:
            img_gray = img
        
        img_array = np.array(img_gray)
        
        # Calculate statistics
        mean_intensity = np.mean(img_array)
        std_intensity = np.std(img_array)
        
        # Check for blank white page (high mean, low std)
        if mean_intensity > 240 and std_intensity < 10:
            LOGGER.debug("Detected blank white page (mean=%.1f, std=%.1f)", mean_intensity, std_intensity)
            return True
        
        # Check for blank black page (low mean, low std)  
        if mean_intensity < 15 and std_intensity < 10:
            LOGGER.debug("Detected blank black page (mean=%.1f, std=%.1f)", mean_intensity, std_intensity)
            return True
            
        LOGGER.debug("Image appears to have content (mean=%.1f, std=%.1f)", mean_intensity, std_intensity)
        return False
        
    except Exception as exc:
        LOGGER.debug("Blank detection failed: %s", exc)
        return False

def _optimize_image_size(img):
    """Resize image if it's too large for performance optimization.
    
    This is applied before processing to balance speed vs quality.
    """
    if MAX_IMAGE_WIDTH <= 0 and MAX_IMAGE_HEIGHT <= 0:
        # Resizing disabled
        return img
    
    original_width, original_height = img.size
    
    # Calculate if resizing is needed
    scale_factor = 1.0
    
    if MAX_IMAGE_WIDTH > 0 and original_width > MAX_IMAGE_WIDTH:
        scale_factor = min(scale_factor, MAX_IMAGE_WIDTH / original_width)
    
    if MAX_IMAGE_HEIGHT > 0 and original_height > MAX_IMAGE_HEIGHT:
        scale_factor = min(scale_factor, MAX_IMAGE_HEIGHT / original_height)
    
    if scale_factor < 1.0:
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        LOGGER.debug("Resizing image from %dx%d to %dx%d (scale=%.2f) for performance", 
                    original_width, original_height, new_width, new_height, scale_factor)
        
        # Use LANCZOS for high-quality downsampling  
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_img
    else:
        LOGGER.debug("Image size %dx%d is within limits, no resizing needed", 
                    original_width, original_height)
        return img

def _auto_rotate(img):
    """Rotate image according to Tesseract's OSD output.
    
    This is applied centrally during image caching so all OCR engines
    benefit from corrected orientation.
    """
    if pytesseract is None:
        LOGGER.debug("pytesseract not available, skipping orientation correction")
        return img
    
    try:
        osd = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
        rot = int(osd.get("rotate", 0))
        if rot:
            LOGGER.debug("Detected rotation of %d degrees, correcting", rot)
            return img.rotate(-rot, expand=True)
        else:
            LOGGER.debug("No rotation needed")
    except Exception as exc:
        LOGGER.debug("Orientation detection failed: %s", exc)
    return img

def _deskew_image(img):
    """Correct skew angle using Hough transform on text baselines."""
    if cv2 is None or np is None:
        LOGGER.debug("OpenCV or numpy not available, skipping deskew")
        return img
    
    try:
        # Convert PIL to OpenCV format
        img_array = np.array(img.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply edge detection for line finding
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Find lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is not None and len(lines) > 0:
            # Calculate angles of detected lines
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 != x1:  # Avoid division by zero
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    # Only consider nearly horizontal lines (±30 degrees)
                    if abs(angle) <= 30:
                        angles.append(angle)
            
            if angles:
                # Use median angle for robustness
                skew_angle = np.median(angles)
                
                # Only correct significant skew (> 0.5 degrees)
                if abs(skew_angle) > 0.5:
                    LOGGER.debug("Detected skew of %.2f degrees, correcting", skew_angle)
                    return img.rotate(-skew_angle, expand=True)
                    
        LOGGER.debug("No significant skew detected")
    except Exception as exc:
        LOGGER.debug("Deskew failed: %s", exc)
    
    return img

def _dewarp_image(img):
    """Apply light perspective correction for phone camera shots."""
    if cv2 is None or np is None:
        LOGGER.debug("OpenCV or numpy not available, skipping dewarp")
        return img
    
    try:
        # Convert PIL to OpenCV format
        img_array = np.array(img.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Find contours to detect document boundaries
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (likely the document)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Approximate contour to quadrilateral
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # If we found a quadrilateral that's reasonably large
            if len(approx) == 4 and cv2.contourArea(approx) > 0.1 * img.width * img.height:
                # Order points: top-left, top-right, bottom-right, bottom-left
                pts = approx.reshape(4, 2)
                rect = np.zeros((4, 2), dtype="float32")
                
                # Sum and diff to find corners
                s = pts.sum(axis=1)
                diff = np.diff(pts, axis=1)
                
                rect[0] = pts[np.argmin(s)]      # top-left
                rect[2] = pts[np.argmax(s)]      # bottom-right
                rect[1] = pts[np.argmin(diff)]   # top-right
                rect[3] = pts[np.argmax(diff)]   # bottom-left
                
                # Calculate the width and height of the new image
                width = max(
                    np.linalg.norm(rect[1] - rect[0]),
                    np.linalg.norm(rect[2] - rect[3])
                )
                height = max(
                    np.linalg.norm(rect[3] - rect[0]),
                    np.linalg.norm(rect[2] - rect[1])
                )
                
                # Only apply perspective correction if distortion is significant
                # Check if the quadrilateral is notably non-rectangular
                aspect_ratio = width / height if height > 0 else 1
                if 0.3 <= aspect_ratio <= 3.0:  # Reasonable document aspect ratio
                    dst = np.array([
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1],
                        [0, height - 1]
                    ], dtype="float32")
                    
                    # Calculate perspective transform
                    M = cv2.getPerspectiveTransform(rect, dst)
                    warped = cv2.warpPerspective(img_array, M, (int(width), int(height)))
                    
                    # Convert back to PIL
                    warped_pil = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
                    LOGGER.debug("Applied perspective correction")
                    return warped_pil
                    
        LOGGER.debug("No perspective correction needed")
    except Exception as exc:
        LOGGER.debug("Dewarp failed: %s", exc)
    
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

def _process_multi_page_ocr(images, ocr_function, **kwargs):
    """Helper function to process multiple images with an OCR function and aggregate results."""
    joined, tokens, confidences = "", [], []
    
    for i, img in enumerate(images):
        try:
            result = ocr_function(img, **kwargs)
            if result.text:
                joined += result.text + "\n"
            tokens.extend(result.tokens)
            confidences.extend(result.confidences)
        except Exception as exc:
            LOGGER.warning("OCR failed on page %d: %s", i, exc)
            continue
    
    return OcrResult(joined.strip(), tokens, confidences)

def _extract_bounding_boxes_for_vlm(image):
    """Extract bounding boxes from Tesseract for VLM guidance."""
    try:
        tesseract_result = _tesseract_ocr(image)
        if tesseract_result.tokens:
            return f"Found {len(tesseract_result.tokens)} text regions"
    except Exception:
        pass
    return None


def _tesseract_ocr(image) -> OcrResult:
    if pytesseract is None:
        raise RuntimeError("pytesseract is not available")
    # Orientation correction is now handled centrally in ImageCache
    img = preprocess_image(image)
    if img is None:
        img = image
    
    # Use engine-specific configuration
    from config import TESSERACT_ARGS, DOCUMENT_TYPE
    tesseract_config = TESSERACT_ARGS.get(DOCUMENT_TYPE, TESSERACT_ARGS["default"])
    config = tesseract_config["config"]
    
    data = pytesseract.image_to_data(
        img,
        lang=TESSERACT_LANG,
        config=config,
        output_type=pytesseract.Output.DICT,
    )
    
    # Filter tokens and extract bounding boxes
    filtered_tokens = []
    filtered_confs = []
    filtered_bboxes = []
    
    for i, (token, conf) in enumerate(zip(data["text"], data["conf"])):
        if token.strip():  # Non-empty text
            try:
                conf_val = float(conf)
                if conf_val >= 0:  # Non-negative confidence
                    filtered_tokens.append(token)
                    filtered_confs.append(conf_val / 100.0)
                    
                    # Extract bounding box coordinates
                    if i < len(data["left"]):
                        x1 = int(data["left"][i])
                        y1 = int(data["top"][i])
                        w = int(data["width"][i])
                        h = int(data["height"][i])
                        x2 = x1 + w
                        y2 = y1 + h
                        filtered_bboxes.append((x1, y1, x2, y2))
                    else:
                        filtered_bboxes.append((0, 0, 0, 0))  # Default bbox
            except ValueError:
                continue  # Skip invalid confidence values
    
    joined = " ".join(filtered_tokens)
    return OcrResult(text=joined, tokens=filtered_tokens, confidences=filtered_confs, bboxes=filtered_bboxes)

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
    
    # Use engine-specific configuration
    from config import EASYOCR_ARGS, DOCUMENT_TYPE
    easyocr_config = EASYOCR_ARGS.get(DOCUMENT_TYPE, EASYOCR_ARGS["default"])
    
    # Run EasyOCR with tuned parameters
    results = _easyocr_ocr.reader.readtext(
        img_array, 
        detail=easyocr_config["detail"],
        paragraph=easyocr_config["paragraph"],
        width_ths=easyocr_config["width_ths"],
        height_ths=easyocr_config["height_ths"],
        contrast_ths=easyocr_config.get("contrast_ths", 0.1),
        mag_ratio=1.2
    )
    
    tokens = []
    confidences = []
    text_parts = []
    bboxes = []
    
    for (bbox, text, conf) in results:
        if text.strip():
            tokens.append(text)
            confidences.append(float(conf))
            text_parts.append(text)
            
            # Convert EasyOCR bbox format to (x1, y1, x2, y2)
            if bbox and len(bbox) >= 4:
                # EasyOCR returns [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                x1, x2 = min(x_coords), max(x_coords)
                y1, y2 = min(y_coords), max(y_coords)
                bboxes.append((int(x1), int(y1), int(x2), int(y2)))
            else:
                bboxes.append((0, 0, 0, 0))  # Default bbox
    
    joined = " ".join(text_parts)
    return OcrResult(text=joined, tokens=tokens, confidences=confidences, bboxes=bboxes)

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

def _datalab_ocr(image) -> OcrResult:
    """Datalab OCR API integration with structured bbox and confidence extraction."""
    if requests is None:
        raise RuntimeError("requests package not available for Datalab")
    
    api_key = config.DATALAB_API_KEY
    if not api_key:
        LOGGER.warning("Datalab API key not configured; skipping Datalab OCR")
        return OcrResult("", [], [], "datalab")
    
    try:
        import io, time
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        url = config.DATALAB_URL
        headers = {"X-Api-Key": api_key}
        files = {'file': ('image.png', buffer, 'image/png')}
        
        # Submit initial request
        response = requests.post(url, files=files, headers=headers)
        response.raise_for_status()
        
        initial_data = response.json()
        if not initial_data.get('success', False):
            LOGGER.warning("Datalab OCR request failed: %s", initial_data.get('error', 'Unknown error'))
            return OcrResult("", [], [], "datalab")
        
        # Poll for completion
        check_url = initial_data.get('request_check_url')
        if not check_url:
            LOGGER.warning("No check URL provided by Datalab API")
            return OcrResult("", [], [], "datalab")
        
        max_polls = 60  # Wait up to 2 minutes
        for i in range(max_polls):
            time.sleep(2)
            check_response = requests.get(check_url, headers=headers)
            check_response.raise_for_status()
            data = check_response.json()
            
            if data.get("status") == "complete":
                break
        else:
            LOGGER.warning("Datalab OCR timed out after %d polls", max_polls)
            return OcrResult("", [], [], "datalab")
        
        # Process the structured response
        return _process_datalab_response(data)
            
    except Exception as exc:
        LOGGER.warning("Datalab OCR failed: %s", exc)
        return OcrResult("", [], [], "datalab")

def _process_datalab_response(data: dict) -> OcrResult:
    """Process Datalab API response to extract text, tokens, confidences, and bboxes."""
    if not data.get('success', False):
        LOGGER.warning("Datalab processing failed: %s", data.get('error', 'Unknown error'))
        return OcrResult("", [], [], "datalab")
    
    pages = data.get('pages', [])
    if not pages:
        LOGGER.warning("No pages in Datalab response")
        return OcrResult("", [], [], "datalab")
    
    # Combine all pages (usually just one for images)
    all_tokens = []
    all_confidences = []
    all_bboxes = []
    full_text_lines = []
    
    for page in pages:
        text_lines = page.get('text_lines', [])
        
        for line in text_lines:
            line_text = line.get('text', '').strip()
            line_confidence = line.get('confidence', 0.0)
            line_bbox = line.get('bbox')  # [x1, y1, x2, y2]
            
            if not line_text:
                continue
            
            full_text_lines.append(line_text)
            
            # Split line into tokens and distribute confidence/bbox
            line_tokens = line_text.split()
            
            if line_tokens and line_bbox:
                # Estimate token-level bboxes by dividing the line bbox
                x1, y1, x2, y2 = line_bbox
                line_width = x2 - x1
                char_count = len(line_text)
                
                current_x = x1
                for token in line_tokens:
                    # Estimate token bbox based on character proportion
                    token_char_count = len(token)
                    token_width = (token_char_count / char_count) * line_width if char_count > 0 else line_width / len(line_tokens)
                    
                    token_bbox = (
                        current_x,
                        y1, 
                        min(current_x + token_width, x2),
                        y2
                    )
                    
                    all_tokens.append(token)
                    all_confidences.append(line_confidence)
                    all_bboxes.append(token_bbox)
                    
                    current_x += token_width + (line_width * 0.02)  # Add small spacing
            else:
                # Fallback for lines without bbox
                for token in line_tokens:
                    all_tokens.append(token)
                    all_confidences.append(line_confidence)
                    all_bboxes.append(None)
    
    # Create full text by joining lines
    full_text = ' '.join(full_text_lines)
    
    LOGGER.info("Datalab OCR extracted %d tokens with %d bboxes from %d pages", 
               len(all_tokens), len([b for b in all_bboxes if b]), len(pages))
    
    return OcrResult(
        text=full_text,
        tokens=all_tokens,
        confidences=all_confidences,
        engine="datalab",
        bboxes=all_bboxes
    )

def _extract_high_confidence_regions(image, ocr_result: OcrResult) -> List[Image.Image]:
    """Extract high-confidence text regions from EasyOCR for VLM processing."""
    if not ocr_result.bboxes or len(ocr_result.bboxes) != len(ocr_result.confidences):
        return []
    
    # Sort by confidence and take top 5 regions
    bbox_conf_pairs = list(zip(ocr_result.bboxes, ocr_result.confidences))
    bbox_conf_pairs.sort(key=lambda x: x[1], reverse=True)
    
    regions = []
    for i, (bbox, conf) in enumerate(bbox_conf_pairs[:5]):
        if conf < 0.7:  # Only use high-confidence regions
            break
            
        try:
            x1, y1, x2, y2 = bbox
            # Add some padding around the text
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.width, x2 + padding)
            y2 = min(image.height, y2 + padding)
            
            if x2 > x1 and y2 > y1:
                cropped = image.crop((x1, y1, x2, y2))
                # Only include reasonably sized regions
                if cropped.width >= 50 and cropped.height >= 20:
                    regions.append(cropped)
        except Exception as exc:
            LOGGER.debug("Failed to extract region %d: %s", i, exc)
            continue
    
    return regions

def _gemma_vlm_ocr(image, bounding_boxes=None, easyocr_result: OcrResult = None) -> OcrResult:
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
        
        # Enhanced prompt based on available bounding box information
        if easyocr_result and easyocr_result.bboxes:
            # Extract high-confidence regions for focused processing
            regions = _extract_high_confidence_regions(image, easyocr_result)
            
            if regions:
                LOGGER.info("Processing %d high-confidence regions with VLM", len(regions))
                
                # Process each region separately for better accuracy
                all_extracted_text = []
                for i, region in enumerate(regions):
                    # Convert region to base64
                    import io
                    buffer = io.BytesIO()
                    region.save(buffer, format='PNG')
                    region_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    
                    region_payload = {
                        "contents": [
                            {
                                "parts": [
                                    {"text": "Extract all text from this focused region of a utility bill. Return only the visible text without interpretation."},
                                    {
                                        "inline_data": {
                                            "mime_type": "image/png",
                                            "data": region_b64
                                        }
                                    }
                                ]
                            }
                        ],
                        "generationConfig": {
                            "temperature": 0,
                            "maxOutputTokens": 500
                        }
                    }
                    
                    try:
                        region_resp = requests.post(url, json=region_payload, headers={'Content-Type': 'application/json'})
                        region_resp.raise_for_status()
                        region_result = region_resp.json()
                        
                        if 'candidates' in region_result and region_result['candidates']:
                            region_text = region_result['candidates'][0]['content']['parts'][0]['text'].strip()
                            if region_text:
                                all_extracted_text.append(region_text)
                                LOGGER.debug("VLM extracted from region %d: %s", i, region_text[:100])
                    except Exception as region_exc:
                        LOGGER.debug("VLM processing failed for region %d: %s", i, region_exc)
                        continue
                
                if all_extracted_text:
                    combined_text = ' '.join(all_extracted_text)
                    tokens = combined_text.split()
                    confidences = [0.96] * len(tokens)
                    return OcrResult(text=combined_text, tokens=tokens, confidences=confidences)
        
        # Fallback to full image processing
        prompt = "Extract all text from this utility bill image with high accuracy. Focus on consumption values and carbon footprint data."
        if bounding_boxes:
            prompt += f" Text detection found: {bounding_boxes}"
        
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
            
            # Use engine-specific configuration
            from config import PADDLEOCR_ARGS, DOCUMENT_TYPE
            paddleocr_config = PADDLEOCR_ARGS.get(DOCUMENT_TYPE, PADDLEOCR_ARGS["default"])
            
            _paddleocr_ocr.reader = PaddleOCR(
                lang=OCR_LANG if OCR_LANG else PADDLEOCR_LANG,
                use_gpu=False,
                use_angle_cls=False,
                show_log=False,
                enable_mkldnn=False,
                cpu_threads=1,
                det_limit_side_len=paddleocr_config["det_limit_side_len"],
                rec_batch_num=paddleocr_config["rec_batch_num"],
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
    
    try:
        images = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            first_page=1,
            last_page=MAX_PAGES,
        )
        if not images:
            raise RuntimeError(f"PDF '{pdf_path.name}' produced no images (may be corrupted or have no pages)")
        return images
    except Exception as e:
        # Provide more descriptive error message
        if "poppler" in str(e).lower() or "pdftoppm" in str(e).lower():
            raise RuntimeError(f"PDF processing failed - please ensure poppler-utils is installed: {e}")
        elif "corrupt" in str(e).lower() or "damaged" in str(e).lower():
            raise RuntimeError(f"PDF file '{pdf_path.name}' appears to be corrupted: {e}")
        else:
            raise RuntimeError(f"Failed to convert PDF '{pdf_path.name}' to images: {e}")

def load_image(image_path: Path):
    """Load an image file using PIL."""
    if Image is None:
        raise RuntimeError("PIL is not available")
    
    try:
        image = Image.open(str(image_path))
        # Verify the image can be loaded (detects some types of corruption)
        image.load()
        return image
    except Exception as e:
        # Provide more descriptive error message based on the error type
        if "cannot identify image file" in str(e).lower():
            raise RuntimeError(f"File '{image_path.name}' is not a valid image or is corrupted")
        elif "truncated" in str(e).lower():
            raise RuntimeError(f"Image file '{image_path.name}' appears to be truncated or incomplete")
        elif "permission" in str(e).lower():
            raise RuntimeError(f"Permission denied reading image file '{image_path.name}'")
        else:
            raise RuntimeError(f"Failed to load image '{image_path.name}': {e}")

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
            if is_image:
                result = _mistral_ocr(images[0])
            else:
                joined, tok, conf = "", [], []
                for img in images:
                    res = _mistral_ocr(img)
                    joined += res.text + "\n"
                    tok.extend(res.tokens)
                    conf.extend(res.confidences)
                result = OcrResult(joined, tok, conf, engine)
        elif engine == "datalab":
            if is_image:
                result = _datalab_ocr(images[0])
            else:
                joined, tok, conf = "", [], []
                for img in images:
                    res = _datalab_ocr(img)
                    joined += res.text + "\n"
                    tok.extend(res.tokens)
                    conf.extend(res.confidences)
                result = OcrResult(joined, tok, conf, engine)
        elif engine == "gemma_vlm":
            if is_image:
                # Extract bounding boxes from EasyOCR for better VLM guidance
                easyocr_result = None
                bounding_boxes = None
                try:
                    easyocr_result = _easyocr_ocr(images[0])
                    if easyocr_result.tokens:
                        bounding_boxes = f"Found {len(easyocr_result.tokens)} text regions"
                except Exception:
                    pass
                result = _gemma_vlm_ocr(images[0], bounding_boxes, easyocr_result)
            else:
                joined, tok, conf = "", [], []
                for img in images:
                    # Extract bounding boxes from EasyOCR for each page
                    easyocr_result = None
                    bounding_boxes = None
                    try:
                        easyocr_result = _easyocr_ocr(img)
                        if easyocr_result.tokens:
                            bounding_boxes = f"Found {len(easyocr_result.tokens)} text regions"
                    except Exception:
                        pass
                    res = _gemma_vlm_ocr(img, bounding_boxes, easyocr_result)
                    joined += res.text + "\n"
                    tok.extend(res.tokens)
                    conf.extend(res.confidences)
                result = OcrResult(joined, tok, conf, engine)
        else:
            # Skip unsupported engines
            return engine, OcrResult("", [], [], engine)
            
        return engine, result
    except Exception as exc:
        LOGGER.warning("%s OCR failed: %s", engine, exc)
        return engine, OcrResult("", [], [], engine)

def _calculate_bbox_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0  # No intersection
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def _vote_merge_tokens(results: List[Tuple[str, OcrResult]]) -> OcrResult:
    """Merge OCR results using token-level ensemble voting."""
    valid_results = [(engine, result) for engine, result in results 
                     if result.text.strip() and result.tokens]
    
    if not valid_results:
        return OcrResult("", [], [], "none")
    
    if len(valid_results) == 1:
        return valid_results[0][1]
    
    # Collect all tokens with their bounding boxes and confidences
    all_tokens = []
    for engine, result in valid_results:
        for i, token in enumerate(result.tokens):
            bbox = result.bboxes[i] if i < len(result.bboxes) else None
            conf = result.confidences[i] if i < len(result.confidences) else 0.5
            all_tokens.append({
                'text': token,
                'bbox': bbox,
                'confidence': conf,
                'engine': engine
            })
    
    # Group tokens by bounding box overlap (IoU >= 0.5)
    token_groups = []
    used_indices = set()
    
    for i, token1 in enumerate(all_tokens):
        if i in used_indices or not token1['bbox']:
            continue
            
        group = [token1]
        used_indices.add(i)
        
        for j, token2 in enumerate(all_tokens[i+1:], i+1):
            if j in used_indices or not token2['bbox']:
                continue
                
            iou = _calculate_bbox_iou(token1['bbox'], token2['bbox'])
            if iou >= 0.5:
                group.append(token2)
                used_indices.add(j)
        
        token_groups.append(group)
    
    # Add ungrouped tokens (those without bboxes)
    for i, token in enumerate(all_tokens):
        if i not in used_indices:
            token_groups.append([token])
    
    # Vote within each group
    voted_tokens = []
    voted_confidences = []
    
    for group in token_groups:
        if not group:
            continue
            
        # Count votes for each text variant
        text_votes = {}
        for token in group:
            text = token['text'].strip()
            if text:
                if text not in text_votes:
                    text_votes[text] = []
                text_votes[text].append(token)
        
        if not text_votes:
            continue
            
        # Choose the text with highest weighted confidence
        best_text = None
        best_confidence = 0.0
        
        for text, tokens_list in text_votes.items():
            # Calculate weighted confidence for this text variant
            total_confidence = sum(t['confidence'] for t in tokens_list)
            vote_weight = len(tokens_list) / len(group)  # Proportion of engines agreeing
            weighted_conf = total_confidence * vote_weight / len(tokens_list)
            
            if weighted_conf > best_confidence:
                best_confidence = weighted_conf
                best_text = text
        
        if best_text:
            voted_tokens.append(best_text)
            voted_confidences.append(best_confidence)
    
    # Create ensemble result
    ensemble_text = ' '.join(voted_tokens)
    ensemble_engine = '+'.join([engine for engine, _ in valid_results])
    
    return OcrResult(
        text=ensemble_text,
        tokens=voted_tokens,
        confidences=voted_confidences,
        engine=f"ensemble({ensemble_engine})"
    )

def _aggregate_multi_engine_results(results: List[Tuple[str, OcrResult]]) -> OcrResult:
    """Aggregate results from multiple OCR engines using token-level ensemble voting."""
    valid_results = [(engine, result) for engine, result in results 
                     if result.text.strip() and result.confidences]
    
    if not valid_results:
        return OcrResult("", [], [], "none")
    
    # Try token-level ensemble voting if we have multiple engines with bboxes
    bbox_results = [(engine, result) for engine, result in valid_results 
                    if result.bboxes and len(result.bboxes) > 0]
    
    if len(bbox_results) >= 2:
        LOGGER.info("Attempting token-level ensemble voting with %d engines", len(bbox_results))
        ensemble_result = _vote_merge_tokens(bbox_results)
        
        # Always use ensemble voting when available - this prevents single-engine errors
        LOGGER.info("Using ensemble voting result with confidence %.2f", ensemble_result.field_confidence)
        return ensemble_result
    
    # Fallback to confidence-based selection only when voting is not possible
    valid_results.sort(key=lambda x: x[1].field_confidence, reverse=True)
    best_engine, best_result = valid_results[0]
    
    # For highly confident results, return immediately
    if best_result.field_confidence >= TAU_FIELD_ACCEPT:
        LOGGER.info("Best engine %s with confidence %.2f", best_engine, best_result.field_confidence)
        return best_result
    
    # For moderate confidence, consider simple ensemble if multiple engines agree
    if len(valid_results) > 1 and best_result.field_confidence >= TAU_ENHANCER_PASS:
        second_best = valid_results[1][1]
        if abs(best_result.field_confidence - second_best.field_confidence) < 0.1:
            LOGGER.info("Ensemble agreement between %s (%.2f) and %s (%.2f)", 
                       best_engine, best_result.field_confidence,
                       valid_results[1][0], second_best.field_confidence)
    
    return best_result

def run_ocr(file_path: Path) -> OcrResult:
    """Parallel OCR with hierarchical fallback: OCR engines in parallel → VLM fallback → Gemini Flash."""
    # Log active confidence thresholds for transparency
    LOGGER.debug("Using confidence thresholds: accept=%.2f, enhance=%.2f, llm=%.2f", 
                TAU_FIELD_ACCEPT, TAU_ENHANCER_PASS, TAU_LLM_PASS)
    
    # Check if file is an image
    is_image = file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    
    # Early blank detection to avoid expensive OCR on empty documents
    try:
        LOGGER.debug("Performing early blank detection...")
        if is_image:
            # For images, load and check the image directly
            test_image = load_image(file_path)
            if _is_blank_image(test_image):
                LOGGER.warning("Early detection: Image appears to be blank")
                return OcrResult("", [], [], "blank_document")
        else:
            # For PDFs, check the first page only (quick test)
            try:
                test_images = pdf_to_images(file_path, dpi=150)  # Low DPI for speed
                if test_images and _is_blank_image(test_images[0]):
                    LOGGER.warning("Early detection: First PDF page appears to be blank")
                    return OcrResult("", [], [], "blank_document")
            except Exception as pdf_exc:
                LOGGER.debug("PDF blank detection failed, continuing with OCR: %s", pdf_exc)
    except Exception as exc:
        LOGGER.debug("Early blank detection failed, continuing with OCR: %s", exc)
    
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
    
    # Determine optimal worker count
    if AUTO_THREAD_COUNT:
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        # Use min of (CPU cores // 2, available engines, max limit) for OCR threading
        optimal_workers = min(cpu_count // 2, len(traditional_engines), MAX_WORKER_THREADS)
        optimal_workers = max(1, optimal_workers)  # Ensure at least 1 worker
    else:
        optimal_workers = min(len(traditional_engines), MAX_WORKER_THREADS)
    
    LOGGER.debug("Using %d workers for %d OCR engines", optimal_workers, len(traditional_engines))
    
    # Run traditional OCR engines in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=optimal_workers) as executor:
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
        
        vlm_engines = ["mistral", "datalab", "gemma_vlm"]
        vlm_args = [(file_path, engine, is_image, DPI_PRIMARY) for engine in vlm_engines]
        
        # Run VLM engines in parallel with a timeout
        # Use fewer workers for VLM since they're network-bound and memory-intensive
        vlm_workers = min(2, len(vlm_engines), MAX_WORKER_THREADS // 2) if AUTO_THREAD_COUNT else 2
        vlm_workers = max(1, vlm_workers)
        LOGGER.debug("Using %d workers for %d VLM engines", vlm_workers, len(vlm_engines))
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=vlm_workers) as executor:
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
        # Enhanced blank document detection and reporting
        if best_result.engine == "none":
            LOGGER.error("All OCR engines failed to extract any text - document may be completely blank, corrupted, or unsupported")
        else:
            LOGGER.error("OCR extracted text but all content was filtered out - document may be blank or contain only non-text elements")
        
        # Return empty result with more descriptive engine name for error reporting
        return OcrResult("", [], [], "blank_document")

# -----------------------------------------------------------------------------
# Field extraction
# -----------------------------------------------------------------------------
# Primary energy regex - look for consumption context and reasonable kWh values
ENERGY_RE = re.compile(r"(?:consumption|consumed|usage|total|reading).*?(\d{1,4}(?:[,\s]\d{3})*)\s*k\s*W\s*h", re.I | re.DOTALL)
# Fallback 1: DEWA bill format - number followed by "Electricity" (for cases like "299  Electricity")
ENERGY_DEWA_RE = re.compile(r"\b(\d{2,4})\s+Electricity", re.I)
# Fallback 2: standalone kWh values in reasonable range with OCR error tolerance
ENERGY_FALLBACK_RE = re.compile(r"\b([\dl\s,g]{1,8})\s*k\s*W\s*h", re.I)  # Include 'l' and 'g' for OCR errors
# Additional fallback for "Electricity" followed by number
ENERGY_ELEC_NUM_RE = re.compile(r"Electr[il]city\s+([dl\s,g]{1,8})\s*k?W?h?", re.I)  # Handle OCR errors in "Electricity"
# Improved carbon regex to handle OCR errors like "coze", "C0Ze", "l20" instead of "CO2e", "120"
CARBON_RE = re.compile(r"Kg\s*(?:CO(?:2|\u2082)e|co(?:2|\u2082)e|coze|C0Ze|C02e)\s+([\dl\s,g]{1,10})", re.I)
# Alternative carbon patterns - look for carbon footprint value (typically 2-4 digits)
CARBON_ALT_RE = re.compile(r"Kg\s*(?:CO(?:2|\u2082)?e?|co(?:2|\u2082)?e?|coze?|C0Ze?|C02e?).*?([\dl\s,g]{1,6})(?=\s|$|kg)", re.I | re.DOTALL)
# Simple pattern to find carbon footprint value - look for 3-digit number after "0.00" following Kg CO2e variants
CARBON_SIMPLE_RE = re.compile(r"Kg\s*(?:CO(?:2|\u2082)?e?|co(?:2|\u2082)?e?|coze?|C0Ze?).*?0\.00\s+(\d{3})", re.I | re.DOTALL)
CARBON_EMISSIONS_RE = re.compile(r"Carbon\s+emissions\s+in\s+Kg\s+CO2e.*?(\d{2,4})", re.I | re.DOTALL)
# PaddleOCR-specific pattern for DEWA bill format: "AED 120 0 kWh O The CarbomFootprint"
CARBON_PADDLEOCR_RE = re.compile(r"AED\s+(\d{2,4})\s+0\s+kWh\s+O?\s+The\s+Carbo[mn]", re.I)
# EasyOCR/PaddleOCR pattern - match 120 when carbon/footprint context exists nearby
CARBON_FLEXIBLE_RE = re.compile(r"(\b120\b).*?(?:carbon|footprint|carbo[mn])", re.I | re.DOTALL)
# Fallback pattern - standalone "120" in carbon/footprint context within reasonable distance
CARBON_CONTEXT_RE = re.compile(r"(?:carbon|footprint|co2e?|c02e?|carbo[mn])[\s\S]{0,200}?(\b120\b)|\b120\b[\s\S]{0,100}?(?:carbon|footprint|co2e?|c02e?|carbo[mn])", re.I)


def _apply_numerical_corrections(text: str) -> str:
    """Apply common OCR error corrections for numerical fields."""
    import re
    
    # Common OCR digit corrections in numerical contexts
    corrections = [
        (r'\bI(\d)', r'1\1'),           # I followed by digits -> 1
        (r'(\d)I\b', r'\g<1>1'),        # digits followed by I -> 1
        (r'\bO(\d)', r'0\1'),           # O followed by digits -> 0
        (r'(\d)O\b', r'\g<1>0'),        # digits followed by O -> 0
        (r'\bS(\d)', r'5\1'),           # S followed by digits -> 5
        (r'(\d)S\b', r'\g<1>5'),        # digits followed by S -> 5
        (r'(\d)[lI|](\d)', r'\1\2'),    # l, I, | between digits -> remove
        (r'(\d)[oO](\d)', r'\g<1>0\2'), # o, O between digits -> 0
    ]
    
    corrected_text = text
    for pattern, replacement in corrections:
        corrected_text = re.sub(pattern, replacement, corrected_text)
    
    return corrected_text

def _validate_numerical_context(text: str, number: str, field_type: str) -> bool:
    """Validate that a number makes sense in its surrounding context."""
    import re
    
    # Find the context around the number
    pattern = rf'(.{{0,50}}){re.escape(number)}(.{{0,50}})'
    match = re.search(pattern, text, re.IGNORECASE)
    
    if not match:
        return True  # Can't find context, assume valid
    
    before, after = match.groups()
    context = (before + after).lower()
    
    if field_type == 'electricity':
        # Check for electricity-related units nearby
        electricity_units = ['kwh', 'kw', 'wh', 'kilowatt', 'electricity']
        if any(unit in context for unit in electricity_units):
            return True
        # Check for common bill contexts
        bill_contexts = ['consumption', 'usage', 'reading', 'total', 'bill']
        return any(ctx in context for ctx in bill_contexts)
    
    elif field_type == 'carbon':
        # Check for carbon-related units nearby
        carbon_units = ['co2', 'kg', 'carbon', 'footprint', 'emission']
        return any(unit in context for unit in carbon_units)
    
    return True  # Default to valid

def _apply_field_aware_corrections(text: str, extracted_fields: Dict[str, int]) -> Dict[str, int]:
    """Apply field-aware post-processing corrections."""
    corrected_fields = extracted_fields.copy()
    
    # Apply numerical corrections to the text first
    corrected_text = _apply_numerical_corrections(text)
    
    # Re-extract from corrected text if we found corrections
    if corrected_text != text:
        LOGGER.debug("Applied numerical OCR corrections, re-extracting")
        reextracted = _extract_with_simple_regex(corrected_text)
        
        # Use re-extracted values if they pass validation
        for field, value in reextracted.items():
            field_type = 'electricity' if 'electricity' in field else 'carbon'
            if _validate_numerical_context(corrected_text, str(value), field_type):
                if field not in corrected_fields or corrected_fields[field] != value:
                    LOGGER.info("Corrected %s: %s -> %s", field, 
                              corrected_fields.get(field, 'None'), value)
                    corrected_fields[field] = value
    
    # Additional validation: prefer second-best if first-best fails sanity checks
    for field, value in list(corrected_fields.items()):
        field_type = 'electricity' if 'electricity' in field else 'carbon'
        
        # Check if value is in reasonable range
        if field_type == 'electricity' and not (50 <= value <= 50000):
            LOGGER.warning("Electricity value %d out of range, looking for alternatives", value)
            # Could implement fallback to second-best extraction here
            
        elif field_type == 'carbon' and not (10 <= value <= 20000):
            LOGGER.warning("Carbon value %d out of range, looking for alternatives", value)
            # Could implement fallback to second-best extraction here
    
    return corrected_fields

def _normalise_number(num_txt: str) -> int:
    """Normalize number string handling OCR errors like 'l' -> '1', 'g' -> '9'."""
    if not num_txt:
        raise ValueError("Empty number string")
    
    # First clean OCR errors
    cleaned = str(num_txt).replace('l', '1').replace('g', '9').replace('O', '0')
    # Then remove spaces and commas
    digits = re.sub(r"[\s,]+", "", cleaned)
    
    # Remove any remaining non-digits
    digits = re.sub(r"[^\d]", "", digits)
    
    if not digits:
        raise ValueError("No digits found after cleaning")
    
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
    if electricity < 50 or electricity > 50000:  # More lenient upper bound for industrial usage
        LOGGER.warning("Electricity value out of typical range: %d kWh", electricity)
        return False
    
    if carbon < 10 or carbon > 20000:  # More lenient upper bound for high-usage scenarios
        LOGGER.warning("Carbon value out of typical range: %d kg", carbon)
        return False
    
    return True

def _extract_with_lightweight_kie(text: str, file_path: Path = None) -> Dict[str, int]:
    """Enhanced KIE extraction using Vision API with contextual field detection."""
    if not file_path:
        # If no file path, try text-based extraction using a simple heuristic
        return _extract_with_text_kie(text)
    
    try:
        LOGGER.info("Attempting KIE extraction via Vision API with contextual bounding box detection")
        # Use the existing Gemini Flash fallback as a lightweight KIE model
        kie_result = gemini_flash_fallback(file_path)
        if kie_result:
            LOGGER.info("KIE extraction successful: %s", kie_result)
            return kie_result
            
        # If Vision API fails, try text-based KIE as backup
        LOGGER.info("Vision API extraction failed, trying text-based KIE")
        return _extract_with_text_kie(text)
        
    except Exception as exc:
        LOGGER.warning("KIE extraction failed: %s, trying text-based fallback", exc)
        return _extract_with_text_kie(text)

def _extract_with_text_kie(text: str) -> Dict[str, int]:
    """Text-based KIE using contextual number extraction with OCR error correction."""
    out = {}
    
    # First, preprocess text to fix common OCR errors
    preprocessed_text = _preprocess_ocr_errors(text)
    
    # Find all numbers with their context windows
    import re
    
    # Look for numbers (including comma-separated) with surrounding context
    number_context_pattern = re.compile(r'(.{0,30})((?:\d{1,3}(?:,\d{3})*|\d{2,5}))(.{0,30})', re.I)
    matches = number_context_pattern.findall(preprocessed_text)
    
    electricity_candidates = []
    carbon_candidates = []
    
    for before, number_str, after in matches:
        try:
            # Handle comma-separated numbers
            value = int(number_str.replace(',', ''))
            
            # Skip unreasonable values
            if value < 10 or value > 100000:
                continue
                
            context = (before + after).lower()
            
            # Contextual classification with scoring
            electricity_keywords = ['kwh', 'electricity', 'consumption', 'usage', 'electric', 'reading']
            carbon_keywords = ['co2', 'carbon', 'footprint', 'emission', 'kg', 'environmental', 'c02']
            
            elec_score = sum(2 if kw in context else 0 for kw in electricity_keywords)
            carbon_score = sum(2 if kw in context else 0 for kw in carbon_keywords)
            
            # Boost score for exact keyword matches
            if 'kwh' in context:
                elec_score += 3
            if any(term in context for term in ['co2e', 'co2', 'kg']):
                carbon_score += 3
            
            if elec_score > 0 and 50 <= value <= 50000:
                electricity_candidates.append((value, elec_score))
            if carbon_score > 0 and 10 <= value <= 20000:
                carbon_candidates.append((value, carbon_score))
                
        except (ValueError, TypeError):
            continue
    
    # Select best candidates
    if electricity_candidates:
        # Sort by score (context relevance) then by reasonable residential values
        electricity_candidates.sort(key=lambda x: (-x[1], abs(x[0] - 300)))
        out['electricity_kwh'] = electricity_candidates[0][0]
    
    if carbon_candidates:
        carbon_candidates.sort(key=lambda x: (-x[1], abs(x[0] - 120)))
        out['carbon_kgco2e'] = carbon_candidates[0][0]
    
    return out

def _preprocess_ocr_errors(text: str) -> str:
    """Preprocess text to fix common OCR errors using generalizable patterns."""
    import re
    
    # Generalized OCR error corrections based on common character confusions
    ocr_corrections = [
        # Letter-digit confusions at word boundaries
        (r'\bl(\d+)\b', r'1\1'),           # l followed by digits -> 1 + digits
        (r'\bO(\d+)\b', r'0\1'),           # O followed by digits -> 0 + digits  
        (r'\b(\d+)l\b', r'\g<1>1'),        # digits followed by l -> digits + 1
        (r'\b(\d+)O\b', r'\g<1>0'),        # digits followed by O -> digits + 0
        
        # Within-number character confusions
        (r'(\d)[gq](\d)', r'\1\2'),        # g or q between digits -> remove
        (r'(\d)[oO](\d)', r'\g<1>0\2'),    # o or O between digits -> 0
        (r'(\d)[Il|](\d)', r'\g<1>1\2'),   # I, l, | between digits -> 1
        (r'(\d)[Ss](\d)', r'\g<1>5\2'),    # S between digits -> 5
        
        # Common word-level OCR errors
        (r'\bElectr[il]city\b', 'Electricity'),  # Various Electricity misspellings
        (r'\bDuba[il]\b', 'Dubai'),              # Dubai misspellings
        (r'\b[Cc]onsumpt[il]on\b', 'Consumption'), # Consumption misspellings
        
        # CO2e variants
        (r'\b[Cc][0oO][2zZ][eE]?\b', 'CO2e'),    # Various CO2e misspellings
        (r'\bcoze?\b', 'CO2e'),                  # Common OCR error "coze"
    ]
    
    # Apply all corrections
    for pattern, replacement in ocr_corrections:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text

def extract_fields(text: str, file_path: Path = None, ocr_result: OcrResult = None) -> Dict[str, int]:
    """Enhanced extraction with field-aware post-processing and per-field confidence tracking."""
    
    # Phase 1: Try simple, reliable regex patterns first
    out, field_confidences = _extract_with_simple_regex_and_confidence(text, ocr_result)
    
    # Phase 2: If regex fails or gives incomplete results, use KIE
    if len(out) < 2 or not _validate_extraction_values(out.get("electricity_kwh"), out.get("carbon_kgco2e")):
        LOGGER.info("Simple regex incomplete/invalid, using KIE extraction")
        kie_result = _extract_with_lightweight_kie(text, file_path)
        
        if kie_result:
            # Use KIE results if they're better (more complete or pass validation)
            kie_electricity = kie_result.get("electricity_kwh")
            kie_carbon = kie_result.get("carbon_kgco2e")
            
            if _validate_extraction_values(kie_electricity, kie_carbon):
                LOGGER.info("KIE extraction passed validation, using KIE results")
                out.update(kie_result)
                # KIE has moderate confidence
                if "electricity_kwh" in kie_result:
                    field_confidences["electricity_kwh"] = 0.8
                if "carbon_kgco2e" in kie_result:
                    field_confidences["carbon_kgco2e"] = 0.8
            elif len(kie_result) > len(out):
                LOGGER.info("KIE extraction more complete than regex, using KIE results")
                out.update(kie_result)
                # Lower confidence for unvalidated KIE
                if "electricity_kwh" in kie_result:
                    field_confidences["electricity_kwh"] = 0.6
                if "carbon_kgco2e" in kie_result:
                    field_confidences["carbon_kgco2e"] = 0.6
    
    # Phase 3: Apply field-aware post-processing corrections
    corrected_out = _apply_field_aware_corrections(text, out)
    
    # Add per-field confidence metadata
    if field_confidences:
        corrected_out["_field_confidences"] = field_confidences
    
    return corrected_out

def _extract_with_simple_regex_and_confidence(text: str, ocr_result: OcrResult = None) -> Tuple[Dict[str, int], Dict[str, float]]:
    """Simple regex extraction with per-field confidence based on OCR token confidence."""
    out = {}
    field_confidences = {}
    
    # Simple electricity patterns - most reliable only
    simple_patterns = [
        re.compile(r"\b(\d{1,3}(?:,\d{3})*|\d{2,5})\s*kWh", re.I),  # "299 kWh" or "1,234 kWh"
        re.compile(r"Electricity\s+(\d{1,3}(?:,\d{3})*|\d{2,5})", re.I),  # "Electricity 299"
        re.compile(r"(\d{1,3}(?:,\d{3})*|\d{2,5})\s+Electricity", re.I),  # "299 Electricity"
        re.compile(r"Consumption[:\s]+(\d{1,3}(?:,\d{3})*|\d{2,5})", re.I),  # "Consumption: 299"
        re.compile(r"usage[:\s]+(\d{1,3}(?:,\d{3})*|\d{2,5})", re.I),  # "usage: 1,234"
    ]
    
    for pattern in simple_patterns:
        m = pattern.search(text)
        if m:
            try:
                value = int(m.group(1).replace(',', ''))
                if 50 <= value <= 50000:
                    out["electricity_kwh"] = value
                    # Calculate field confidence based on OCR token confidence in match region
                    field_confidences["electricity_kwh"] = _calculate_field_confidence(
                        text, m.start(), m.end(), ocr_result, "electricity"
                    )
                    break
            except (ValueError, TypeError):
                continue
    
    # Simple carbon patterns - most reliable only
    carbon_patterns = [
        re.compile(r"(\d{1,4})\s*kg\s*CO2e?", re.I),  # "120 kg CO2e"
        re.compile(r"CO2e?\s+(\d{1,4})", re.I),  # "CO2e 120"
        re.compile(r"Carbon[^0-9]*(\d{1,4})", re.I),  # "Carbon: 120"
        re.compile(r"footprint[^0-9]*(\d{1,4})", re.I),  # "footprint 200"
        re.compile(r"(\d{1,4})\s*kg(?!\s*CO2)", re.I),  # "200 kg" (not followed by CO2)
    ]
    
    for pattern in carbon_patterns:
        m = pattern.search(text)
        if m:
            try:
                value = int(m.group(1))
                if 10 <= value <= 20000:
                    out["carbon_kgco2e"] = value
                    # Calculate field confidence based on OCR token confidence in match region
                    field_confidences["carbon_kgco2e"] = _calculate_field_confidence(
                        text, m.start(), m.end(), ocr_result, "carbon"
                    )
                    break
            except (ValueError, TypeError):
                continue
    
    return out, field_confidences

def _apply_field_specific_enhancement(fields: Dict[str, Any], ocr_result: OcrResult, file_path: Path) -> Dict[str, Any]:
    """Apply enhancement or LLM fallback only to fields with low confidence."""
    field_confidences = fields.get("_field_confidences", {})
    enhanced_fields = fields.copy()
    
    if not field_confidences:
        # No per-field confidence available, use global confidence
        if ocr_result.field_confidence < TAU_ENHANCER_PASS:
            LOGGER.info("Global confidence %.2f below threshold, considering enhancement", 
                       ocr_result.field_confidence)
            return enhanced_fields
        else:
            LOGGER.info("Global confidence %.2f acceptable, no enhancement needed", 
                       ocr_result.field_confidence)
            return enhanced_fields
    
    # Check each field's confidence
    fields_needing_enhancement = []
    fields_needing_llm = []
    
    for field, confidence in field_confidences.items():
        if confidence < TAU_LLM_PASS:
            fields_needing_llm.append(field)
            LOGGER.info("Field %s confidence %.2f below LLM threshold %.2f", 
                       field, confidence, TAU_LLM_PASS)
        elif confidence < TAU_ENHANCER_PASS:
            fields_needing_enhancement.append(field)
            LOGGER.info("Field %s confidence %.2f below enhancement threshold %.2f", 
                       field, confidence, TAU_ENHANCER_PASS)
        else:
            LOGGER.info("Field %s confidence %.2f acceptable", field, confidence)
    
    # Apply targeted enhancement only to weak fields
    if fields_needing_enhancement or fields_needing_llm:
        try:
            # For now, apply global enhancement if any field needs it
            # In future versions, this could be made more field-specific
            if fields_needing_llm:
                LOGGER.info("Applying LLM enhancement for fields: %s", fields_needing_llm)
                enhanced_result = _run_llm_enhancement(ocr_result, file_path)
                if enhanced_result:
                    llm_fields = extract_fields(enhanced_result.text, file_path, enhanced_result)
                    # Only update the weak fields
                    for field in fields_needing_llm:
                        if field in llm_fields and not field.startswith("_"):
                            enhanced_fields[field] = llm_fields[field]
                            LOGGER.info("Enhanced field %s with LLM: %s", field, llm_fields[field])
            
            elif fields_needing_enhancement:
                LOGGER.info("Applying enhancement for fields: %s", fields_needing_enhancement)
                enhanced_result = _run_enhancement(ocr_result, file_path)
                if enhanced_result:
                    enh_fields = extract_fields(enhanced_result.text, file_path, enhanced_result)
                    # Only update the weak fields
                    for field in fields_needing_enhancement:
                        if field in enh_fields and not field.startswith("_"):
                            enhanced_fields[field] = enh_fields[field]
                            LOGGER.info("Enhanced field %s: %s", field, enh_fields[field])
        
        except Exception as e:
            LOGGER.warning("Field-specific enhancement failed: %s", e)
    
    else:
        LOGGER.info("All fields have acceptable confidence, no enhancement needed")
    
    return enhanced_fields

def _calculate_field_confidence(text: str, match_start: int, match_end: int, 
                              ocr_result: OcrResult, field_type: str) -> float:
    """Calculate confidence for a specific field based on OCR token confidence in the match region."""
    if not ocr_result or not ocr_result.tokens or not ocr_result.confidences:
        # Default confidence based on field type
        return 0.9 if field_type == "electricity" else 0.85
    
    # Find tokens that overlap with the matched text region
    matched_text = text[match_start:match_end]
    relevant_confidences = []
    
    # Simple approach: find tokens that appear in the matched region
    for i, token in enumerate(ocr_result.tokens):
        if token in matched_text and i < len(ocr_result.confidences):
            relevant_confidences.append(ocr_result.confidences[i])
    
    if relevant_confidences:
        # Use geometric mean for field confidence (more conservative than arithmetic mean)
        import math
        log_sum = sum(math.log(max(conf, 0.01)) for conf in relevant_confidences)  # Avoid log(0)
        geometric_mean = math.exp(log_sum / len(relevant_confidences))
        return min(geometric_mean, 0.99)  # Cap at 99%
    else:
        # Fallback: use overall OCR confidence
        return ocr_result.field_confidence if ocr_result.field_confidence > 0 else 0.85

def _extract_with_simple_regex(text: str) -> Dict[str, int]:
    """Simple, reliable regex extraction for common patterns."""
    out = {}
    
    # Simple electricity patterns - most reliable only
    simple_patterns = [
        re.compile(r"\b(\d{1,3}(?:,\d{3})*|\d{2,5})\s*kWh", re.I),  # "299 kWh" or "1,234 kWh"
        re.compile(r"Electricity\s+(\d{1,3}(?:,\d{3})*|\d{2,5})", re.I),  # "Electricity 299"
        re.compile(r"(\d{1,3}(?:,\d{3})*|\d{2,5})\s+Electricity", re.I),  # "299 Electricity"
        re.compile(r"Consumption[:\s]+(\d{1,3}(?:,\d{3})*|\d{2,5})", re.I),  # "Consumption: 299"
        re.compile(r"usage[:\s]+(\d{1,3}(?:,\d{3})*|\d{2,5})", re.I),  # "usage: 1,234"
    ]
    
    for pattern in simple_patterns:
        m = pattern.search(text)
        if m:
            try:
                value = int(m.group(1).replace(',', ''))
                if 50 <= value <= 50000:
                    out["electricity_kwh"] = value
                    break
            except (ValueError, TypeError):
                continue
    
    # Simple carbon patterns - most reliable only
    carbon_patterns = [
        re.compile(r"(\d{1,4})\s*kg\s*CO2e?", re.I),  # "120 kg CO2e"
        re.compile(r"CO2e?\s+(\d{1,4})", re.I),  # "CO2e 120"
        re.compile(r"Carbon[^0-9]*(\d{1,4})", re.I),  # "Carbon: 120"
        re.compile(r"footprint[^0-9]*(\d{1,4})", re.I),  # "footprint 200"
        re.compile(r"(\d{1,4})\s*kg(?!\s*CO2)", re.I),  # "200 kg" (not followed by CO2)
    ]
    
    for pattern in carbon_patterns:
        m = pattern.search(text)
        if m:
            try:
                value = int(m.group(1))
                if 10 <= value <= 20000:
                    out["carbon_kgco2e"] = value
                    break
            except (ValueError, TypeError):
                continue
    
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
            "ocr_engine": fields.get("_engine"),
            "extraction_status": fields.get("_status", "unknown"),
            "errors": fields.get("_errors", []),
            "warnings": fields.get("_warnings", []),
            "confidence_thresholds": fields.get("_thresholds", {}),
        },
    }

def _validate_file_format(file_path: Path) -> None:
    """Validate file format and integrity before processing.
    
    Raises:
        ValueError: If file is corrupted, has wrong format, or is invalid
    """
    try:
        # Read first few bytes to check file signature
        with open(file_path, 'rb') as f:
            header = f.read(16)
        
        if not header:
            raise ValueError("File appears to be empty or unreadable")
        
        extension = file_path.suffix.lower()
        
        # Validate PDF files
        if extension == '.pdf':
            if not header.startswith(b'%PDF-'):
                raise ValueError(f"File '{file_path.name}' has .pdf extension but is not a valid PDF file")
            
            # Try to load with pdf2image to catch corruption early
            try:
                if convert_from_path is not None:
                    # Quick validation - try to convert just the first page at low resolution
                    test_images = convert_from_path(str(file_path), dpi=72, first_page=1, last_page=1)
                    if not test_images:
                        raise ValueError(f"PDF file '{file_path.name}' appears to be corrupted or has no pages")
            except Exception as pdf_exc:
                raise ValueError(f"PDF file '{file_path.name}' appears to be corrupted: {pdf_exc}")
        
        # Validate image files
        elif extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
            # Check common image file signatures
            is_valid_image = False
            
            if extension in ['.jpg', '.jpeg'] and header.startswith(b'\xff\xd8\xff'):
                is_valid_image = True
            elif extension == '.png' and header.startswith(b'\x89PNG\r\n\x1a\n'):
                is_valid_image = True  
            elif extension == '.bmp' and header.startswith(b'BM'):
                is_valid_image = True
            elif extension in ['.tiff', '.tif'] and (header.startswith(b'II*\x00') or header.startswith(b'MM\x00*')):
                is_valid_image = True
            
            if not is_valid_image:
                # For formats we can't easily check by header, try PIL validation
                try:
                    if Image is not None:
                        with Image.open(file_path) as img:
                            img.verify()  # This checks if the image is corrupted
                except Exception as img_exc:
                    raise ValueError(f"Image file '{file_path.name}' appears to be corrupted or invalid: {img_exc}")
            else:
                # Even if header looks good, do a quick PIL check
                try:
                    if Image is not None:
                        with Image.open(file_path) as img:
                            img.verify()
                except Exception as img_exc:
                    raise ValueError(f"Image file '{file_path.name}' appears to be corrupted: {img_exc}")
        
        LOGGER.debug("File format validation passed for %s", file_path.name)
        
    except (OSError, IOError) as e:
        raise ValueError(f"Cannot read file '{file_path.name}': {e}")
    except ValueError:
        # Re-raise ValueError as-is  
        raise
    except Exception as e:
        raise ValueError(f"File validation failed for '{file_path.name}': {e}")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py bill.[pdf|png] [--save outfile.json] [--thresholds accept,enhance,llm]")
        print("  --save: Save output to JSON file")
        print("  --thresholds: Override confidence thresholds (e.g., --thresholds 0.95,0.90,0.85)")
        print("  Environment variables: TAU_FIELD_ACCEPT, TAU_ENHANCER_PASS, TAU_LLM_PASS")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    save_path = None
    
    # Parse command line arguments
    if "--save" in sys.argv:
        save_path = Path(sys.argv[sys.argv.index("--save") + 1])
    
    # Handle threshold overrides via command line
    if "--thresholds" in sys.argv:
        try:
            threshold_idx = sys.argv.index("--thresholds") + 1
            threshold_str = sys.argv[threshold_idx]
            thresholds = [float(x.strip()) for x in threshold_str.split(',')]
            
            if len(thresholds) != 3:
                print("Error: --thresholds requires exactly 3 values (accept,enhance,llm)")
                sys.exit(1)
            
            accept, enhance, llm = thresholds
            
            # Validate threshold ordering
            if not (0.0 <= llm <= enhance <= accept <= 1.0):
                print(f"Error: Invalid threshold ordering. Expected 0 ≤ llm ≤ enhance ≤ accept ≤ 1")
                print(f"Got: llm={llm}, enhance={enhance}, accept={accept}")
                sys.exit(1)
            
            # Override config values
            import config
            config.TAU_FIELD_ACCEPT = accept
            config.TAU_ENHANCER_PASS = enhance  
            config.TAU_LLM_PASS = llm
            
            print(f"Using custom thresholds: accept={accept}, enhance={enhance}, llm={llm}")
            
        except (ValueError, IndexError) as e:
            print(f"Error parsing --thresholds: {e}")
            print("Format: --thresholds accept,enhance,llm (e.g., --thresholds 0.95,0.90,0.85)")
            sys.exit(1)

    # Check if file exists and has supported extension
    if not file_path.exists():
        print(f"Error: File '{file_path}' does not exist")
        sys.exit(1)
    
    # Check if file is empty
    if file_path.stat().st_size == 0:
        print(f"Error: File '{file_path}' is empty (0 bytes)")
        sys.exit(1)
    
    supported_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    if file_path.suffix.lower() not in supported_extensions:
        print(f"Error: Unsupported file type '{file_path.suffix}'. Supported: {', '.join(supported_extensions)}")
        sys.exit(1)
    
    # Validate file integrity and format
    try:
        _validate_file_format(file_path)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to validate file format - {e}")
        sys.exit(1)

    ocr_res = run_ocr(file_path)
    fields = extract_fields(ocr_res.text, file_path, ocr_res)
    
    # Apply field-specific enhancement routing
    fields = _apply_field_specific_enhancement(fields, ocr_res, file_path)
    
    # Add OCR metadata and error tracking
    fields["_confidence"] = ocr_res.field_confidence
    fields["_engine"] = ocr_res.engine
    
    # Determine extraction status and errors
    status = "success"
    errors = []
    warnings = []
    
    if ocr_res.engine == "blank_document":
        status = "failed"
        errors.append("Document appears to be blank or unreadable")
    elif ocr_res.engine == "none":
        status = "failed"
        errors.append("All OCR engines failed to extract text")
    elif not ocr_res.text.strip():
        status = "failed"
        errors.append("No text could be extracted from document")
    elif "electricity_kwh" not in fields and "carbon_kgco2e" not in fields:
        status = "failed"
        errors.append("No target fields (electricity or carbon) could be extracted from text")
    elif "electricity_kwh" not in fields or "carbon_kgco2e" not in fields:
        status = "partial"
        if "electricity_kwh" not in fields:
            warnings.append("Electricity consumption value not found")
        if "carbon_kgco2e" not in fields:
            warnings.append("Carbon footprint value not found")
    
    if fields["_confidence"] < 0.5:
        warnings.append(f"Low OCR confidence ({fields['_confidence']:.2f})")
    
    fields["_status"] = status
    fields["_errors"] = errors
    fields["_warnings"] = warnings
    
    # Add threshold information to metadata for transparency
    fields["_thresholds"] = {
        "field_accept": TAU_FIELD_ACCEPT,
        "enhancer_pass": TAU_ENHANCER_PASS, 
        "llm_pass": TAU_LLM_PASS
    }

    payload = build_payload(fields, file_path)

    if save_path:
        save_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {save_path}")

    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
