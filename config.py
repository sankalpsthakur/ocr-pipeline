"""Configuration settings for the OCR pipeline.

This configuration provides centralized settings for the OCR pipeline,
allowing users to control various aspects of text extraction without
modifying the main pipeline code.
"""

import torch
import subprocess
import psutil
import os

def _detect_gpu_availability():
    """Auto-detect GPU availability for OCR engines."""
    gpu_available = False
    gpu_memory_gb = 0
    
    # Check for CUDA
    if torch.cuda.is_available():
        gpu_available = True
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    # Check for Apple Metal (MPS)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        gpu_available = True
        # Estimate Metal memory (simplified)
        gpu_memory_gb = 8.0  # Conservative estimate for Apple Silicon
    
    return gpu_available, gpu_memory_gb

def _detect_system_memory():
    """Detect available system RAM."""
    return psutil.virtual_memory().total / (1024**3)

# Auto-detect hardware capabilities
_GPU_AVAILABLE, _GPU_MEMORY = _detect_gpu_availability()
_SYSTEM_MEMORY = _detect_system_memory()

# Maximum number of PDF pages to process
MAX_PAGES = 3

# --- OCR Engine Configuration ------------------------------------------------
# Primary OCR backend: "tesseract", "easyocr", or "paddleocr"
OCR_BACKEND       = "tesseract"

# Document type for engine-specific tuning ("bills", "default")
DOCUMENT_TYPE     = "bills"

# Language settings for different OCR engines
OCR_LANG          = None           # Override for all engines (e.g., "eng", "fra")
TESSERACT_LANG    = "eng"          # Tesseract-specific language
EASYOCR_LANG      = ["en"]         # EasyOCR language list
PADDLEOCR_LANG    = "en"           # PaddleOCR language code

# EasyOCR GPU acceleration - auto-detected based on hardware
EASYOCR_GPU       = _GPU_AVAILABLE

# Auto-enable GPU optimizations based on detected hardware
USE_LIGHTWEIGHT_MODELS = _SYSTEM_MEMORY < 16.0  # Use lite models on <16GB RAM systems
ENABLE_PADDLEOCR = _SYSTEM_MEMORY >= 8.0  # Disable PaddleOCR on very low memory systems

# Image processing settings
DPI_PRIMARY       = 300   # Primary DPI for PDF conversion
DPI_ENHANCED      = 600   # Enhanced DPI for low-confidence extractions

# --- Performance Optimization Settings ------------------------------------------
# These control speed vs quality tradeoffs

# Maximum image dimensions for resizing (0 to disable resizing)
MAX_IMAGE_WIDTH   = int(os.getenv("MAX_IMAGE_WIDTH", "2000"))   # Max width in pixels
MAX_IMAGE_HEIGHT  = int(os.getenv("MAX_IMAGE_HEIGHT", "2000"))  # Max height in pixels

# Thread pool optimization
AUTO_THREAD_COUNT = os.getenv("AUTO_THREAD_COUNT", "true").lower() == "true"  # Auto-detect optimal thread count
MAX_WORKER_THREADS = int(os.getenv("MAX_WORKER_THREADS", "4"))  # Max concurrent OCR workers

# Image cache limits to prevent memory issues
MAX_CACHE_SIZE_MB = int(os.getenv("MAX_CACHE_SIZE_MB", "500"))  # Max cache size in MB

# --- Tesseract Configuration -------------------------------------------------
TESSERACT_OEM     = 3     # OCR Engine Mode
TESSERACT_PSM     = 6     # Page Segmentation Mode

# Engine-specific tuning configurations
TESSERACT_ARGS = {
    "bills": {
        "psm": 6,  # Uniform block for bills
        "oem": 3,
        "whitelist": "0123456789kKwWhHcCoO2eE.",  # Digits and units
        "config": "--psm 6 -c tessedit_char_whitelist=0123456789kKwWhHcCoO2eE."
    },
    "default": {
        "psm": 6,
        "oem": 3,
        "config": "--oem 3 --psm 6"
    }
}

EASYOCR_ARGS = {
    "bills": {
        "detail": 1,
        "paragraph": False,
        "contrast_ths": 0.05,
        "width_ths": 0.7,
        "height_ths": 0.7
    },
    "default": {
        "detail": 1,
        "paragraph": False,
        "width_ths": 0.7,
        "height_ths": 0.7
    }
}

PADDLEOCR_ARGS = {
    "bills": {
        "rec_image_shape": "3, 32, 640",  # Wider for number strings
        "det_limit_side_len": 960,
        "rec_batch_num": 1
    },
    "default": {
        "rec_image_shape": "3, 32, 320",
        "det_limit_side_len": 320,
        "rec_batch_num": 1
    }
}

# --- Confidence thresholds ---------------------------------------------------
# These control the OCR pipeline's decision points:
# - TAU_FIELD_ACCEPT: High confidence threshold for immediate acceptance
# - TAU_ENHANCER_PASS: Medium confidence threshold for enhanced processing
# - TAU_LLM_PASS: Low confidence threshold for LLM fallback

import os

def _get_float_env_var(var_name: str, default: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Get a float from environment variable with validation."""
    try:
        value = float(os.getenv(var_name, default))
        if not (min_val <= value <= max_val):
            print(f"Warning: {var_name}={value} is out of range [{min_val}, {max_val}], using default {default}")
            return default
        return value
    except (ValueError, TypeError):
        print(f"Warning: Invalid {var_name} value, using default {default}")
        return default

# Load thresholds from environment variables with fallback to defaults
TAU_FIELD_ACCEPT  = _get_float_env_var("TAU_FIELD_ACCEPT", 0.95)   # auto-accept threshold
TAU_ENHANCER_PASS = _get_float_env_var("TAU_ENHANCER_PASS", 0.90)  # enhanced processing threshold  
TAU_LLM_PASS      = _get_float_env_var("TAU_LLM_PASS", 0.85)       # LLM fallback threshold

# Validate threshold ordering (should be descending)
if not (TAU_LLM_PASS <= TAU_ENHANCER_PASS <= TAU_FIELD_ACCEPT):
    print(f"Warning: Threshold ordering is incorrect. Expected TAU_LLM_PASS ≤ TAU_ENHANCER_PASS ≤ TAU_FIELD_ACCEPT")
    print(f"Current: {TAU_LLM_PASS} ≤ {TAU_ENHANCER_PASS} ≤ {TAU_FIELD_ACCEPT}")
    print("Using default values to ensure correct ordering.")
    TAU_FIELD_ACCEPT = 0.95
    TAU_ENHANCER_PASS = 0.90
    TAU_LLM_PASS = 0.85

# --- Google Gemini -----------------------------------------------------------
# Vision model used for LLM fallback
GEMINI_MODEL      = "gemini-2.0-flash"
# API key for Gemini Flash fallback
GEMINI_API_KEY    = "AIzaSyAw1RNympifj3sIZlUvhxVH5trb8Mz6JrA"

# --- Mistral ------------------------------------------------------------------
# Mistral OCR API settings
MISTRAL_API_KEY   = "BJRjnwMi6jvna8zrfIvxyyr9Tf54HYgW"
MISTRAL_MODEL     = "mistral-ocr-latest"

# --- Datalab ------------------------------------------------------------------
# Datalab OCR API settings
DATALAB_API_KEY   = "vfaQi0_wwj5emvhyU2JtFJNtsLKSbkMsvaaD3nBwWLY"
DATALAB_URL       = "https://www.datalab.to/api/v1/ocr"

