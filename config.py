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

# --- Tesseract Configuration -------------------------------------------------
TESSERACT_OEM     = 3     # OCR Engine Mode
TESSERACT_PSM     = 6     # Page Segmentation Mode

# --- Confidence thresholds ---------------------------------------------------
TAU_FIELD_ACCEPT  = 0.95  # auto-accept threshold
TAU_ENHANCER_PASS = 0.90  # after enhancer / alt engine
TAU_LLM_PASS      = 0.85  # LLM fallback

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

