"""Minimal configuration for OCR pipeline"""

import os

# Language settings
OCR_LANG = "eng"
PADDLEOCR_LANG = "en"

# Confidence thresholds
TAU_FIELD_ACCEPT = 0.95
TAU_ENHANCER_PASS = 0.90
TAU_LLM_PASS = 0.85

# Performance settings
DPI_PRIMARY = 300
DPI_ENHANCED = 600
MAX_WORKER_THREADS = 3
AUTO_THREAD_COUNT = True
ENABLE_PADDLEOCR = True
GPU_AVAILABLE = False

# Cache settings
MAX_CACHE_SIZE_MB = 500
MAX_IMAGE_WIDTH = 4096
MAX_IMAGE_HEIGHT = 4096

# API Keys (empty for testing)
MISTRAL_API_KEY = ""
DATALAB_API_KEY = ""
GEMINI_API_KEY = ""

# Document settings
DOCUMENT_TYPE = "bill"
PADDLEOCR_ARGS = {
    "default": {"det_limit_side_len": 960, "rec_batch_num": 6},
    "bill": {"det_limit_side_len": 960, "rec_batch_num": 6}
}