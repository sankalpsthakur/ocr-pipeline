"""Configuration settings for the OCR pipeline.

This configuration provides centralized settings for the OCR pipeline,
allowing users to control various aspects of text extraction without
modifying the main pipeline code.
"""

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

# EasyOCR GPU acceleration (set to False for CPU-only)
EASYOCR_GPU       = False

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

