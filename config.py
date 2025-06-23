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

# --- OpenAI -------------------------------------------------------------------
# Vision model used for LLM fallback
OPENAI_MODEL      = "gpt-4o"
# Hard-coded API key for GPT-4o fallback
OPENAI_API_KEY    = "sk-proj-REPLACE_WITH_YOUR_OPENAI_API_KEY_HERE"

