
"""Central configuration – hard‑coded secrets & thresholds"""


# --- OCR back‑end selection --------------------------------------------------
# Options: "tesseract", "easyocr", "paddleocr", "gcv", "azure"
OCR_BACKEND       = "easyocr"

# --- Confidence thresholds ---------------------------------------------------
TAU_FIELD_ACCEPT  = 0.95  # auto‑accept threshold
TAU_ENHANCER_PASS = 0.9  # after enhancer / alt engine
TAU_LLM_PASS      = 0.85  # LLM fallback

# --- Misc --------------------------------------------------------------------
MAX_PAGES         = 20    # safety cap to avoid 100‑page uploads
DPI_PRIMARY       = 300
DPI_ENHANCED      = 600

# --- Tesseract options -------------------------------------------------------
# Language, OCR engine mode and page segmentation mode can be tuned depending
# on the type of documents processed. They are exposed here so that pipeline
# users can adjust them without touching the codebase.
TESSERACT_LANG    = "eng"
TESSERACT_OEM     = 3     # OCR Engine Mode
TESSERACT_PSM     = 6     # Page Segmentation Mode



