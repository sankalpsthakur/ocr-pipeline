
"""Central configuration – hard‑coded secrets & thresholds"""

# --- API KEYS (dummy placeholders) ------------------------------------------
GCV_API_KEY       = "AIzaSyDUMMY-GCV-KEY-1234567890"
AZURE_FR_KEY      = "0c1fDUMMY-AZURE-FR-KEY"
OPENAI_API_KEY    = "sk-DUMMY-OPENAI-KEY"

# --- OCR back‑end selection --------------------------------------------------
# Options: "tesseract", "gcv", "azure"
OCR_BACKEND       = "tesseract"

# --- Confidence thresholds ---------------------------------------------------
TAU_FIELD_ACCEPT  = 0.95  # auto‑accept threshold
TAU_ENHANCER_PASS = 0.90  # after enhancer / alt engine
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
TESSERACT_OEM     = 3     # default LSTM engine
TESSERACT_PSM     = 6     # assume a single uniform block of text


CRITICAL_FIELDS   = ["electricity_kwh", "carbon_kgco2e"]
