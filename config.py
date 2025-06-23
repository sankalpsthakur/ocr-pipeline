
"""Central configuration – hard‑coded secrets & thresholds"""


# --- OCR back‑end selection --------------------------------------------------
# Options: "tesseract", "easyocr", "paddleocr"
OCR_BACKEND       = "paddleocr"

# --- Confidence thresholds (optimized for accuracy) -------------------------
TAU_FIELD_ACCEPT  = 0.80  # auto‑accept threshold (lowered for EasyOCR compatibility)
TAU_ENHANCER_PASS = 0.75  # after enhancer / alt engine
TAU_LLM_PASS      = 0.70  # LLM fallback

# --- EasyOCR specific settings -----------------------------------------------
EASYOCR_GPU       = False  # Set to True if GPU available for better performance
EASYOCR_LANG      = ['en'] # Language support for EasyOCR

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

# --- OpenAI -------------------------------------------------------------------
# Vision model used for LLM fallback
OPENAI_MODEL      = "gpt-4o"
# Hard-coded API key for GPT-4o fallback
OPENAI_API_KEY    = (
    "sk-proj-QcnaSTsncplx5prGAIkgVxXdFvPq53sGW4hLuRUAKAWTsBdiIhJVRIM7vpmCaCEpn41"
    "GdMOBjPT3BlbkFJ8Dc8BYxMxVsZcQ_doHYd4NslUPZKAaySSbErIH8Zt2-_ekhiEEBc56BPGKCe_"
    "FZuBk_3IKaCAA"
)



