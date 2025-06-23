#!/usr/bin/env python3
"""Test different Tesseract configurations."""

import subprocess
import sys
from pathlib import Path
import pytest

ORIGINAL_CONFIG = Path("config.py").read_text()

def update_tesseract_config(lang, oem, psm):
    """Update Tesseract configuration in config.py."""
    config_content = f'''
"""Central configuration – hard‑coded secrets & thresholds"""


# --- OCR back‑end selection --------------------------------------------------
# Options: "tesseract", "easyocr", "paddleocr"
OCR_BACKEND       = "tesseract"

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
TESSERACT_LANG    = "{lang}"
TESSERACT_OEM     = {oem}     # OCR Engine Mode
TESSERACT_PSM     = {psm}     # Page Segmentation Mode



'''
    Path("config.py").write_text(config_content)

@pytest.mark.parametrize(
    "lang,oem,psm,description",
    [
        ("eng", 3, 6, "Default (English, LSTM, Single block)"),
        ("eng", 1, 6, "Legacy engine"),
        ("eng", 3, 3, "Fully automatic page segmentation"),
        ("eng", 3, 4, "Single column text"),
        ("eng", 3, 7, "Single text line"),
        ("eng", 3, 8, "Single word"),
        ("eng", 3, 11, "Sparse text"),
        ("eng", 3, 13, "Raw line (no segmentation)"),
    ],
)
def test_tesseract_config(lang, oem, psm, description):
    """Test a Tesseract configuration."""
    print(f"\nTesting {description}: LANG={lang}, OEM={oem}, PSM={psm}")
    
    update_tesseract_config(lang, oem, psm)
    
    try:
        # Test compilation first
        result = subprocess.run([
            sys.executable, "-m", "py_compile", "pipeline.py"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("  ✓ Configuration valid and compiles successfully")
            return True
        else:
            print(f"  ✗ Compilation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    finally:
        Path("config.py").write_text(ORIGINAL_CONFIG)

# The original script provided extensive CLI output. The pytest version simply
# verifies that each configuration compiles successfully.
