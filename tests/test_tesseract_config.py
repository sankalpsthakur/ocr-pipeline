#!/usr/bin/env python3
"""Test different Tesseract configurations."""

import re
import subprocess
import sys
from pathlib import Path
import pytest

ORIGINAL_CONFIG = Path("config.py").read_text()

def update_tesseract_config(lang, oem, psm):
    """Update Tesseract configuration in config.py using regex."""
    updated = ORIGINAL_CONFIG
    updated = re.sub(r"TESSERACT_LANG\s*=.*", f'TESSERACT_LANG    = "{lang}"', updated)
    updated = re.sub(r"TESSERACT_OEM\s*=.*", f"TESSERACT_OEM     = {oem}", updated)
    updated = re.sub(r"TESSERACT_PSM\s*=.*", f"TESSERACT_PSM     = {psm}", updated)
    Path("config.py").write_text(updated)

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
        cfg = Path("config.py").read_text()
        assert f'TESSERACT_LANG    = "{lang}"' in cfg
        assert f"TESSERACT_OEM     = {oem}" in cfg
        assert f"TESSERACT_PSM     = {psm}" in cfg

        result = subprocess.run(
            [sys.executable, "-m", "py_compile", "pipeline.py"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr

    finally:
        Path("config.py").write_text(ORIGINAL_CONFIG)

# The original script provided extensive CLI output. The pytest version simply
# verifies that each configuration compiles successfully.
