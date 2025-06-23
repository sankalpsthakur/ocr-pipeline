#!/usr/bin/env python3
"""Test different hyperparameter configurations to find optimal settings."""

import json
import re
import subprocess
import sys
from pathlib import Path
import pytest

# Test configurations
configs = [
    # [TAU_FIELD_ACCEPT, TAU_ENHANCER_PASS, TAU_LLM_PASS, DPI_PRIMARY, DPI_ENHANCED]
    [0.95, 0.90, 0.85, 300, 600],  # Original
    [0.90, 0.85, 0.80, 300, 600],  # Lower thresholds
    [0.98, 0.95, 0.90, 300, 600],  # Higher thresholds
    [0.95, 0.90, 0.85, 150, 300],  # Lower DPI
    [0.95, 0.90, 0.85, 400, 800],  # Higher DPI
]

ORIGINAL_CONFIG = Path("config.py").read_text()


def update_config(tau_accept, tau_enhance, tau_llm, dpi_primary, dpi_enhanced):
    """Update config.py with new values using regex replacements."""
    updated = ORIGINAL_CONFIG
    updated = re.sub(
        r"TAU_FIELD_ACCEPT\s*=.*",
        f"TAU_FIELD_ACCEPT  = {tau_accept}",
        updated,
    )
    updated = re.sub(
        r"TAU_ENHANCER_PASS\s*=.*",
        f"TAU_ENHANCER_PASS = {tau_enhance}",
        updated,
    )
    updated = re.sub(
        r"TAU_LLM_PASS\s*=.*",
        f"TAU_LLM_PASS      = {tau_llm}",
        updated,
    )
    updated = re.sub(
        r"DPI_PRIMARY\s*=.*",
        f"DPI_PRIMARY       = {dpi_primary}",
        updated,
    )
    updated = re.sub(
        r"DPI_ENHANCED\s*=.*",
        f"DPI_ENHANCED      = {dpi_enhanced}",
        updated,
    )
    Path("config.py").write_text(updated)

@pytest.mark.parametrize("config_params", configs)
def test_config(config_params):
    """Compile pipeline with different configuration values."""
    tau_accept, tau_enhance, tau_llm, dpi_primary, dpi_enhanced = config_params

    update_config(tau_accept, tau_enhance, tau_llm, dpi_primary, dpi_enhanced)

    try:
        cfg = Path("config.py").read_text()
        assert f"TAU_FIELD_ACCEPT  = {tau_accept}" in cfg
        assert f"TAU_ENHANCER_PASS = {tau_enhance}" in cfg
        assert f"TAU_LLM_PASS      = {tau_llm}" in cfg
        assert f"DPI_PRIMARY       = {dpi_primary}" in cfg
        assert f"DPI_ENHANCED      = {dpi_enhanced}" in cfg

        result = subprocess.run(
            [sys.executable, "-m", "py_compile", "pipeline.py"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr

    finally:
        Path("config.py").write_text(ORIGINAL_CONFIG)

# The original script provided a CLI for experimentation. The pytest version
# simply parametrises the configurations and asserts that the pipeline runs
# successfully for each of them. Any additional analysis or printing of the best
# configuration is outside the scope of automated tests.
