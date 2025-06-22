#!/usr/bin/env python3
"""Test different hyperparameter configurations to find optimal settings."""

import json
import subprocess
import sys
from pathlib import Path

# Test configurations
configs = [
    # [TAU_FIELD_ACCEPT, TAU_ENHANCER_PASS, TAU_LLM_PASS, DPI_PRIMARY, DPI_ENHANCED]
    [0.95, 0.90, 0.85, 300, 600],  # Original
    [0.90, 0.85, 0.80, 300, 600],  # Lower thresholds
    [0.98, 0.95, 0.90, 300, 600],  # Higher thresholds
    [0.95, 0.90, 0.85, 150, 300],  # Lower DPI
    [0.95, 0.90, 0.85, 400, 800],  # Higher DPI
]

def update_config(tau_accept, tau_enhance, tau_llm, dpi_primary, dpi_enhanced):
    """Update config.py with new values."""
    config_content = f'''
"""Central configuration – hard‑coded secrets & thresholds"""


# --- OCR back‑end selection --------------------------------------------------
# Options: "tesseract", "gcv", "azure"
OCR_BACKEND       = "tesseract"

# --- Confidence thresholds ---------------------------------------------------
TAU_FIELD_ACCEPT  = {tau_accept}  # auto‑accept threshold
TAU_ENHANCER_PASS = {tau_enhance}  # after enhancer / alt engine
TAU_LLM_PASS      = {tau_llm}  # LLM fallback

# --- Misc --------------------------------------------------------------------
MAX_PAGES         = 20    # safety cap to avoid 100‑page uploads
DPI_PRIMARY       = {dpi_primary}
DPI_ENHANCED      = {dpi_enhanced}

# --- Tesseract options -------------------------------------------------------
# Language, OCR engine mode and page segmentation mode can be tuned depending
# on the type of documents processed. They are exposed here so that pipeline
# users can adjust them without touching the codebase.
TESSERACT_LANG    = "eng"
TESSERACT_OEM     = 3     # default LSTM engine
TESSERACT_PSM     = 6     # assume a single uniform block of text



'''
    Path("config.py").write_text(config_content)

def test_config(config_params):
    """Test a configuration and return results."""
    tau_accept, tau_enhance, tau_llm, dpi_primary, dpi_enhanced = config_params
    
    print(f"Testing: TAU_ACCEPT={tau_accept}, TAU_ENHANCE={tau_enhance}, TAU_LLM={tau_llm}, DPI={dpi_primary}/{dpi_enhanced}")
    
    update_config(tau_accept, tau_enhance, tau_llm, dpi_primary, dpi_enhanced)
    
    try:
        # Run pipeline and capture output
        result = subprocess.run([
            sys.executable, "pipeline.py", "ActualbillDownload_250618_115704.pdf"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            # Parse JSON output
            try:
                data = json.loads(result.stdout)
                confidence = data.get("meta", {}).get("extraction_confidence", 0)
                electricity = data.get("electricity", {}).get("consumption", {}).get("value")
                carbon = data.get("carbon", {}).get("location_based", {}).get("value")
                
                # Count log messages to understand processing path
                stderr_lines = result.stderr.split('\n')
                primary_pass = any("Primary pass accepted" in line for line in stderr_lines)
                enhancement = any("Enhancement triggered" in line for line in stderr_lines)
                llm_warning = any("below LLM threshold" in line for line in stderr_lines)
                
                return {
                    "config": config_params,
                    "confidence": confidence,
                    "electricity": electricity,
                    "carbon": carbon,
                    "primary_pass": primary_pass,
                    "enhancement": enhancement,
                    "llm_warning": llm_warning,
                    "success": True
                }
            except json.JSONDecodeError:
                return {"config": config_params, "success": False, "error": "JSON decode error"}
        else:
            return {"config": config_params, "success": False, "error": result.stderr}
    
    except subprocess.TimeoutExpired:
        return {"config": config_params, "success": False, "error": "Timeout"}
    except Exception as e:
        return {"config": config_params, "success": False, "error": str(e)}

def main():
    """Test all configurations and report results."""
    print("Testing hyperparameter configurations...\n")
    
    results = []
    for config in configs:
        result = test_config(config)
        results.append(result)
        
        if result["success"]:
            print(f"✓ Confidence: {result['confidence']:.3f}, "
                  f"Fields: {result['electricity']}/{result['carbon']}, "
                  f"Primary: {result['primary_pass']}, "
                  f"Enhanced: {result['enhancement']}, "
                  f"LLM Warning: {result['llm_warning']}")
        else:
            print(f"✗ Failed: {result['error']}")
        print()
    
    # Find best configuration
    successful_results = [r for r in results if r["success"]]
    if successful_results:
        # Best = highest confidence with correct field extraction
        best = max(successful_results, key=lambda x: (
            x['electricity'] == 299 and x['carbon'] == 120,  # Correct extraction
            x['confidence']  # Highest confidence
        ))
        
        print("=" * 60)
        print("OPTIMAL CONFIGURATION:")
        print(f"TAU_FIELD_ACCEPT = {best['config'][0]}")
        print(f"TAU_ENHANCER_PASS = {best['config'][1]}")
        print(f"TAU_LLM_PASS = {best['config'][2]}")
        print(f"DPI_PRIMARY = {best['config'][3]}")
        print(f"DPI_ENHANCED = {best['config'][4]}")
        print(f"Confidence: {best['confidence']:.3f}")
        print(f"Extracted: {best['electricity']} kWh, {best['carbon']} kgCO2e")
        
        # Restore optimal config
        update_config(*best['config'])
        print("\nOptimal configuration has been applied to config.py")

if __name__ == "__main__":
    main()