"""
Minimal pipeline stub for CI testing.
Provides basic functionality without heavy ML dependencies.
"""

import re
from typing import Dict


class OcrResult:
    """Basic OCR result class for CI testing."""
    
    def __init__(self, text: str, tokens: list, confidences: list):
        self.text = text
        self.tokens = tokens
        self.confidences = confidences
    
    @property
    def field_confidence(self) -> float:
        """Geometric mean of token confidences."""
        if not self.confidences:
            return 0.0
        
        # Simple geometric mean calculation
        product = 1.0
        for conf in self.confidences:
            product *= max(conf, 0.001)  # Avoid zero
        
        return product ** (1.0 / len(self.confidences))


def extract_fields(text: str) -> Dict[str, int]:
    """Basic field extraction for testing."""
    fields = {}
    
    # Extract electricity kWh
    kwh_match = re.search(r'(\d+)\s*kWh', text, re.IGNORECASE)
    if kwh_match:
        fields['electricity_kwh'] = int(kwh_match.group(1))
    
    # Extract carbon footprint
    carbon_match = re.search(r'(\d+)\s*(?:kg|Kg)\s*CO2e?', text, re.IGNORECASE)
    if carbon_match:
        fields['carbon_kgco2e'] = int(carbon_match.group(1))
    
    return fields


def run_ocr(file_path) -> OcrResult:
    """Stub OCR function for CI testing."""
    return OcrResult("Stub OCR result", ["stub", "result"], [0.9, 0.8])