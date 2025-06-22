
import re, math, json, logging
from typing import Dict

LOGGER = logging.getLogger(__name__)

# allow commas in numbers (e.g. "1,234 kWh")
ENERGY_RE  = re.compile(r"(\d[\d\s,]{0,10})\s*k\s*W\s*h", re.I)
# handle both "CO2e" and "CO₂e" notations
CARBON_RE  = re.compile(r"Kg\s*CO(?:2|₂)e\s*(\d[\d\s,]{0,10})", re.I)

def _normalise_number(num_txt: str) -> int:
    """Strip common separators before casting to int."""
    digits = re.sub(r"[^0-9]", "", num_txt)
    return int(digits)

def extract_fields(text: str) -> Dict[str, int]:
    out = {}
    m = ENERGY_RE.search(text)
    if m:
        out["electricity_kwh"] = _normalise_number(m.group(1))

    m = CARBON_RE.search(text)
    if m:
        out["carbon_kgco2e"] = _normalise_number(m.group(1))
    return out
