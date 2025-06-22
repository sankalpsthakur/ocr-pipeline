
import re, math, json, logging
from typing import Dict

LOGGER = logging.getLogger(__name__)

ENERGY_RE  = re.compile(r"(\d[\d\s]{0,10})\s*k\s*W\s*h", re.I)
CARBON_RE  = re.compile(r"Kg\s*CO2e\s*(\d[\d\s]{0,10})", re.I)

def _normalise_number(num_txt: str) -> int:
    digits = re.sub(r"\s+", "", num_txt)
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
