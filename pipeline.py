
"""End‑to‑end pipeline entry point."""
import json, sys, logging
from pathlib import Path
from typing import Any, Dict

from . import ocr, extractor, config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

def build_payload(fields: Dict[str, Any], source_path: Path) -> Dict[str, Any]:
    sha256 = _sha256(source_path)
    return {
        "electricity": {
            "consumption": {
                "value": fields.get("electricity_kwh"),
                "unit": "kWh"
            }
        } if "electricity_kwh" in fields else {},
        "carbon": {
            "location_based": {
                "value": fields.get("carbon_kgco2e"),
                "unit": "kgCO2e"
            }
        } if "carbon_kgco2e" in fields else {},
        "source_document": {
            "file_name": source_path.name,
            "sha256": sha256
        },
        "meta": {
            "extraction_confidence": fields.get("_confidence")
        }
    }

def _sha256(fp: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with fp.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m robust_ocr_pipeline.pipeline path/to/bill.pdf [--save outfile.json]")
        sys.exit(1)

    bill_path = Path(sys.argv[1])
    save_path = None
    if "--save" in sys.argv:
        save_path = Path(sys.argv[sys.argv.index("--save") + 1])

    # OCR
    ocr_res = ocr.run_ocr(bill_path)

    # Extract
    fields = extractor.extract_fields(ocr_res.text)
    fields["_confidence"] = ocr_res.field_confidence

    # Build payload
    payload = build_payload(fields, bill_path)

    # Persist if requested
    if save_path:
        save_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {save_path}")

    # Print
    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
