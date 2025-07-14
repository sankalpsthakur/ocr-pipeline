# OCR Pipeline - Claude Development Guide

## Project Overview
Production-ready PyTorch Mobile OCR pipeline for utility bill extraction with comprehensive JSON schema compliance and statistical confidence calibration.

## Key Commands
```bash
# Activate environment
source venv/bin/activate

# Run OCR with utility bill schema
python pytorch_mobile/ocr_pipeline.py [IMAGE] --format utility_bill --save [OUTPUT.json]

# Run stress testing
python stress_test.py

# Analyze confidence correlation
python confidence_analysis.py

# Run basic field extraction
python pytorch_mobile/ocr_pipeline.py [IMAGE]
```

## Core Architecture
- **Text Detection**: DBNet with MobileNetV3 backbone
- **Text Recognition**: CRNN with attention mechanism
- **Fallback Engine**: Tesseract OCR for enhanced accuracy
- **Confidence Calibration**: Statistical formula (40% OCR + 35% accuracy + 25% completeness)

## Validated Performance
- **Ground Truth**: DEWA (299 kWh, 120 kg CO2e), SEWA (358 kWh, 121.3 m³)
- **Stress Testing**: 22/22 tests passed across compression/noise/scaling
- **Confidence Correlation**: r=0.590, p=0.004 (statistically significant)
- **Processing Speed**: 0.7-2.3s per document

## Key Files
- `pytorch_mobile/ocr_pipeline.py` - Main production pipeline
- `stress_test.py` - Comprehensive testing framework
- `confidence_analysis.py` - Statistical validation tool
- `DEWA_Utility_Bill_Extracted.json` - DEWA validation output
- `SEWA_Utility_Bill_Extracted.json` - SEWA validation output

## Development Notes
- Fixed FPN channel dimensions: [24,40,80,160] → [16,40,80,160]
- Fixed numpy deprecation: np.int0 → np.int32
- Tesseract provides higher accuracy than PyTorch models on utility bills
- Confidence scores correlate significantly with extraction accuracy
- System handles various image formats (PNG, JPEG, WebP) and quality levels

## Testing
Run full test suite before commits:
```bash
python stress_test.py
python confidence_analysis.py
```

## Schema Compliance
Follows comprehensive utility bill JSON schema with:
- Document metadata and validation
- Consumption data (electricity, water, gas)
- Emissions data (Scope 2 CO2e)
- Field-level accuracy tracking
- 6-decimal precision for numeric values