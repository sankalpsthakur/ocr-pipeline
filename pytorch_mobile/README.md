# PyTorch Mobile OCR Pipeline

⚠️ **IMPORTANT DISCLAIMER**: This implementation currently uses Tesseract OCR as the primary engine. The PyTorch models are experimental and require training before use. See [PYTORCH_MOBILE_STATUS.md](../PYTORCH_MOBILE_STATUS.md) for detailed performance analysis.

## 🚀 Quick Start

### Installation

```bash
# Install PyTorch (CPU version for mobile development)
pip install torch torchvision

# Install dependencies
pip install opencv-python pillow numpy pytesseract

# Install Tesseract OCR (required)
# macOS: brew install tesseract
# Ubuntu: sudo apt-get install tesseract-ocr
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki

# Optional: Install ONNX for cross-platform deployment
pip install onnx onnxruntime
```

### Basic Usage

```python
from ocr_pipeline import run_ocr_with_fields

# Process a utility bill (uses Tesseract by default)
result = run_ocr_with_fields("ActualBill.png")

print(f"Electricity: {result.get('electricity_kwh')} kWh")
print(f"Carbon: {result.get('carbon_kgco2e')} kg CO2e")
print(f"Confidence: {result.get('_ocr_confidence'):.3f}")
```

### Command Line

```bash
# Run OCR on an image with utility bill schema
python ocr_pipeline.py image.png --format utility_bill --save output.json

# Run tests
python test_pipeline.py
```

## 📊 Actual Performance Metrics

Based on comprehensive stress testing across 22 image variations:

### Accuracy (Ground Truth Validated)
| Image Quality | Confidence | Field Accuracy | Notes |
|--------------|------------|----------------|-------|
| Original PNG | 91.7% | 100% | Perfect extraction |
| JPEG Q70 | 70.9% | 66.7% | Some fields missed |
| JPEG Q30 | 91.2% | 50% | Compression artifacts |
| 50% Scale | 43.1% | 33.3% | Resolution too low |
| With Noise | 0-23.5% | 0-31.2% | Depends on noise level |

### Processing Speed
- **Average**: 1.71 seconds per image
- **Range**: 0.7 - 2.3 seconds
- **Platform**: Desktop CPU (mobile performance not yet tested)

### Confidence Calibration
- **Correlation with accuracy**: r=0.59 (p=0.004)
- **High confidence (>80%)**: 62.5% average accuracy
- **Low confidence (<40%)**: 31.2% average accuracy

## 🔧 Current Implementation Status

### ✅ Working Features
- Tesseract OCR integration with high accuracy on quality images
- Comprehensive field extraction for utility bills
- Statistical confidence calibration
- Multi-format support (PNG, JPEG, WebP)
- JSON schema compliance

### ⚠️ Experimental Features
- PyTorch text detection (DBNet with MobileNetV3)
- PyTorch text recognition (CRNN)
- Angle classification
- Mobile model export

### ❌ Not Yet Implemented
- Pre-trained PyTorch model weights
- Actual mobile deployment testing
- GPU acceleration on mobile devices
- Model quantization verification

## 📱 Mobile Deployment Guide (Experimental)

**Note**: The mobile deployment features are experimental. Models must be trained before use.

### Prerequisites
1. Train the PyTorch models (weights not included)
2. Export models using the export function
3. Verify model sizes and performance

### Step 1: Export Models (After Training)

```python
from ocr_pipeline import export_models_for_mobile

# This will fail without trained models
export_models_for_mobile("mobile_models")
```

### Step 2: iOS Integration (Theoretical)

The README previously contained iOS integration examples, but these are untested without trained models.

### Step 3: Android Integration (Theoretical)

Android integration code is provided but requires trained models to function.

## 🧪 Testing

Run the test suite to verify your setup:

```bash
python test_pipeline.py
```

**Note**: Most PyTorch model tests will fail without proper training. The pipeline falls back to Tesseract for reliable OCR.

## 🛠️ Troubleshooting

### Common Issues

**"No module named 'pytesseract'"**
- Install pytesseract: `pip install pytesseract`
- Install Tesseract binary for your OS

**Low accuracy on compressed images**
- This is expected behavior
- Use high-quality source images when possible
- Check confidence scores to assess reliability

**PyTorch models producing random outputs**
- Models are not trained
- System falls back to Tesseract automatically

## 📚 Architecture

### Current Production Pipeline
1. Image preprocessing and format conversion
2. Tesseract OCR for text extraction
3. Regex-based field extraction
4. Confidence calibration based on field completeness
5. JSON schema formatting

### Experimental PyTorch Pipeline
1. DBNet text detection (MobileNetV3 backbone)
2. CRNN text recognition (with attention)
3. Angle classification for rotated text
4. Character-level correction

## 🔍 Limitations

1. **No Pre-trained Weights**: PyTorch models require training
2. **Desktop Performance**: 1.7s average (mobile untested)
3. **Image Quality Sensitive**: Performance degrades with compression/noise
4. **Primary Engine**: Relies on Tesseract, not PyTorch

## 📄 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

To contribute:
1. Focus on training the PyTorch models
2. Test on actual mobile devices
3. Improve accuracy on compressed images
4. Add pre-trained model weights

## 📞 Support

For production use:
- Use the current Tesseract-based implementation
- Monitor confidence scores
- Use high-quality input images

For experimental PyTorch features:
- See [PYTORCH_MOBILE_STATUS.md](../PYTORCH_MOBILE_STATUS.md)
- Contribute training code or datasets
- Help with mobile testing