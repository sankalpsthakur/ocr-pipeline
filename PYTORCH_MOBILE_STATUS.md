# PyTorch Mobile OCR - Actual Performance vs Claims

## Executive Summary
The PyTorch Mobile OCR implementation has significant discrepancies between documented claims and actual performance. The system currently relies on Tesseract OCR as a fallback rather than the PyTorch models, which are randomly initialized and untrained.

## Performance Comparison

### 1. Model Sizes
| Component | Claimed | Actual (Uncompressed) | Actual (Quantized Est.) | Status |
|-----------|---------|----------------------|------------------------|---------|
| Detection Model | 15.8 MB | 16.58 MB | 4.15 MB | ⚠️ Architecture issue |
| Recognition Model | 25.3 MB | 32.56 MB | 8.14 MB | ⚠️ Architecture issue |
| Angle Classifier | 4.9 MB | 0.91 MB | 0.23 MB | ✅ Smaller than claimed |
| **Total Package** | **46.0 MB** | **50.05 MB** | **12.51 MB** | **❌ Export fails** |

**Finding**: 
- Uncompressed models are 50.05 MB (close to claims)
- Quantized estimate is 12.51 MB (much smaller than claimed 46 MB)
- Export fails due to CRNN architecture issues in recognition model
- Models appear randomly initialized (std ~0.11)

### 2. Inference Speed
| Device | Claimed | Actual | Variance |
|--------|---------|---------|----------|
| iPhone 12 | 110ms | No data | ❌ Untested |
| Desktop (for reference) | N/A | 1,714ms avg | - |
| Processing Range | 110-250ms | 700-2,345ms | 7-20x slower |

**Finding**: Desktop performance is significantly slower than claimed mobile speeds, suggesting mobile claims are unrealistic.

### 3. Field Extraction Accuracy
| Metric | Claimed | Actual | Evidence |
|--------|---------|---------|----------|
| Overall Field Accuracy | >90% | 4.5% perfect | Stress test results |
| Electricity (kWh) | 100% | Variable | Only on high-quality images |
| Carbon (kg CO2e) | 100% | Variable | Often missed in compressed images |
| Average Confidence | >94% | 48.5% | Across 22 test variations |

**Finding**: Accuracy claims are vastly overstated. Only original high-quality images achieve claimed accuracy.

### 4. Stress Test Results (22 tests)
- **Perfect extractions**: 1/22 (4.5%)
- **Partial extractions**: 18/22 (81.8%)
- **Failed extractions**: 3/22 (13.6%)
- **Confidence correlation**: r=0.59 (moderate positive)

## Technical Issues Identified

### 1. Model Initialization
```python
# Models are randomly initialized, not pre-trained
self.detector = TextDetector()  # No weights loaded
self.recognizer = TextRecognizer()  # No weights loaded
```

### 2. Tesseract Fallback
The system primarily uses Tesseract OCR, not PyTorch models:
```python
def run_ocr_with_tesseract(image_path: Union[str, Path]) -> Dict[str, any]:
    """Fallback OCR using Tesseract for better accuracy."""
    # This is the primary OCR engine being used
```

### 3. Architecture Issues
- FPN channel mismatch: Expected [24,40,80,160], got [16,40,80,160]
- No pre-trained weights available
- Models produce random outputs without training

### 4. Missing Components
- No trained model weights
- No exported mobile models (.pt or .onnx files)
- No actual mobile app implementations
- No real device benchmarks

## Actual Capabilities

### What Works:
1. **Tesseract Integration**: Provides reliable OCR with 91.7% confidence on high-quality images
2. **Field Extraction**: Regex patterns correctly extract utility bill fields when text is detected
3. **Confidence Calibration**: Statistical correlation between confidence and accuracy (r=0.59)
4. **Image Preprocessing**: Handles multiple formats (PNG, JPEG, WebP)

### What Doesn't Work:
1. **PyTorch Models**: Untrained, produce random outputs
2. **Mobile Deployment**: No evidence of actual mobile implementation
3. **Performance Claims**: Desktop is 7-20x slower than claimed mobile speeds
4. **Accuracy Claims**: Only 4.5% perfect extraction rate across image variations

## Recommendations

### Immediate Actions:
1. **Update Documentation**: Remove unverified claims from README.md
2. **Add Disclaimers**: Clearly state current reliance on Tesseract
3. **Train Models**: Implement proper training pipeline for PyTorch models

### Medium-term Actions:
1. **Export Models**: Generate actual mobile models and verify sizes
2. **Mobile Testing**: Create test apps and benchmark on real devices
3. **Improve Architecture**: Fix channel mismatches and optimize for mobile

### Long-term Actions:
1. **Dataset Creation**: Build utility bill training dataset
2. **Model Training**: Train models from scratch or fine-tune existing OCR models
3. **Performance Optimization**: Implement quantization and pruning for mobile

## Current Production Readiness

**For Production Use**: ✅ Ready (using Tesseract fallback)
- Reliable extraction on high-quality images
- Good confidence calibration
- Comprehensive error handling

**For Mobile Deployment**: ❌ Not Ready
- No trained PyTorch models
- No exported mobile models
- Unverified performance claims

## Conclusion

The PyTorch Mobile OCR implementation is currently a wrapper around Tesseract OCR with aspirational documentation about mobile deployment. While the architecture and code structure exist for a PyTorch-based solution, the models are untrained and the mobile claims are unsubstantiated. The system works well in production using Tesseract but does not deliver on its PyTorch Mobile promises.