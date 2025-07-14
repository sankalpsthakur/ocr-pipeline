# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a **dual-pipeline OCR system** for utility bill field extraction:

1. **Legacy Pipeline** (`pipeline.py`): PaddleOCR-based system with VLM fallbacks
2. **Advanced Pipeline** (`adaptive_ocr_pipeline.py`): JAX denoising + QAT models with adaptive quality routing

The system addresses the core problem: **"Near 0% confidence on degraded images"** through intelligent preprocessing and model training.

### Key Components Integration

**Adaptive Quality Assessment Flow:**
```
Input Image → Quality Assessment (6 metrics) → Strategy Selection → Preprocessing → OCR Engine → Field Extraction
    ↓              ↓                              ↓               ↓            ↓
Quality Tier   Multi-metric Analysis      JAX Denoising     QAT Models   Pattern Matching
(high/med/low)  (sharpness, contrast...)   U-Net (<1M)      (INT8)       (Regex + Validation)
```

**Training Infrastructure:**
- `jax_denoising_adapter.py`: Lightweight U-Net denoising (310 LOC)
- `qat_robust_models.py`: Mobile-optimized quantized models (357 LOC)  
- `synthetic_degradation.py`: 12-type degradation engine (368 LOC)
- `train_jax_denoising.py`: JAX training pipeline (387 LOC)
- `train_qat_robust.py`: QAT training infrastructure (254 LOC)

## Essential Commands

### Basic OCR Processing
```bash
# Legacy pipeline (PP-OCRv5)
python pipeline.py ActualBill.png

# PyTorch mobile pipeline  
python pytorch_mobile/ocr_pipeline.py DEWA.png --format utility_bill

# Advanced adaptive pipeline
python adaptive_ocr_pipeline.py DEWA.png
```

### Training & Model Development
```bash
# Generate synthetic training data (12 degradation types)
python synthetic_degradation.py --input DEWA.png --num-samples 10

# Train JAX denoising models (Noise2Noise approach)
python train_jax_denoising.py

# Train QAT models for mobile deployment
python train_qat_robust.py

# Create/update pre-trained weights
python create_pretrained_weights.py
```

### Testing & Validation
```bash
# Legacy system testing
python run_comprehensive_tests.py
python stress_test.py

# Advanced system validation
python test_trained_system.py
python robustness_evaluation.py

# Confidence correlation analysis
python confidence_analysis.py
```

### Performance Analysis
```bash
# System summary and metrics
python training_summary.py

# Benchmark noise robustness
python robustness_evaluation.py --test-degradation
```

## Configuration Management

**Core Config** (`config.py`):
- Confidence thresholds: `TAU_FIELD_ACCEPT=0.95`, `TAU_ENHANCER_PASS=0.90`, `TAU_LLM_PASS=0.85`
- DPI settings: Primary=300, Enhanced=600
- Performance: `MAX_WORKER_THREADS=3`, GPU detection

**Adaptive Config** (`adaptive_ocr_pipeline.py`):
```python
AdaptiveConfig(
    use_jax_denoising=True,
    use_qat_models=True, 
    confidence_boost_factor=1.2,
    max_image_size=2048
)
```

## Model Architecture Details

### JAX Denoising Models
- **U-Net**: Encoder [16,32,64,128], Decoder [128,64,32,16]
- **Quality Classifier**: 3-class (low/medium/high) 
- **Training**: Noise2Noise on synthetic degraded data
- **Checkpoints**: `jax_checkpoints/best_checkpoint.pkl` (15MB)

### QAT Models  
- **Architecture**: MobileNetV3 + quantization-aware layers
- **Optimization**: INT8 quantization, 4x size reduction
- **Training**: Built-in noise augmentation layers
- **Export**: TorchScript for mobile deployment

### Synthetic Data Generation
- **Degradation Types**: 12 types (noise, blur, compression, shadows, geometric)
- **Severity Levels**: Low/Medium/High with parameter ranges
- **Output**: Clean/degraded pairs with comprehensive metadata

## Critical Integration Points

### Pipeline Routing Logic
```python
# Quality assessment determines preprocessing strategy
if quality_tier == 'high':
    # Direct processing with original OCR
elif quality_tier == 'medium':  
    # Light bilateral filtering
else:  # low quality
    # JAX denoising + QAT models
```

### Confidence Calibration
```python
# Multi-source confidence fusion
final_confidence = 0.7 × ocr_confidence + 
                  0.2 × preprocessing_boost + 
                  0.1 × pattern_match_boost
```

### Field Extraction Pattern Hierarchy
1. **Primary patterns**: Context-aware regex (high precision)
2. **Fallback patterns**: Simple numeric extraction (high recall)  
3. **Range validation**: Utility bill value bounds (50-9999 kWh)
4. **Cross-field consistency**: Logical relationship checks

## Ground Truth & Validation

**Test Images:**
- DEWA.png: 299 kWh, 120 kg CO2e
- SEWA.png: 358 kWh, 121.3 m³ 

**Performance Targets:**
- Legacy system: 95%+ field accuracy
- Advanced system: 40-60% improvement on degraded images
- Mobile deployment: <200MB footprint, <200ms inference

**Quality Metrics:**
- Confidence-accuracy correlation: r>0.9
- Stress testing: 22/22 tests passing  
- Cross-scale robustness: 100%/50%/25% image scales

## Development Workflow

### Adding New Degradation Types
1. Extend `synthetic_degradation.py` with new degradation function
2. Update severity level parameters 
3. Regenerate training data: `python synthetic_degradation.py`
4. Retrain models: `python train_jax_denoising.py`

### Improving Quality Assessment  
1. Modify metrics in `ImageQualityAssessor.assess_quality()`
2. Update routing thresholds in `_select_processing_strategy()`
3. Validate with: `python test_trained_system.py`

### Mobile Deployment
1. Train QAT models: `python train_qat_robust.py` 
2. Export quantized models: `pipeline.export_for_mobile()`
3. Validate INT8 performance: `pipeline.benchmark_quantization()`

## Pre-trained Assets

**JAX Models** (`jax_checkpoints/`):
- 50 epochs training on synthetic data
- ~850K parameters total
- Best validation loss: 0.0209

**Training Data** (`synthetic_training_data/`):  
- 10 degraded training pairs
- 12 degradation types implemented
- Comprehensive metadata tracking

**Performance Expectations:**
- 40-60% accuracy improvement on degraded images
- <200ms processing time (640x480)
- 90%+ mobile deployment compatibility