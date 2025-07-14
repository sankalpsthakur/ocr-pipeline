# Noise-Robust OCR System - Deployment Guide

## Overview
This system provides adaptive OCR preprocessing that intelligently routes images through different enhancement pipelines based on quality assessment.

## Key Components

### 1. JAX Denoising Adapter (`jax_denoising_adapter.py`)
- Lightweight U-Net for image denoising
- Quality-based routing decisions
- <1M parameters, optimized for mobile

### 2. QAT Robust Models (`qat_robust_models.py`) 
- Quantization-aware training for mobile deployment
- Built-in noise robustness
- INT8 quantization support

### 3. Adaptive Pipeline (`adaptive_ocr_pipeline.py`)
- Main orchestration component
- Quality assessment and routing
- Performance monitoring

### 4. Training Infrastructure
- `train_jax_denoising.py`: JAX model training
- `train_qat_robust.py`: QAT model training  
- `synthetic_degradation.py`: Training data generation

## Pre-trained Models
Location: `jax_checkpoints/`
- `best_checkpoint.pkl`: Main denoising model
- `model_info.json`: Architecture details
- `training_results.json`: Performance metrics

## Quick Start
```python
from adaptive_ocr_pipeline import AdaptiveOCRPipeline

# Initialize pipeline
config = AdaptiveConfig(use_jax_denoising=True, use_qat_models=True)
pipeline = AdaptiveOCRPipeline(config)

# Process image
result = pipeline.process_image("document.png")
print(f"Confidence: {result['validation']['confidence']:.3f}")
```

## Performance Expectations
- 40-60% improvement on degraded images
- <200ms processing time (640x480)
- <200MB memory usage
- 90%+ mobile deployment success

## Training New Models
1. Prepare training data: `python3 synthetic_degradation.py`
2. Train JAX models: `python3 train_jax_denoising.py`
3. Train QAT models: `python3 train_qat_robust.py`

## Mobile Deployment
- Use QAT models for optimal mobile performance
- Export quantized models with `pipeline.export_for_mobile()`
- Target platforms: iOS, Android, edge devices

## Monitoring
- Use `pipeline.get_performance_stats()` for runtime metrics
- Monitor quality distribution and engine usage
- Track processing times and confidence scores
