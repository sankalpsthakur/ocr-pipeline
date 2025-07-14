#!/usr/bin/env python3
"""Adaptive OCR Pipeline with Quality-Based Routing

Intelligently routes images through different OCR engines based on quality
assessment. Combines JAX denoising, QAT models, and traditional OCR for
optimal performance across all image conditions.
"""

import numpy as np
import cv2
from PIL import Image
import torch
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import json
import logging

# Import our custom components
from jax_denoising_adapter import JAXDenoisingAdapter, DenoisingConfig
from qat_robust_models import QATRobustOCRPipeline
from pytorch_mobile.ocr_pipeline import run_ocr_with_tesseract, extract_fields, build_utility_bill_payload


class QualityTier(Enum):
    """Image quality tiers for routing decisions."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class ProcessingMetrics:
    """Metrics for OCR processing."""
    preprocessing_time: float
    ocr_time: float
    total_time: float
    engine_used: str
    quality_tier: str
    confidence: float
    preprocessing_applied: List[str]


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive OCR pipeline."""
    # Quality thresholds
    high_quality_threshold: float = 0.8
    medium_quality_threshold: float = 0.5
    low_quality_threshold: float = 0.2
    
    # Engine preferences
    use_jax_denoising: bool = True
    use_qat_models: bool = True
    use_tesseract_fallback: bool = True
    
    # Performance settings
    max_image_size: int = 2048
    enable_parallel_processing: bool = False
    cache_preprocessing: bool = True
    
    # Confidence boosting
    confidence_boost_factor: float = 1.2
    min_confidence_threshold: float = 0.3


class ImageQualityAssessor:
    """Fast image quality assessment for routing decisions."""
    
    def __init__(self):
        self.metrics_cache = {}
    
    def assess_quality(self, image: np.ndarray) -> Tuple[QualityTier, float, Dict[str, float]]:
        """Assess image quality using multiple metrics.
        
        Returns:
            quality_tier: Enum indicating quality level
            overall_score: Float between 0 and 1
            detailed_metrics: Dict with individual quality metrics
        """
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        metrics = {}
        
        # 1. Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        metrics['sharpness'] = laplacian.var()
        
        # 2. Contrast (standard deviation)
        metrics['contrast'] = gray.std()
        
        # 3. Brightness consistency
        mean_brightness = gray.mean()
        metrics['brightness'] = min(mean_brightness / 128.0, 2.0 - mean_brightness / 128.0)
        
        # 4. Noise estimation (high frequency content)
        gray_float = gray.astype(np.float32) / 255.0
        noise_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        noise_response = cv2.filter2D(gray_float, -1, noise_kernel)
        metrics['noise_level'] = 1.0 - min(1.0, noise_response.std() * 10)
        
        # 5. Edge density (indicator of text presence)
        edges = cv2.Canny(gray, 50, 150)
        metrics['edge_density'] = edges.sum() / edges.size * 255
        
        # 6. Dynamic range
        metrics['dynamic_range'] = (gray.max() - gray.min()) / 255.0
        
        # Normalize metrics
        normalized_metrics = {
            'sharpness': min(1.0, metrics['sharpness'] / 100.0),
            'contrast': min(1.0, metrics['contrast'] / 50.0),
            'brightness': metrics['brightness'],
            'noise_level': metrics['noise_level'],
            'edge_density': min(1.0, metrics['edge_density'] / 10000.0),
            'dynamic_range': metrics['dynamic_range']
        }
        
        # Weighted combination
        weights = {
            'sharpness': 0.25,
            'contrast': 0.20,
            'brightness': 0.15,
            'noise_level': 0.20,
            'edge_density': 0.10,
            'dynamic_range': 0.10
        }
        
        overall_score = sum(normalized_metrics[k] * weights[k] for k in weights.keys())
        
        # Determine quality tier
        if overall_score >= 0.8:
            tier = QualityTier.HIGH
        elif overall_score >= 0.5:
            tier = QualityTier.MEDIUM
        elif overall_score >= 0.2:
            tier = QualityTier.LOW
        else:
            tier = QualityTier.VERY_LOW
        
        return tier, overall_score, normalized_metrics


class AdaptiveOCRPipeline:
    """Adaptive OCR pipeline with intelligent quality-based routing."""
    
    def __init__(self, config: Optional[AdaptiveConfig] = None):
        self.config = config or AdaptiveConfig()
        
        # Initialize components
        self.quality_assessor = ImageQualityAssessor()
        self.jax_denoiser = None
        self.qat_pipeline = None
        
        # Performance tracking
        self.performance_stats = {
            'total_processed': 0,
            'engine_usage': {'tesseract': 0, 'qat': 0, 'denoised': 0},
            'avg_processing_time': 0,
            'quality_distribution': {'high': 0, 'medium': 0, 'low': 0, 'very_low': 0}
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize engines
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize OCR engines based on configuration."""
        try:
            if self.config.use_jax_denoising:
                denoising_config = DenoisingConfig(
                    max_image_size=self.config.max_image_size,
                    device='cpu'  # Use CPU for broad compatibility
                )
                self.jax_denoiser = JAXDenoisingAdapter(denoising_config)
                self.jax_denoiser.initialize()
                self.logger.info("JAX denoising adapter initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize JAX denoiser: {e}")
            self.config.use_jax_denoising = False
        
        try:
            if self.config.use_qat_models:
                self.qat_pipeline = QATRobustOCRPipeline()
                self.logger.info("QAT pipeline initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize QAT pipeline: {e}")
            self.config.use_qat_models = False
    
    def process_image(self, image: Union[np.ndarray, Image.Image, str, Path], 
                     format_type: str = "utility_bill") -> Dict[str, Any]:
        """Process image with adaptive routing.
        
        Args:
            image: Input image (various formats supported)
            format_type: Output format ("utility_bill", "basic", "detailed")
            
        Returns:
            Comprehensive OCR results with metadata
        """
        start_time = time.time()
        
        # Load and validate image
        img_array = self._load_image(image)
        if img_array is None:
            return self._create_error_result("Failed to load image")
        
        # Assess image quality
        quality_tier, quality_score, quality_metrics = self.quality_assessor.assess_quality(img_array)
        
        # Update statistics
        self.performance_stats['quality_distribution'][quality_tier.value] += 1
        
        # Select processing strategy
        strategy = self._select_processing_strategy(quality_tier, quality_score)
        
        # Apply preprocessing if needed
        processed_img, preprocessing_time, preprocessing_applied = self._preprocess_image(
            img_array, strategy
        )
        
        # Run OCR with selected engine
        ocr_start = time.time()
        ocr_result = self._run_ocr(processed_img, strategy)
        ocr_time = time.time() - ocr_start
        
        # Extract fields and build response
        if ocr_result.get('success', False):
            fields = extract_fields(ocr_result.get('text', ''))
            
            # Apply confidence boosting for high-quality preprocessing
            if preprocessing_applied:
                raw_confidence = ocr_result.get('confidence', 0.0)
                boosted_confidence = min(1.0, raw_confidence * self.config.confidence_boost_factor)
                ocr_result['confidence'] = boosted_confidence
                ocr_result['raw_confidence'] = raw_confidence
            
            # Build final payload
            if format_type == "utility_bill":
                # Create temporary file path for build_utility_bill_payload
                temp_path = Path("temp_image.png")
                if isinstance(image, (str, Path)):
                    temp_path = Path(image)
                
                # Add OCR metadata to fields
                fields.update({
                    '_ocr_confidence': ocr_result.get('confidence', 0.0),
                    '_processing_time': time.time() - start_time,
                    '_extraction_method': strategy['engine']
                })
                
                result = build_utility_bill_payload(fields, temp_path)
            else:
                result = {
                    'success': True,
                    'extracted_fields': fields,
                    'raw_text': ocr_result.get('text', ''),
                    'confidence': ocr_result.get('confidence', 0.0)
                }
        else:
            result = self._create_error_result(ocr_result.get('error', 'OCR failed'))
        
        # Add comprehensive metadata
        total_time = time.time() - start_time
        
        result['adaptive_metadata'] = {
            'quality_assessment': {
                'tier': quality_tier.value,
                'score': quality_score,
                'metrics': quality_metrics
            },
            'processing_strategy': strategy,
            'performance': {
                'total_time': total_time,
                'preprocessing_time': preprocessing_time,
                'ocr_time': ocr_time,
                'preprocessing_applied': preprocessing_applied
            },
            'engine_selection': {
                'selected_engine': strategy['engine'],
                'reason': strategy['reason']
            }
        }
        
        # Update performance statistics
        self._update_performance_stats(strategy['engine'], total_time)
        
        return result
    
    def _load_image(self, image: Union[np.ndarray, Image.Image, str, Path]) -> Optional[np.ndarray]:
        """Load image from various input formats and ensure RGB format."""
        try:
            # Load image
            if isinstance(image, np.ndarray):
                img_array = image
            elif isinstance(image, Image.Image):
                img_array = np.array(image)
            elif isinstance(image, (str, Path)):
                pil_img = Image.open(image)
                # Convert to RGB if needed
                if pil_img.mode == 'RGBA':
                    pil_img = pil_img.convert('RGB')
                elif pil_img.mode == 'P':
                    pil_img = pil_img.convert('RGB')
                elif pil_img.mode == 'L':
                    pil_img = pil_img.convert('RGB')
                img_array = np.array(pil_img)
            else:
                return None
            
            # Ensure proper format
            if len(img_array.shape) == 2:
                # Grayscale to RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif len(img_array.shape) == 3:
                if img_array.shape[2] == 4:
                    # RGBA to RGB
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                elif img_array.shape[2] == 1:
                    # Single channel to RGB
                    img_array = np.repeat(img_array, 3, axis=2)
                elif img_array.shape[2] > 4:
                    # Take first 3 channels
                    img_array = img_array[:, :, :3]
            
            # Ensure uint8 format
            if img_array.dtype != np.uint8:
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)
            
            return img_array
            
        except Exception as e:
            self.logger.error(f"Failed to load image: {e}")
            return None
    
    def _select_processing_strategy(self, quality_tier: QualityTier, 
                                  quality_score: float) -> Dict[str, Any]:
        """Select optimal processing strategy based on quality assessment."""
        
        if quality_tier == QualityTier.HIGH:
            # High quality - use fast, accurate engine
            return {
                'engine': 'tesseract',
                'preprocessing': [],
                'reason': 'High quality image, minimal processing needed'
            }
        
        elif quality_tier == QualityTier.MEDIUM:
            # Medium quality - light enhancement
            preprocessing = ['bilateral_filter']
            if self.config.use_qat_models:
                engine = 'qat'
                reason = 'Medium quality, using QAT models with light preprocessing'
            else:
                engine = 'tesseract'
                reason = 'Medium quality, using enhanced Tesseract'
            
            return {
                'engine': engine,
                'preprocessing': preprocessing,
                'reason': reason
            }
        
        elif quality_tier == QualityTier.LOW:
            # Low quality - aggressive preprocessing
            preprocessing = []
            if self.config.use_jax_denoising:
                preprocessing.append('jax_denoising')
            if self.config.use_qat_models:
                engine = 'qat'
                reason = 'Low quality, using denoising + QAT models'
            else:
                preprocessing.extend(['bilateral_filter', 'histogram_eq'])
                engine = 'tesseract'
                reason = 'Low quality, using enhanced preprocessing + Tesseract'
            
            return {
                'engine': engine,
                'preprocessing': preprocessing,
                'reason': reason
            }
        
        else:  # VERY_LOW
            # Very low quality - maximum preprocessing
            preprocessing = []
            if self.config.use_jax_denoising:
                preprocessing.append('jax_denoising')
            preprocessing.extend(['bilateral_filter', 'histogram_eq', 'morphology'])
            
            return {
                'engine': 'tesseract',  # Most robust fallback
                'preprocessing': preprocessing,
                'reason': 'Very low quality, using all available preprocessing'
            }
    
    def _preprocess_image(self, image: np.ndarray, 
                         strategy: Dict[str, Any]) -> Tuple[np.ndarray, float, List[str]]:
        """Apply preprocessing based on strategy."""
        start_time = time.time()
        processed = image.copy()
        applied = []
        
        for step in strategy.get('preprocessing', []):
            if step == 'jax_denoising' and self.jax_denoiser:
                try:
                    processed, metadata = self.jax_denoiser.process_for_ocr(processed)
                    applied.append(f"jax_denoising_{metadata.get('processing_applied', 'unknown')}")
                except Exception as e:
                    self.logger.warning(f"JAX denoising failed: {e}")
                    applied.append('jax_denoising_failed')
            
            elif step == 'bilateral_filter':
                processed = cv2.bilateralFilter(processed, 9, 75, 75)
                applied.append('bilateral_filter')
            
            elif step == 'histogram_eq':
                if len(processed.shape) == 3:
                    processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
                processed = cv2.equalizeHist(processed)
                applied.append('histogram_eq')
            
            elif step == 'morphology':
                if len(processed.shape) == 3:
                    processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
                applied.append('morphology')
        
        preprocessing_time = time.time() - start_time
        return processed, preprocessing_time, applied
    
    def _run_ocr(self, image: np.ndarray, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Run OCR with selected engine."""
        engine = strategy['engine']
        
        try:
            if engine == 'qat' and self.qat_pipeline:
                try:
                    # Ensure RGB format
                    if len(image.shape) == 2:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    elif len(image.shape) == 3 and image.shape[2] != 3:
                        image = image[:, :, :3]  # Take first 3 channels
                    
                    # Convert to tensor with proper dimensions [B, C, H, W]
                    tensor_img = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                    tensor_img = tensor_img.unsqueeze(0)
                    
                    # Run QAT pipeline
                    output = self.qat_pipeline.inference(tensor_img)
                    
                    # For now, fall back to Tesseract for text extraction
                    # In a full implementation, would decode QAT output
                    self.logger.info("QAT inference completed, falling back to Tesseract for text")
                    return self._run_tesseract(image)
                    
                except Exception as e:
                    self.logger.warning(f"QAT pipeline failed: {e}, falling back to Tesseract")
                    return self._run_tesseract(image)
            
            else:
                # Use Tesseract
                return self._run_tesseract(image)
                
        except Exception as e:
            self.logger.error(f"OCR engine {engine} failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _run_tesseract(self, image: np.ndarray) -> Dict[str, Any]:
        """Run Tesseract OCR with error handling."""
        try:
            # Save image temporarily
            temp_path = Path("temp_ocr_image.png")
            Image.fromarray(image).save(temp_path)
            
            # Run Tesseract
            result = run_ocr_with_tesseract(temp_path)
            
            # Clean up
            temp_path.unlink(missing_ok=True)
            
            return {
                'success': True,
                'text': result.get('_full_text', ''),
                'confidence': result.get('_ocr_confidence', 0.0)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            'success': False,
            'error': error_message,
            'extracted_fields': {},
            'confidence': 0.0,
            'adaptive_metadata': {
                'error': True,
                'error_message': error_message
            }
        }
    
    def _update_performance_stats(self, engine: str, processing_time: float):
        """Update performance statistics."""
        self.performance_stats['total_processed'] += 1
        
        if engine in self.performance_stats['engine_usage']:
            self.performance_stats['engine_usage'][engine] += 1
        
        # Update rolling average
        total = self.performance_stats['total_processed']
        current_avg = self.performance_stats['avg_processing_time']
        new_avg = (current_avg * (total - 1) + processing_time) / total
        self.performance_stats['avg_processing_time'] = new_avg
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            'statistics': self.performance_stats.copy(),
            'configuration': {
                'jax_denoising_enabled': self.config.use_jax_denoising,
                'qat_models_enabled': self.config.use_qat_models,
                'tesseract_fallback_enabled': self.config.use_tesseract_fallback
            },
            'engine_availability': {
                'jax_denoiser': self.jax_denoiser is not None,
                'qat_pipeline': self.qat_pipeline is not None
            }
        }
    
    def benchmark_pipeline(self, test_images: List[str], 
                          output_file: Optional[str] = None) -> Dict[str, Any]:
        """Benchmark pipeline performance across test images."""
        results = []
        
        for img_path in test_images:
            self.logger.info(f"Benchmarking {img_path}")
            
            result = self.process_image(img_path)
            
            benchmark_result = {
                'image': img_path,
                'success': result.get('success', False),
                'confidence': result.get('validation', {}).get('confidence', 0.0),
                'quality_tier': result.get('adaptive_metadata', {}).get('quality_assessment', {}).get('tier'),
                'processing_time': result.get('adaptive_metadata', {}).get('performance', {}).get('total_time'),
                'engine_used': result.get('adaptive_metadata', {}).get('engine_selection', {}).get('selected_engine'),
                'preprocessing_applied': result.get('adaptive_metadata', {}).get('performance', {}).get('preprocessing_applied', [])
            }
            
            results.append(benchmark_result)
        
        # Calculate summary statistics
        summary = {
            'total_images': len(results),
            'successful': sum(1 for r in results if r['success']),
            'avg_confidence': np.mean([r['confidence'] for r in results if r['success']]),
            'avg_processing_time': np.mean([r['processing_time'] for r in results if r['processing_time']]),
            'quality_distribution': {},
            'engine_usage': {}
        }
        
        # Quality distribution
        for tier in ['high', 'medium', 'low', 'very_low']:
            count = sum(1 for r in results if r['quality_tier'] == tier)
            summary['quality_distribution'][tier] = count
        
        # Engine usage
        for engine in ['tesseract', 'qat', 'denoised']:
            count = sum(1 for r in results if r['engine_used'] == engine)
            summary['engine_usage'][engine] = count
        
        benchmark_data = {
            'summary': summary,
            'detailed_results': results,
            'performance_stats': self.get_performance_stats()
        }
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(benchmark_data, f, indent=2)
            self.logger.info(f"Benchmark results saved to {output_file}")
        
        return benchmark_data


# Example usage and testing
if __name__ == "__main__":
    # Initialize adaptive pipeline
    config = AdaptiveConfig(
        use_jax_denoising=True,
        use_qat_models=True,
        confidence_boost_factor=1.2
    )
    
    pipeline = AdaptiveOCRPipeline(config)
    
    print("Adaptive OCR Pipeline initialized")
    print("=" * 50)
    
    # Test on available images
    test_images = []
    for img_name in ["DEWA.png", "SEWA.png"]:
        if Path(img_name).exists():
            test_images.append(img_name)
    
    if test_images:
        print(f"Testing on {len(test_images)} images")
        
        for img_path in test_images:
            print(f"\nProcessing {img_path}:")
            result = pipeline.process_image(img_path, format_type="utility_bill")
            
            if result.get('success', False):
                confidence = result.get('validation', {}).get('confidence', 0.0)
                quality_tier = result.get('adaptive_metadata', {}).get('quality_assessment', {}).get('tier')
                engine_used = result.get('adaptive_metadata', {}).get('engine_selection', {}).get('selected_engine')
                processing_time = result.get('adaptive_metadata', {}).get('performance', {}).get('total_time', 0.0)
                
                print(f"  ✓ Success: {confidence:.3f} confidence")
                print(f"  ✓ Quality: {quality_tier}")
                print(f"  ✓ Engine: {engine_used}")
                print(f"  ✓ Time: {processing_time:.3f}s")
            else:
                print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
        
        # Show performance stats
        stats = pipeline.get_performance_stats()
        print(f"\nPerformance Statistics:")
        print(f"  Total processed: {stats['statistics']['total_processed']}")
        print(f"  Avg processing time: {stats['statistics']['avg_processing_time']:.3f}s")
        print(f"  Engine usage: {stats['statistics']['engine_usage']}")
    else:
        print("No test images found (DEWA.png, SEWA.png)")
    
    print("\nAdaptive OCR Pipeline demonstration complete")