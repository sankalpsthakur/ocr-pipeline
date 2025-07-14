#!/usr/bin/env python3
"""JAX-based Denoising Adapter for Robust OCR

A lightweight, plug-and-play denoising adapter that pre-processes noisy images
before OCR. Uses JAX for efficient computation and supports multiple noise types.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
import flax.linen as nn
from flax import struct
import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Optional, Union, Dict, Any
from pathlib import Path
import time


class ConvBlock(nn.Module):
    """Basic convolutional block with batch norm and activation."""
    features: int
    kernel_size: Tuple[int, int] = (3, 3)
    use_batch_norm: bool = True
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        x = nn.Conv(self.features, self.kernel_size, padding='SAME')(x)
        if self.use_batch_norm:
            x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        return x


class DownBlock(nn.Module):
    """Downsampling block for U-Net encoder."""
    features: int
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        # Two conv blocks followed by max pooling
        x = ConvBlock(self.features)(x, training)
        x = ConvBlock(self.features)(x, training)
        skip = x  # Skip connection
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block for U-Net decoder."""
    features: int
    
    @nn.compact
    def __call__(self, x, skip, training: bool = False):
        # Upsample
        x = nn.ConvTranspose(self.features, kernel_size=(2, 2), 
                           strides=(2, 2), padding='SAME')(x)
        # Concatenate with skip connection
        x = jnp.concatenate([x, skip], axis=-1)
        # Two conv blocks
        x = ConvBlock(self.features)(x, training)
        x = ConvBlock(self.features)(x, training)
        return x


class LightweightUNet(nn.Module):
    """Lightweight U-Net for image denoising.
    
    Designed for fast inference on mobile devices with <1M parameters.
    """
    features: Tuple[int, ...] = (16, 32, 64, 128)  # Reduced from typical UNet
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        # Normalize input to [-1, 1]
        x = (x - 0.5) * 2.0
        
        # Initial conv
        x = ConvBlock(self.features[0])(x, training)
        
        # Encoder (downsampling)
        skips = []
        for feat in self.features[1:]:
            x, skip = DownBlock(feat)(x, training)
            skips.append(skip)
        
        # Bottleneck
        x = ConvBlock(self.features[-1] * 2)(x, training)
        x = ConvBlock(self.features[-1] * 2)(x, training)
        
        # Decoder (upsampling)
        for feat, skip in zip(reversed(self.features[1:]), reversed(skips)):
            x = UpBlock(feat)(x, skip, training)
        
        # Final conv to output channels
        x = nn.Conv(3, kernel_size=(1, 1))(x)  # RGB output
        
        # Residual connection and denormalize
        return jnp.tanh(x) * 0.5 + 0.5


class QualityClassifier(nn.Module):
    """Fast image quality classifier for routing decisions."""
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        # Downsample for speed
        x = nn.avg_pool(x, window_shape=(4, 4), strides=(4, 4))
        
        # Small CNN
        x = ConvBlock(16)(x, training)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = ConvBlock(32)(x, training)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = ConvBlock(64)(x, training)
        
        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))
        
        # FC layers
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(3)(x)  # 3 quality levels: low, medium, high
        
        return nn.softmax(x)


@struct.dataclass
class DenoisingConfig:
    """Configuration for denoising adapter."""
    patch_size: int = 256
    overlap: int = 32
    quality_threshold_low: float = 0.3
    quality_threshold_high: float = 0.7
    max_image_size: int = 2048
    device: str = 'cpu'  # 'cpu' or 'gpu'


class JAXDenoisingAdapter:
    """JAX-based denoising adapter for robust OCR preprocessing."""
    
    def __init__(self, config: Optional[DenoisingConfig] = None):
        self.config = config or DenoisingConfig()
        self.denoiser = None
        self.quality_net = None
        self.params = None
        self.quality_params = None
        self._initialized = False
        
        # Set JAX device
        if self.config.device == 'gpu' and jax.devices('gpu'):
            self.device = jax.devices('gpu')[0]
        else:
            self.device = jax.devices('cpu')[0]
    
    def initialize(self, checkpoint_path: Optional[str] = None):
        """Initialize models and load weights."""
        # Initialize models
        self.denoiser = LightweightUNet()
        self.quality_net = QualityClassifier()
        
        # Initialize parameters with dummy input
        key = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, 256, 256, 3))
        
        self.params = self.denoiser.init(key, dummy_input)
        self.quality_params = self.quality_net.init(key, dummy_input)
        
        # Load checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            self._load_checkpoint(checkpoint_path)
        else:
            print("Warning: No checkpoint loaded, using random initialization")
        
        # JIT compile inference functions
        self._denoise_patch = jit(self._denoise_patch_impl)
        self._classify_quality = jit(self._classify_quality_impl)
        
        self._initialized = True
    
    def _denoise_patch_impl(self, patch: jnp.ndarray) -> jnp.ndarray:
        """Denoise a single patch."""
        return self.denoiser.apply(self.params, patch[None, ...], training=False)[0]
    
    def _classify_quality_impl(self, image: jnp.ndarray) -> jnp.ndarray:
        """Classify image quality."""
        return self.quality_net.apply(self.quality_params, image[None, ...], training=False)[0]
    
    def assess_quality(self, image: np.ndarray) -> Tuple[str, float]:
        """Assess image quality for routing decision.
        
        Returns:
            quality_tier: 'high', 'medium', or 'low'
            confidence: float between 0 and 1
        """
        if not self._initialized:
            self.initialize()
        
        # Ensure RGB format (convert RGBA to RGB if needed)
        if len(image.shape) == 3 and image.shape[2] == 4:
            # Convert RGBA to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif len(image.shape) == 2:
            # Convert grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] != 3:
            # Ensure exactly 3 channels
            image = image[:, :, :3]
        
        # Resize for fast inference
        h, w = image.shape[:2]
        scale = min(1.0, 512 / max(h, w))
        if scale < 1.0:
            resized = cv2.resize(image, None, fx=scale, fy=scale)
        else:
            resized = image
        
        # Convert to JAX array with proper shape [H, W, 3]
        img_jax = jnp.array(resized, dtype=jnp.float32) / 255.0
        
        # Ensure exactly 3 channels
        if img_jax.shape[-1] != 3:
            if img_jax.shape[-1] == 4:
                img_jax = img_jax[:, :, :3]  # Remove alpha channel
            elif img_jax.shape[-1] == 1:
                img_jax = jnp.repeat(img_jax, 3, axis=-1)  # Convert grayscale to RGB
        
        # Get quality scores
        scores = self._classify_quality(img_jax)
        quality_idx = jnp.argmax(scores)
        confidence = float(scores[quality_idx])
        
        tiers = ['low', 'medium', 'high']
        return tiers[quality_idx], confidence
    
    def denoise(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Denoise image using patch-based processing.
        
        Args:
            image: Input noisy image (numpy array or PIL Image)
            
        Returns:
            Denoised image as numpy array (uint8)
        """
        if not self._initialized:
            self.initialize()
        
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure RGB format (handle all channel configurations)
        if len(image.shape) == 2:
            # Grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3:
            if image.shape[2] == 4:
                # RGBA to RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 1:
                # Single channel to RGB
                image = np.repeat(image, 3, axis=2)
            elif image.shape[2] > 4:
                # More than 4 channels, take first 3
                image = image[:, :, :3]
        
        # Ensure we have exactly 3 channels
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image with 3 channels, got shape {image.shape}")
        
        h, w = image.shape[:2]
        
        # Resize if too large
        scale = 1.0
        if max(h, w) > self.config.max_image_size:
            scale = self.config.max_image_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image_scaled = cv2.resize(image, (new_w, new_h))
        else:
            image_scaled = image
            new_h, new_w = h, w
        
        # Process in patches for memory efficiency
        denoised = self._process_patches(image_scaled)
        
        # Resize back if needed
        if scale < 1.0:
            denoised = cv2.resize(denoised, (w, h))
        
        return denoised
    
    def _process_patches(self, image: np.ndarray) -> np.ndarray:
        """Process image in overlapping patches."""
        h, w = image.shape[:2]
        patch_size = self.config.patch_size
        overlap = self.config.overlap
        stride = patch_size - overlap
        
        # Pad image to fit patches
        pad_h = (stride - (h - patch_size) % stride) % stride
        pad_w = (stride - (w - patch_size) % stride) % stride
        padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        
        # Initialize output and weight map
        output = np.zeros_like(padded, dtype=np.float32)
        weights = np.zeros(padded.shape[:2], dtype=np.float32)
        
        # Create weight mask for blending (higher in center)
        weight_mask = self._create_weight_mask(patch_size)
        
        # Process each patch
        for y in range(0, padded.shape[0] - patch_size + 1, stride):
            for x in range(0, padded.shape[1] - patch_size + 1, stride):
                # Extract patch
                patch = padded[y:y+patch_size, x:x+patch_size]
                patch_jax = jnp.array(patch, dtype=jnp.float32) / 255.0
                
                # Denoise patch
                denoised_patch = self._denoise_patch(patch_jax)
                denoised_patch = np.array(denoised_patch) * 255.0
                
                # Accumulate with weights
                output[y:y+patch_size, x:x+patch_size] += denoised_patch * weight_mask[..., None]
                weights[y:y+patch_size, x:x+patch_size] += weight_mask
        
        # Normalize by weights
        output = output / weights[..., None]
        
        # Crop to original size
        output = output[:h, :w]
        
        return output.astype(np.uint8)
    
    def _create_weight_mask(self, size: int) -> np.ndarray:
        """Create weight mask for patch blending."""
        # Linear weight from edges to center
        mask = np.ones((size, size), dtype=np.float32)
        border = self.config.overlap // 2
        
        for i in range(border):
            weight = (i + 1) / border
            mask[i, :] *= weight
            mask[-i-1, :] *= weight
            mask[:, i] *= weight
            mask[:, -i-1] *= weight
        
        return mask
    
    def process_for_ocr(self, image: Union[np.ndarray, Image.Image]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Complete preprocessing pipeline for OCR.
        
        Returns:
            Processed image and metadata dict
        """
        start_time = time.time()
        
        # Convert to numpy
        if isinstance(image, Image.Image):
            original = np.array(image)
        else:
            original = image.copy()
        
        # Assess quality
        quality_tier, quality_score = self.assess_quality(original)
        
        # Decide processing based on quality
        if quality_tier == 'high':
            processed = original
            processing_applied = 'none'
        elif quality_tier == 'medium':
            # Light denoising
            processed = cv2.bilateralFilter(original, 9, 75, 75)
            processing_applied = 'bilateral_filter'
        else:
            # Full denoising
            processed = self.denoise(original)
            processing_applied = 'deep_denoising'
        
        # Additional OCR-specific enhancements
        processed = self._enhance_for_ocr(processed)
        
        metadata = {
            'quality_tier': quality_tier,
            'quality_score': float(quality_score),
            'processing_applied': processing_applied,
            'processing_time': time.time() - start_time,
            'original_shape': original.shape,
            'processed_shape': processed.shape
        }
        
        return processed, metadata
    
    def _enhance_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Apply OCR-specific enhancements."""
        # Convert to grayscale for OCR
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Slight sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) / 9
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
    
    def _load_checkpoint(self, path: str):
        """Load model checkpoint."""
        # Placeholder for checkpoint loading
        # In production, would load from saved JAX checkpoint
        print(f"Would load checkpoint from {path}")
    
    def benchmark(self, image_sizes: list = [(640, 480), (1280, 960), (1920, 1080)]):
        """Benchmark denoising performance."""
        if not self._initialized:
            self.initialize()
        
        print("JAX Denoising Adapter Benchmark")
        print("=" * 50)
        
        for size in image_sizes:
            # Create dummy noisy image
            h, w = size
            image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            
            # Warm up
            _ = self.denoise(image)
            
            # Time multiple runs
            times = []
            for _ in range(5):
                start = time.time()
                _ = self.denoise(image)
                times.append(time.time() - start)
            
            avg_time = np.mean(times)
            print(f"Image size {w}x{h}: {avg_time*1000:.1f}ms (avg of 5 runs)")


# Example usage and testing
if __name__ == "__main__":
    # Initialize adapter
    config = DenoisingConfig(
        patch_size=256,
        overlap=32,
        device='cpu'  # Use 'gpu' if available
    )
    
    adapter = JAXDenoisingAdapter(config)
    adapter.initialize()
    
    print("JAX Denoising Adapter initialized successfully")
    
    # Run benchmark
    adapter.benchmark()
    
    # Test on actual noisy image
    if Path("DEWA.png").exists():
        # Load image
        image = Image.open("DEWA.png")
        image_np = np.array(image)
        
        # Add synthetic noise for testing
        noisy = image_np + np.random.normal(0, 25, image_np.shape)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        
        # Process
        processed, metadata = adapter.process_for_ocr(noisy)
        
        print(f"\nProcessing metadata:")
        for k, v in metadata.items():
            print(f"  {k}: {v}")
        
        # Save results
        Image.fromarray(noisy).save("test_noisy.png")
        Image.fromarray(processed).save("test_denoised.png")
        print("\nSaved test_noisy.png and test_denoised.png for comparison")