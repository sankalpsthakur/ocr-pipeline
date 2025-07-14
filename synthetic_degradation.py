#!/usr/bin/env python3
"""Synthetic Image Degradation for Training Robust OCR Models

Generates realistic degradations to create training pairs for denoising
and robust OCR models. Supports various noise types, compression artifacts,
and geometric distortions commonly found in real-world document images.
"""

import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance
from typing import Tuple, Dict, List, Optional, Union, Callable
import random
from pathlib import Path
import json
from dataclasses import dataclass
from scipy import ndimage
from skimage import transform as skimage_transform


@dataclass
class DegradationConfig:
    """Configuration for synthetic degradations."""
    # Noise parameters
    gaussian_noise_range: Tuple[float, float] = (5, 50)  # std deviation
    salt_pepper_range: Tuple[float, float] = (0.01, 0.1)  # proportion
    speckle_noise_range: Tuple[float, float] = (0.01, 0.05)
    
    # Blur parameters
    motion_blur_range: Tuple[int, int] = (5, 15)  # kernel size
    gaussian_blur_range: Tuple[float, float] = (0.5, 3.0)  # sigma
    defocus_blur_range: Tuple[int, int] = (3, 7)  # radius
    
    # Compression parameters
    jpeg_quality_range: Tuple[int, int] = (20, 80)
    webp_quality_range: Tuple[int, int] = (20, 80)
    
    # Geometric parameters
    perspective_range: float = 0.002  # perspective transform strength
    rotation_range: Tuple[float, float] = (-5, 5)  # degrees
    shear_range: Tuple[float, float] = (-0.1, 0.1)
    
    # Lighting parameters
    brightness_range: Tuple[float, float] = (0.5, 1.5)
    contrast_range: Tuple[float, float] = (0.5, 1.5)
    shadow_opacity_range: Tuple[float, float] = (0.3, 0.7)
    
    # Resolution parameters
    downscale_range: Tuple[float, float] = (0.3, 0.8)
    
    # Combination probability
    multi_degradation_prob: float = 0.7  # Probability of applying multiple degradations


class SyntheticDegradation:
    """Generate synthetic degradations for document images."""
    
    def __init__(self, config: Optional[DegradationConfig] = None):
        self.config = config or DegradationConfig()
        self.degradation_functions = {
            'gaussian_noise': self.add_gaussian_noise,
            'salt_pepper': self.add_salt_pepper_noise,
            'speckle': self.add_speckle_noise,
            'motion_blur': self.add_motion_blur,
            'gaussian_blur': self.add_gaussian_blur,
            'defocus_blur': self.add_defocus_blur,
            'jpeg_compression': self.add_jpeg_compression,
            'webp_compression': self.add_webp_compression,
            'perspective': self.add_perspective_distortion,
            'rotation': self.add_rotation,
            'shear': self.add_shear,
            'brightness': self.adjust_brightness,
            'contrast': self.adjust_contrast,
            'shadows': self.add_shadows,
            'downscale': self.add_downscaling
        }
    
    def degrade_image(self, image: Union[np.ndarray, Image.Image], 
                     degradation_types: Optional[List[str]] = None,
                     severity: str = 'medium') -> Tuple[np.ndarray, Dict[str, any]]:
        """Apply degradations to an image.
        
        Args:
            image: Input clean image
            degradation_types: List of degradations to apply. If None, randomly selected
            severity: 'low', 'medium', or 'high' - affects parameter ranges
            
        Returns:
            Degraded image and metadata about applied degradations
        """
        # Convert to numpy array
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()
        
        # Ensure uint8
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        
        # Select degradations
        if degradation_types is None:
            # Randomly select degradations
            if random.random() < self.config.multi_degradation_prob:
                num_degradations = random.randint(2, 4)
                degradation_types = random.sample(
                    list(self.degradation_functions.keys()), 
                    num_degradations
                )
            else:
                degradation_types = [random.choice(list(self.degradation_functions.keys()))]
        
        # Apply degradations
        metadata = {
            'degradations': [],
            'severity': severity
        }
        
        for deg_type in degradation_types:
            if deg_type in self.degradation_functions:
                img, params = self.degradation_functions[deg_type](img, severity)
                metadata['degradations'].append({
                    'type': deg_type,
                    'parameters': params
                })
        
        return img, metadata
    
    def add_gaussian_noise(self, img: np.ndarray, severity: str) -> Tuple[np.ndarray, Dict]:
        """Add Gaussian noise."""
        severity_map = {'low': 0.2, 'medium': 0.5, 'high': 0.8}
        factor = severity_map[severity]
        
        min_std, max_std = self.config.gaussian_noise_range
        std = min_std + (max_std - min_std) * factor
        
        noise = np.random.normal(0, std, img.shape)
        noisy = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        return noisy, {'std': std}
    
    def add_salt_pepper_noise(self, img: np.ndarray, severity: str) -> Tuple[np.ndarray, Dict]:
        """Add salt and pepper noise."""
        severity_map = {'low': 0.2, 'medium': 0.5, 'high': 0.8}
        factor = severity_map[severity]
        
        min_prop, max_prop = self.config.salt_pepper_range
        proportion = min_prop + (max_prop - min_prop) * factor
        
        noisy = img.copy()
        mask = np.random.random(img.shape[:2])
        
        # Salt (white pixels)
        noisy[mask < proportion/2] = 255
        
        # Pepper (black pixels)
        noisy[(mask > proportion/2) & (mask < proportion)] = 0
        
        return noisy, {'proportion': proportion}
    
    def add_speckle_noise(self, img: np.ndarray, severity: str) -> Tuple[np.ndarray, Dict]:
        """Add speckle (multiplicative) noise."""
        severity_map = {'low': 0.2, 'medium': 0.5, 'high': 0.8}
        factor = severity_map[severity]
        
        min_var, max_var = self.config.speckle_noise_range
        variance = min_var + (max_var - min_var) * factor
        
        noise = np.random.randn(*img.shape) * np.sqrt(variance)
        noisy = img + img * noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        
        return noisy, {'variance': variance}
    
    def add_motion_blur(self, img: np.ndarray, severity: str) -> Tuple[np.ndarray, Dict]:
        """Add motion blur."""
        severity_map = {'low': 0.2, 'medium': 0.5, 'high': 0.8}
        factor = severity_map[severity]
        
        min_size, max_size = self.config.motion_blur_range
        size = int(min_size + (max_size - min_size) * factor)
        
        # Create motion blur kernel
        kernel = np.zeros((size, size))
        kernel[int((size-1)/2), :] = np.ones(size)
        kernel = kernel / size
        
        # Random angle
        angle = random.randint(0, 180)
        M = cv2.getRotationMatrix2D((size/2, size/2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (size, size))
        
        blurred = cv2.filter2D(img, -1, kernel)
        
        return blurred, {'kernel_size': size, 'angle': angle}
    
    def add_gaussian_blur(self, img: np.ndarray, severity: str) -> Tuple[np.ndarray, Dict]:
        """Add Gaussian blur."""
        severity_map = {'low': 0.2, 'medium': 0.5, 'high': 0.8}
        factor = severity_map[severity]
        
        min_sigma, max_sigma = self.config.gaussian_blur_range
        sigma = min_sigma + (max_sigma - min_sigma) * factor
        
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        
        return blurred, {'sigma': sigma}
    
    def add_defocus_blur(self, img: np.ndarray, severity: str) -> Tuple[np.ndarray, Dict]:
        """Add defocus blur (disk kernel)."""
        severity_map = {'low': 0.2, 'medium': 0.5, 'high': 0.8}
        factor = severity_map[severity]
        
        min_radius, max_radius = self.config.defocus_blur_range
        radius = int(min_radius + (max_radius - min_radius) * factor)
        
        # Create circular kernel
        kernel = np.zeros((2*radius+1, 2*radius+1))
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = x**2 + y**2 <= radius**2
        kernel[mask] = 1
        kernel = kernel / kernel.sum()
        
        blurred = cv2.filter2D(img, -1, kernel)
        
        return blurred, {'radius': radius}
    
    def add_jpeg_compression(self, img: np.ndarray, severity: str) -> Tuple[np.ndarray, Dict]:
        """Add JPEG compression artifacts."""
        severity_map = {'low': 0.8, 'medium': 0.5, 'high': 0.2}
        factor = severity_map[severity]
        
        min_q, max_q = self.config.jpeg_quality_range
        quality = int(max_q - (max_q - min_q) * (1 - factor))
        
        # Encode and decode JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        compressed = cv2.imdecode(encimg, cv2.IMREAD_COLOR if len(img.shape) == 3 else cv2.IMREAD_GRAYSCALE)
        
        return compressed, {'quality': quality}
    
    def add_webp_compression(self, img: np.ndarray, severity: str) -> Tuple[np.ndarray, Dict]:
        """Add WebP compression artifacts."""
        severity_map = {'low': 0.8, 'medium': 0.5, 'high': 0.2}
        factor = severity_map[severity]
        
        min_q, max_q = self.config.webp_quality_range
        quality = int(max_q - (max_q - min_q) * (1 - factor))
        
        # Use PIL for WebP
        pil_img = Image.fromarray(img)
        import io
        buffer = io.BytesIO()
        pil_img.save(buffer, format='WebP', quality=quality)
        buffer.seek(0)
        compressed = np.array(Image.open(buffer))
        
        return compressed, {'quality': quality}
    
    def add_perspective_distortion(self, img: np.ndarray, severity: str) -> Tuple[np.ndarray, Dict]:
        """Add perspective distortion."""
        severity_map = {'low': 0.3, 'medium': 0.6, 'high': 1.0}
        factor = severity_map[severity]
        
        h, w = img.shape[:2]
        strength = self.config.perspective_range * factor
        
        # Define source points (corners)
        src_pts = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
        
        # Add random perspective distortion
        dst_pts = src_pts.copy()
        for i in range(4):
            dst_pts[i, 0] += random.uniform(-w*strength, w*strength)
            dst_pts[i, 1] += random.uniform(-h*strength, h*strength)
        
        # Compute perspective transform
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        distorted = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        return distorted, {'transform_matrix': M.tolist()}
    
    def add_rotation(self, img: np.ndarray, severity: str) -> Tuple[np.ndarray, Dict]:
        """Add rotation."""
        severity_map = {'low': 0.3, 'medium': 0.6, 'high': 1.0}
        factor = severity_map[severity]
        
        min_angle, max_angle = self.config.rotation_range
        angle = random.uniform(min_angle * factor, max_angle * factor)
        
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        return rotated, {'angle': angle}
    
    def add_shear(self, img: np.ndarray, severity: str) -> Tuple[np.ndarray, Dict]:
        """Add shear transformation."""
        severity_map = {'low': 0.3, 'medium': 0.6, 'high': 1.0}
        factor = severity_map[severity]
        
        min_shear, max_shear = self.config.shear_range
        shear_x = random.uniform(min_shear * factor, max_shear * factor)
        shear_y = random.uniform(min_shear * factor, max_shear * factor)
        
        transform = skimage_transform.AffineTransform(shear=(shear_x, shear_y))
        sheared = skimage_transform.warp(img, transform.inverse, preserve_range=True)
        sheared = sheared.astype(np.uint8)
        
        return sheared, {'shear_x': shear_x, 'shear_y': shear_y}
    
    def adjust_brightness(self, img: np.ndarray, severity: str) -> Tuple[np.ndarray, Dict]:
        """Adjust brightness."""
        severity_map = {'low': 0.3, 'medium': 0.6, 'high': 1.0}
        factor = severity_map[severity]
        
        min_b, max_b = self.config.brightness_range
        # Make it darker or lighter
        if random.random() < 0.5:
            brightness = 1.0 - (1.0 - min_b) * factor
        else:
            brightness = 1.0 + (max_b - 1.0) * factor
        
        adjusted = cv2.convertScaleAbs(img, alpha=brightness, beta=0)
        
        return adjusted, {'factor': brightness}
    
    def adjust_contrast(self, img: np.ndarray, severity: str) -> Tuple[np.ndarray, Dict]:
        """Adjust contrast."""
        severity_map = {'low': 0.3, 'medium': 0.6, 'high': 1.0}
        factor = severity_map[severity]
        
        min_c, max_c = self.config.contrast_range
        # Lower or higher contrast
        if random.random() < 0.5:
            contrast = 1.0 - (1.0 - min_c) * factor
        else:
            contrast = 1.0 + (max_c - 1.0) * factor
        
        # Apply contrast
        mean = np.mean(img)
        adjusted = mean + contrast * (img - mean)
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        return adjusted, {'factor': contrast}
    
    def add_shadows(self, img: np.ndarray, severity: str) -> Tuple[np.ndarray, Dict]:
        """Add realistic shadows."""
        severity_map = {'low': 0.3, 'medium': 0.6, 'high': 1.0}
        factor = severity_map[severity]
        
        h, w = img.shape[:2]
        
        # Create shadow mask
        shadow_type = random.choice(['corner', 'edge', 'circular'])
        
        if shadow_type == 'corner':
            # Corner shadow
            mask = np.zeros((h, w))
            corner = random.choice(['tl', 'tr', 'bl', 'br'])
            if corner == 'tl':
                cv2.ellipse(mask, (0, 0), (int(w*0.7), int(h*0.7)), 0, 0, 90, 1, -1)
            elif corner == 'tr':
                cv2.ellipse(mask, (w, 0), (int(w*0.7), int(h*0.7)), 0, 90, 180, 1, -1)
            elif corner == 'bl':
                cv2.ellipse(mask, (0, h), (int(w*0.7), int(h*0.7)), 0, 270, 360, 1, -1)
            else:
                cv2.ellipse(mask, (w, h), (int(w*0.7), int(h*0.7)), 0, 180, 270, 1, -1)
        
        elif shadow_type == 'edge':
            # Edge shadow
            mask = np.zeros((h, w))
            edge = random.choice(['top', 'bottom', 'left', 'right'])
            shadow_width = int(min(h, w) * 0.3 * factor)
            if edge == 'top':
                mask[:shadow_width, :] = 1
            elif edge == 'bottom':
                mask[-shadow_width:, :] = 1
            elif edge == 'left':
                mask[:, :shadow_width] = 1
            else:
                mask[:, -shadow_width:] = 1
        
        else:
            # Circular shadow
            mask = np.zeros((h, w))
            center_x = random.randint(int(w*0.3), int(w*0.7))
            center_y = random.randint(int(h*0.3), int(h*0.7))
            radius = int(min(h, w) * 0.4 * factor)
            cv2.circle(mask, (center_x, center_y), radius, 1, -1)
        
        # Blur shadow mask
        mask = cv2.GaussianBlur(mask, (51, 51), 20)
        
        # Apply shadow
        min_opacity, max_opacity = self.config.shadow_opacity_range
        opacity = min_opacity + (max_opacity - min_opacity) * factor
        
        shadow_img = img.copy()
        for i in range(img.shape[2] if len(img.shape) == 3 else 1):
            if len(img.shape) == 3:
                shadow_img[:, :, i] = img[:, :, i] * (1 - mask * opacity)
            else:
                shadow_img = img * (1 - mask * opacity)
        
        return shadow_img.astype(np.uint8), {'type': shadow_type, 'opacity': opacity}
    
    def add_downscaling(self, img: np.ndarray, severity: str) -> Tuple[np.ndarray, Dict]:
        """Add downscaling degradation."""
        severity_map = {'low': 0.8, 'medium': 0.5, 'high': 0.2}
        factor = severity_map[severity]
        
        min_scale, max_scale = self.config.downscale_range
        scale = max_scale - (max_scale - min_scale) * (1 - factor)
        
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Downscale and upscale back
        downscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        upscaled = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_CUBIC)
        
        return upscaled, {'scale': scale}
    
    def create_training_pair(self, clean_image: Union[np.ndarray, Image.Image],
                           num_variants: int = 5) -> List[Tuple[np.ndarray, np.ndarray, Dict]]:
        """Create multiple degraded variants of a clean image for training.
        
        Returns:
            List of (clean, degraded, metadata) tuples
        """
        pairs = []
        
        # Convert to numpy
        if isinstance(clean_image, Image.Image):
            clean = np.array(clean_image)
        else:
            clean = clean_image.copy()
        
        severities = ['low', 'medium', 'high']
        
        for i in range(num_variants):
            # Random severity
            severity = random.choice(severities)
            
            # Apply degradation
            degraded, metadata = self.degrade_image(clean, severity=severity)
            
            pairs.append((clean, degraded, metadata))
        
        return pairs


def create_ocr_training_dataset(image_paths: List[str], 
                              output_dir: str,
                              num_variants_per_image: int = 10):
    """Create a training dataset for robust OCR from clean images.
    
    Args:
        image_paths: List of paths to clean document images
        output_dir: Directory to save degraded images and metadata
        num_variants_per_image: Number of degraded variants per clean image
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    degrader = SyntheticDegradation()
    metadata_all = []
    
    for idx, img_path in enumerate(image_paths):
        print(f"Processing {idx+1}/{len(image_paths)}: {img_path}")
        
        # Load image
        clean_img = Image.open(img_path)
        
        # Create variants
        pairs = degrader.create_training_pair(clean_img, num_variants_per_image)
        
        for var_idx, (clean, degraded, metadata) in enumerate(pairs):
            # Save degraded image
            degraded_name = f"{Path(img_path).stem}_degraded_{var_idx:03d}.png"
            degraded_path = output_path / degraded_name
            Image.fromarray(degraded).save(degraded_path)
            
            # Save clean image (first variant only)
            if var_idx == 0:
                clean_name = f"{Path(img_path).stem}_clean.png"
                clean_path = output_path / clean_name
                Image.fromarray(clean).save(clean_path)
            
            # Add to metadata
            metadata['source_image'] = Path(img_path).name
            metadata['degraded_image'] = degraded_name
            metadata['clean_image'] = f"{Path(img_path).stem}_clean.png"
            metadata_all.append(metadata)
    
    # Save metadata
    metadata_path = output_path / "degradation_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata_all, f, indent=2)
    
    print(f"Created {len(metadata_all)} degraded images in {output_dir}")
    print(f"Metadata saved to {metadata_path}")


# Example usage
if __name__ == "__main__":
    # Test degradation
    degrader = SyntheticDegradation()
    
    # Test on DEWA image if available
    if Path("DEWA.png").exists():
        img = Image.open("DEWA.png")
        
        # Apply different severities
        for severity in ['low', 'medium', 'high']:
            degraded, metadata = degrader.degrade_image(
                img, 
                degradation_types=['gaussian_noise', 'motion_blur', 'jpeg_compression'],
                severity=severity
            )
            
            # Save result
            Image.fromarray(degraded).save(f"test_degraded_{severity}.png")
            print(f"Created test_degraded_{severity}.png")
            print(f"Metadata: {metadata}")
            print()
    
    # Create sample training dataset
    if Path("DEWA.png").exists() and Path("SEWA.png").exists():
        create_ocr_training_dataset(
            ["DEWA.png", "SEWA.png"],
            "synthetic_training_data",
            num_variants_per_image=5
        )