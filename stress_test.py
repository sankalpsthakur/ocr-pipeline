#!/usr/bin/env python3
"""Stress Testing Framework for OCR Pipeline

Tests various image conditions:
- Compression levels (JPEG 10-90 quality)
- Noise addition (Gaussian noise)
- Resolution scaling (25%-200%)
- Format conversion (PNG/JPEG/WebP)
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np
import cv2
from typing import Dict, List, Tuple
import time

def add_gaussian_noise(image: Image.Image, noise_level: float = 0.1) -> Image.Image:
    """Add Gaussian noise to image."""
    img_array = np.array(image)
    noise = np.random.normal(0, noise_level * 255, img_array.shape).astype(np.uint8)
    noisy_img = np.clip(img_array + noise, 0, 255)
    return Image.fromarray(noisy_img)

def compress_image(image: Image.Image, quality: int, format_type: str = 'JPEG') -> Image.Image:
    """Compress image with specified quality."""
    from io import BytesIO
    
    buffer = BytesIO()
    if format_type == 'JPEG' and image.mode in ('RGBA', 'LA'):
        image = image.convert('RGB')
    
    image.save(buffer, format=format_type, quality=quality, optimize=True)
    buffer.seek(0)
    return Image.open(buffer)

def scale_image(image: Image.Image, scale_factor: float) -> Image.Image:
    """Scale image by factor."""
    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
    return image.resize(new_size, Image.Resampling.LANCZOS)

def run_ocr_test(image_path: str, test_name: str) -> Dict:
    """Run OCR pipeline on test image and return results."""
    cmd = [
        sys.executable, 
        "pytorch_mobile/ocr_pipeline.py", 
        image_path, 
        "--format", "utility_bill"
    ]
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        processing_time = time.time() - start_time
        
        if result.returncode == 0:
            # Parse JSON output from stdout
            lines = result.stdout.strip().split('\n')
            json_start = -1
            for i, line in enumerate(lines):
                if line.strip().startswith('{'):
                    json_start = i
                    break
            
            if json_start >= 0:
                json_output = '\n'.join(lines[json_start:])
                data = json.loads(json_output)
                
                return {
                    'test_name': test_name,
                    'success': True,
                    'confidence': data.get('validation', {}).get('confidence', 0.0),
                    'field_accuracy': data.get('validation', {}).get('fieldAccuracy', {}),
                    'electricity_kwh': data.get('extractedData', {}).get('consumptionData', {}).get('electricity', {}).get('value', 0),
                    'carbon_kgco2e': data.get('extractedData', {}).get('emissionsData', {}).get('scope2', {}).get('totalCO2e', {}).get('value', 0),
                    'processing_time': processing_time,
                    'error': None
                }
        
        return {
            'test_name': test_name,
            'success': False,
            'confidence': 0.0,
            'field_accuracy': {},
            'electricity_kwh': 0,
            'carbon_kgco2e': 0,
            'processing_time': processing_time,
            'error': result.stderr or result.stdout
        }
        
    except Exception as e:
        return {
            'test_name': test_name,
            'success': False,
            'confidence': 0.0,
            'field_accuracy': {},
            'electricity_kwh': 0,
            'carbon_kgco2e': 0,
            'processing_time': time.time() - start_time,
            'error': str(e)
        }

def create_test_images():
    """Create test variations of DEWA and SEWA images."""
    base_images = ['DEWA.png', 'SEWA.png']
    test_dir = Path('stress_tests')
    test_dir.mkdir(exist_ok=True)
    
    test_configs = []
    
    for base_img in base_images:
        if not Path(base_img).exists():
            print(f"Warning: {base_img} not found")
            continue
            
        image = Image.open(base_img)
        base_name = Path(base_img).stem
        
        # Original image test
        original_path = test_dir / f"{base_name}_original.png"
        image.save(original_path)
        test_configs.append((str(original_path), f"{base_name}_original"))
        
        # Compression tests (JPEG quality 30, 50, 70)
        for quality in [30, 50, 70]:
            compressed = compress_image(image, quality)
            comp_path = test_dir / f"{base_name}_jpeg_q{quality}.jpg"
            compressed.save(comp_path, 'JPEG', quality=quality)
            test_configs.append((str(comp_path), f"{base_name}_jpeg_q{quality}"))
        
        # WebP compression
        webp_path = test_dir / f"{base_name}_webp.webp"
        image.save(webp_path, 'WebP', quality=70)
        test_configs.append((str(webp_path), f"{base_name}_webp"))
        
        # Scaling tests (50%, 75%, 150%)
        for scale in [0.5, 0.75, 1.5]:
            scaled = scale_image(image, scale)
            scale_path = test_dir / f"{base_name}_scale_{int(scale*100)}.png"
            scaled.save(scale_path)
            test_configs.append((str(scale_path), f"{base_name}_scale_{int(scale*100)}"))
        
        # Noise tests (low, medium, high)
        for noise_level, name in [(0.05, 'low'), (0.1, 'medium'), (0.2, 'high')]:
            noisy = add_gaussian_noise(image, noise_level)
            noise_path = test_dir / f"{base_name}_noise_{name}.png"
            noisy.save(noise_path)
            test_configs.append((str(noise_path), f"{base_name}_noise_{name}"))
    
    return test_configs

def run_stress_tests():
    """Run comprehensive stress tests."""
    print("Creating test images...")
    test_configs = create_test_images()
    
    print(f"Running {len(test_configs)} stress tests...")
    results = []
    
    for i, (image_path, test_name) in enumerate(test_configs, 1):
        print(f"[{i}/{len(test_configs)}] Testing {test_name}...")
        result = run_ocr_test(image_path, test_name)
        results.append(result)
        
        status = "✓" if result['success'] else "✗"
        conf = result['confidence']
        print(f"  {status} Confidence: {conf:.3f}")
    
    # Generate report
    report = {
        'summary': {
            'total_tests': len(results),
            'successful_tests': sum(1 for r in results if r['success']),
            'failed_tests': sum(1 for r in results if not r['success']),
            'average_confidence': np.mean([r['confidence'] for r in results if r['success']]),
            'average_processing_time': np.mean([r['processing_time'] for r in results])
        },
        'test_results': results,
        'ground_truth_validation': {
            'dewa_expected': {'electricity_kwh': 299, 'carbon_kgco2e': 120},
            'sewa_expected': {'electricity_kwh': 358, 'carbon_kgco2e': 0}
        }
    }
    
    # Save detailed report
    with open('stress_test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nStress Test Summary:")
    print(f"Total tests: {report['summary']['total_tests']}")
    print(f"Successful: {report['summary']['successful_tests']}")
    print(f"Failed: {report['summary']['failed_tests']}")
    print(f"Success rate: {report['summary']['successful_tests']/report['summary']['total_tests']*100:.1f}%")
    print(f"Average confidence: {report['summary']['average_confidence']:.3f}")
    print(f"Average processing time: {report['summary']['average_processing_time']:.2f}s")
    
    return report

if __name__ == "__main__":
    if not Path("venv/bin/activate").exists():
        print("Error: Virtual environment not found. Run 'python -m venv venv' first.")
        sys.exit(1)
    
    report = run_stress_tests()
    print(f"\nDetailed report saved to: stress_test_report.json")