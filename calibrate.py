#!/usr/bin/env python3
"""
CLI module for training confidence calibration models.

Usage:
    python -m pipeline.calibrate path/to/ground_truth.json

This module loads labeled OCR validation data and trains the ConfidenceCalibrator
to map raw OCR confidence scores to empirical accuracy rates.
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pickle

# Import pipeline components
from pipeline import ConfidenceCalibrator, run_ocr, extract_fields, OcrResult
import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

def load_ground_truth_data(json_path: Path) -> List[Dict[str, Any]]:
    """Load ground truth validation data from JSON file.
    
    Expected format:
    [
        {
            "file_path": "path/to/document.pdf",
            "ground_truth": {
                "electricity_kwh": 299,
                "carbon_kgco2e": 120
            }
        },
        ...
    ]
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    LOGGER.info(f"Loaded {len(data)} ground truth samples from {json_path}")
    return data

def calculate_field_accuracy(predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """Calculate field-level accuracy for a single prediction."""
    correct_fields = 0
    total_fields = 0
    
    for field, true_value in ground_truth.items():
        if field.startswith('_'):  # Skip metadata fields
            continue
            
        total_fields += 1
        pred_value = predicted.get(field)
        
        if pred_value == true_value:
            correct_fields += 1
        elif isinstance(true_value, (int, float)) and isinstance(pred_value, (int, float)):
            # Allow small numerical tolerance for floating point values
            if abs(pred_value - true_value) < 0.01:
                correct_fields += 1
    
    return correct_fields / total_fields if total_fields > 0 else 0.0

def run_validation_and_collect_calibration_data(ground_truth_data: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    """Run OCR on validation set and collect (confidence, accuracy) pairs for calibration."""
    confidences = []
    accuracies = []
    
    total_samples = len(ground_truth_data)
    LOGGER.info(f"Processing {total_samples} validation samples...")
    
    for i, sample in enumerate(ground_truth_data):
        file_path = Path(sample['file_path'])
        ground_truth = sample['ground_truth']
        
        if not file_path.exists():
            LOGGER.warning(f"File not found: {file_path}, skipping")
            continue
        
        try:
            # Run OCR
            ocr_result = run_ocr(file_path)
            
            # Extract fields
            extracted_fields = extract_fields(ocr_result.text)
            
            # Calculate accuracy
            accuracy = calculate_field_accuracy(extracted_fields, ground_truth)
            
            # Record confidence and accuracy
            confidences.append(ocr_result.field_confidence)
            accuracies.append(accuracy)
            
            LOGGER.info(f"Sample {i+1}/{total_samples}: {file_path.name} - "
                       f"Confidence: {ocr_result.field_confidence:.3f}, "
                       f"Accuracy: {accuracy:.3f}, "
                       f"Engine: {ocr_result.engine}")
            
        except Exception as e:
            LOGGER.error(f"Error processing {file_path}: {e}")
            continue
    
    LOGGER.info(f"Collected {len(confidences)} valid calibration data points")
    return confidences, accuracies

def train_calibration_model(confidences: List[float], accuracies: List[float]) -> ConfidenceCalibrator:
    """Train confidence calibration model and save to disk."""
    if len(confidences) < 10:
        raise ValueError(f"Insufficient calibration data: {len(confidences)} samples (need at least 10)")
    
    LOGGER.info("Training confidence calibration model...")
    
    # Create and train calibrator
    calibrator = ConfidenceCalibrator()
    calibrator.fit_from_validation_data(confidences, accuracies)
    
    # Save calibration model
    calibration_path = Path("calibration_models.pkl")
    with open(calibration_path, 'wb') as f:
        pickle.dump(calibrator, f)
    
    LOGGER.info(f"Calibration model saved to {calibration_path}")
    
    # Report calibration statistics
    import numpy as np
    pre_calibration_accuracy = np.mean(accuracies)
    
    # Test calibration on training data (for reporting only)
    calibrated_confidences = [calibrator.calibrate_confidence(conf) for conf in confidences]
    
    LOGGER.info(f"Pre-calibration mean accuracy: {pre_calibration_accuracy:.3f}")
    LOGGER.info(f"Mean raw confidence: {np.mean(confidences):.3f}")
    LOGGER.info(f"Mean calibrated confidence: {np.mean(calibrated_confidences):.3f}")
    
    return calibrator

def update_config_thresholds(calibrator: ConfidenceCalibrator, target_precision: float = 0.95):
    """Update config thresholds based on calibration curve."""
    # Find confidence threshold that achieves target precision
    test_confidences = [i/100 for i in range(50, 100)]  # 0.50 to 0.99
    
    best_threshold = 0.95  # Default fallback
    
    for conf in test_confidences:
        calibrated_conf = calibrator.calibrate_confidence(conf)
        if calibrated_conf >= target_precision:
            best_threshold = conf
            break
    
    LOGGER.info(f"Recommended TAU_FIELD_ACCEPT threshold: {best_threshold:.3f} "
               f"(achieves {target_precision:.1%} precision)")
    
    # Save recommendation to a config file
    recommendations = {
        'TAU_FIELD_ACCEPT': best_threshold,
        'TAU_ENHANCER_PASS': max(0.85, best_threshold - 0.10),
        'TAU_LLM_PASS': max(0.80, best_threshold - 0.15),
    }
    
    with open('recommended_thresholds.json', 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    LOGGER.info("Threshold recommendations saved to recommended_thresholds.json")
    return recommendations

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Train OCR confidence calibration model')
    parser.add_argument('ground_truth_path', type=Path, 
                       help='Path to ground truth JSON file')
    parser.add_argument('--target-precision', type=float, default=0.95,
                       help='Target precision for threshold recommendations (default: 0.95)')
    parser.add_argument('--min-samples', type=int, default=10,
                       help='Minimum number of samples required (default: 10)')
    
    args = parser.parse_args()
    
    if not args.ground_truth_path.exists():
        LOGGER.error(f"Ground truth file not found: {args.ground_truth_path}")
        return 1
    
    try:
        # Load validation data
        ground_truth_data = load_ground_truth_data(args.ground_truth_path)
        
        if len(ground_truth_data) < args.min_samples:
            LOGGER.error(f"Insufficient samples: {len(ground_truth_data)} < {args.min_samples}")
            return 1
        
        # Collect calibration data
        confidences, accuracies = run_validation_and_collect_calibration_data(ground_truth_data)
        
        if len(confidences) < args.min_samples:
            LOGGER.error(f"Insufficient valid samples after processing: {len(confidences)} < {args.min_samples}")
            return 1
        
        # Train calibration model
        calibrator = train_calibration_model(confidences, accuracies)
        
        # Update threshold recommendations
        recommendations = update_config_thresholds(calibrator, args.target_precision)
        
        LOGGER.info("Calibration training completed successfully!")
        LOGGER.info("To use the calibrated model, ensure calibration_models.pkl is in your working directory")
        LOGGER.info("Consider updating your config with the recommended thresholds")
        
        return 0
        
    except Exception as e:
        LOGGER.error(f"Calibration training failed: {e}")
        return 1

if __name__ == '__main__':
    exit(main())