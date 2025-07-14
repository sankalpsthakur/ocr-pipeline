#!/usr/bin/env python3
"""Confidence-Accuracy Correlation Analysis

Validates that confidence scores correlate with actual field extraction accuracy.
"""

import json
import numpy as np
from scipy.stats import pearsonr
from pathlib import Path

def analyze_confidence_accuracy():
    """Analyze correlation between confidence scores and accuracy."""
    
    # Load stress test results
    with open('stress_test_report.json') as f:
        report = json.load(f)
    
    # Ground truth values
    ground_truth = {
        'DEWA': {'electricity_kwh': 299, 'carbon_kgco2e': 120},
        'SEWA': {'electricity_kwh': 358, 'carbon_kgco2e': 0}  # SEWA has no carbon data
    }
    
    # Extract data for correlation analysis
    confidence_scores = []
    accuracy_scores = []
    
    for result in report['test_results']:
        if not result['success']:
            continue
            
        test_name = result['test_name']
        confidence = result['confidence']
        
        # Determine which ground truth to use
        if 'DEWA' in test_name:
            gt = ground_truth['DEWA']
        elif 'SEWA' in test_name:
            gt = ground_truth['SEWA']
        else:
            continue
        
        # Calculate field accuracy
        electricity_accuracy = 1.0 if abs(result['electricity_kwh'] - gt['electricity_kwh']) < 5 else 0.0
        carbon_accuracy = 1.0 if abs(result['carbon_kgco2e'] - gt['carbon_kgco2e']) < 5 else 0.0
        
        # For SEWA, carbon accuracy is always 1.0 since expected is 0
        if 'SEWA' in test_name:
            carbon_accuracy = 1.0 if result['carbon_kgco2e'] == 0 else 0.0
        
        # Overall accuracy (average of field accuracies)
        overall_accuracy = (electricity_accuracy + carbon_accuracy) / 2
        
        confidence_scores.append(confidence)
        accuracy_scores.append(overall_accuracy)
    
    # Calculate correlation
    if len(confidence_scores) >= 3:
        correlation, p_value = pearsonr(confidence_scores, accuracy_scores)
    else:
        correlation, p_value = 0, 1
    
    # Analyze confidence ranges
    high_conf = [acc for conf, acc in zip(confidence_scores, accuracy_scores) if conf > 0.8]
    medium_conf = [acc for conf, acc in zip(confidence_scores, accuracy_scores) if 0.4 <= conf <= 0.8]
    low_conf = [acc for conf, acc in zip(confidence_scores, accuracy_scores) if conf < 0.4]
    
    analysis = {
        'correlation_analysis': {
            'pearson_correlation': correlation,
            'p_value': p_value,
            'interpretation': 'Strong positive correlation' if correlation > 0.7 else 
                           'Moderate positive correlation' if correlation > 0.4 else
                           'Weak correlation',
            'statistical_significance': 'Significant' if p_value < 0.05 else 'Not significant'
        },
        'confidence_ranges': {
            'high_confidence': {
                'range': '> 0.8',
                'count': len(high_conf),
                'avg_accuracy': np.mean(high_conf) if high_conf else 0,
                'accuracy_std': np.std(high_conf) if high_conf else 0
            },
            'medium_confidence': {
                'range': '0.4 - 0.8',
                'count': len(medium_conf),
                'avg_accuracy': np.mean(medium_conf) if medium_conf else 0,
                'accuracy_std': np.std(medium_conf) if medium_conf else 0
            },
            'low_confidence': {
                'range': '< 0.4',
                'count': len(low_conf),
                'avg_accuracy': np.mean(low_conf) if low_conf else 0,
                'accuracy_std': np.std(low_conf) if low_conf else 0
            }
        },
        'calibration_quality': {
            'total_samples': len(confidence_scores),
            'perfect_extractions': sum(1 for acc in accuracy_scores if acc == 1.0),
            'failed_extractions': sum(1 for acc in accuracy_scores if acc == 0.0),
            'partial_extractions': sum(1 for acc in accuracy_scores if 0 < acc < 1.0)
        },
        'detailed_results': [
            {
                'test_name': result['test_name'],
                'confidence': conf,
                'accuracy': acc,
                'electricity_correct': abs(result['electricity_kwh'] - 
                    (ground_truth['DEWA']['electricity_kwh'] if 'DEWA' in result['test_name'] 
                     else ground_truth['SEWA']['electricity_kwh'])) < 5,
                'carbon_correct': abs(result['carbon_kgco2e'] - 
                    (ground_truth['DEWA']['carbon_kgco2e'] if 'DEWA' in result['test_name'] 
                     else ground_truth['SEWA']['carbon_kgco2e'])) < 5
            }
            for result, conf, acc in zip(report['test_results'], confidence_scores, accuracy_scores)
            if result['success']
        ]
    }
    
    return analysis

if __name__ == "__main__":
    analysis = analyze_confidence_accuracy()
    
    # Save analysis
    with open('confidence_accuracy_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print("Confidence-Accuracy Correlation Analysis")
    print("=" * 50)
    print(f"Pearson Correlation: {analysis['correlation_analysis']['pearson_correlation']:.3f}")
    print(f"P-value: {analysis['correlation_analysis']['p_value']:.3f}")
    print(f"Interpretation: {analysis['correlation_analysis']['interpretation']}")
    print(f"Statistical Significance: {analysis['correlation_analysis']['statistical_significance']}")
    print()
    
    print("Confidence Range Analysis:")
    for range_name, data in analysis['confidence_ranges'].items():
        print(f"{range_name.replace('_', ' ').title()}: {data['range']}")
        print(f"  Count: {data['count']}")
        print(f"  Avg Accuracy: {data['avg_accuracy']:.3f}")
        print(f"  Std Dev: {data['accuracy_std']:.3f}")
        print()
    
    print("Calibration Quality:")
    cal = analysis['calibration_quality']
    print(f"Total samples: {cal['total_samples']}")
    print(f"Perfect extractions: {cal['perfect_extractions']} ({cal['perfect_extractions']/cal['total_samples']*100:.1f}%)")
    print(f"Failed extractions: {cal['failed_extractions']} ({cal['failed_extractions']/cal['total_samples']*100:.1f}%)")
    print(f"Partial extractions: {cal['partial_extractions']} ({cal['partial_extractions']/cal['total_samples']*100:.1f}%)")
    
    print(f"\nDetailed analysis saved to: confidence_accuracy_analysis.json")