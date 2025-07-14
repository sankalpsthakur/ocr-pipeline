#!/usr/bin/env python3
"""Robustness Evaluation Framework

Comprehensive evaluation framework for comparing OCR robustness before and
after implementing adaptive preprocessing. Tests performance across various
degradation levels and generates detailed robustness curves.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Import our components
from adaptive_ocr_pipeline import AdaptiveOCRPipeline, AdaptiveConfig
from synthetic_degradation import SyntheticDegradation, DegradationConfig
from pytorch_mobile.ocr_pipeline import run_ocr_with_tesseract, extract_fields


@dataclass
class EvaluationConfig:
    """Configuration for robustness evaluation."""
    # Test parameters
    degradation_levels: List[str] = None  # ['low', 'medium', 'high']
    degradation_types: List[str] = None   # Specific degradation types to test
    num_variants_per_level: int = 3
    
    # Evaluation settings
    ground_truth_fields: Dict[str, Dict[str, str]] = None
    confidence_thresholds: List[float] = None  # [0.3, 0.5, 0.7, 0.9]
    
    # Output settings
    save_degraded_images: bool = False
    generate_plots: bool = True
    output_dir: str = "robustness_evaluation"
    
    def __post_init__(self):
        if self.degradation_levels is None:
            self.degradation_levels = ['low', 'medium', 'high']
        
        if self.degradation_types is None:
            self.degradation_types = [
                'gaussian_noise', 'salt_pepper', 'motion_blur', 'jpeg_compression',
                'downscale', 'brightness', 'shadows'
            ]
        
        if self.confidence_thresholds is None:
            self.confidence_thresholds = [0.3, 0.5, 0.7, 0.9]
        
        if self.ground_truth_fields is None:
            self.ground_truth_fields = {
                'DEWA.png': {
                    'electricity_kwh': '299',
                    'carbon_kgco2e': '120',
                    'account_number': '2052672303'
                },
                'SEWA.png': {
                    'electricity_kwh': '358',
                    'account_number': '7965198366'
                }
            }


class RobustnessEvaluator:
    """Framework for evaluating OCR robustness."""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.degrader = SyntheticDegradation(DegradationConfig())
        
        # Initialize pipelines
        self.baseline_pipeline = None  # Simple Tesseract
        self.adaptive_pipeline = AdaptiveOCRPipeline(AdaptiveConfig())
        
        # Results storage
        self.evaluation_results = {
            'baseline': [],
            'adaptive': [],
            'comparison': {}
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def evaluate_image_robustness(self, image_path: str) -> Dict[str, any]:
        """Evaluate robustness of a single image across all degradations."""
        self.logger.info(f"Evaluating robustness for {image_path}")
        
        # Load clean image
        clean_image = Image.open(image_path)
        image_name = Path(image_path).name
        
        results = {
            'image': image_name,
            'clean_results': {},
            'degraded_results': {
                'baseline': {},
                'adaptive': {}
            }
        }
        
        # Test clean image first
        results['clean_results'] = self._evaluate_clean_image(clean_image, image_name)
        
        # Test each degradation type and level
        for deg_type in self.config.degradation_types:
            for severity in self.config.degradation_levels:
                self.logger.info(f"  Testing {deg_type} at {severity} severity")
                
                # Generate degraded variants
                degraded_results = []
                
                for variant in range(self.config.num_variants_per_level):
                    degraded_img, metadata = self.degrader.degrade_image(
                        clean_image, [deg_type], severity
                    )
                    
                    # Save degraded image if requested
                    if self.config.save_degraded_images:
                        degraded_path = self.output_dir / f"{Path(image_path).stem}_{deg_type}_{severity}_{variant}.png"
                        Image.fromarray(degraded_img).save(degraded_path)
                    
                    # Evaluate with both pipelines
                    baseline_result = self._evaluate_with_baseline(degraded_img, image_name)
                    adaptive_result = self._evaluate_with_adaptive(degraded_img, image_name)
                    
                    degraded_results.append({
                        'variant': variant,
                        'degradation_metadata': metadata,
                        'baseline': baseline_result,
                        'adaptive': adaptive_result
                    })
                
                # Store results
                key = f"{deg_type}_{severity}"
                results['degraded_results']['baseline'][key] = degraded_results
                results['degraded_results']['adaptive'][key] = degraded_results
        
        return results
    
    def _evaluate_clean_image(self, image: Image.Image, image_name: str) -> Dict[str, any]:
        """Evaluate clean image with both pipelines."""
        img_array = np.array(image)
        
        baseline_result = self._evaluate_with_baseline(img_array, image_name)
        adaptive_result = self._evaluate_with_adaptive(img_array, image_name)
        
        return {
            'baseline': baseline_result,
            'adaptive': adaptive_result
        }
    
    def _evaluate_with_baseline(self, image: np.ndarray, image_name: str) -> Dict[str, any]:
        """Evaluate with baseline Tesseract pipeline."""
        start_time = time.time()
        
        try:
            # Save temporary image
            temp_path = self.output_dir / "temp_baseline.png"
            Image.fromarray(image).save(temp_path)
            
            # Run Tesseract
            ocr_result = run_ocr_with_tesseract(temp_path)
            processing_time = time.time() - start_time
            
            # Extract fields
            fields = extract_fields(ocr_result.get('_full_text', ''))
            
            # Calculate accuracy
            accuracy_metrics = self._calculate_accuracy(fields, image_name)
            
            # Clean up
            temp_path.unlink(missing_ok=True)
            
            return {
                'success': True,
                'confidence': ocr_result.get('_ocr_confidence', 0.0),
                'processing_time': processing_time,
                'extracted_fields': fields,
                'accuracy_metrics': accuracy_metrics,
                'engine': 'tesseract_baseline'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'confidence': 0.0,
                'processing_time': time.time() - start_time,
                'extracted_fields': {},
                'accuracy_metrics': {'field_accuracy': 0.0, 'exact_matches': 0},
                'engine': 'tesseract_baseline'
            }
    
    def _evaluate_with_adaptive(self, image: np.ndarray, image_name: str) -> Dict[str, any]:
        """Evaluate with adaptive pipeline."""
        try:
            result = self.adaptive_pipeline.process_image(image, format_type="basic")
            
            if result.get('success', False):
                fields = result.get('extracted_fields', {})
                accuracy_metrics = self._calculate_accuracy(fields, image_name)
                
                return {
                    'success': True,
                    'confidence': result.get('confidence', 0.0),
                    'processing_time': result.get('adaptive_metadata', {}).get('performance', {}).get('total_time', 0.0),
                    'extracted_fields': fields,
                    'accuracy_metrics': accuracy_metrics,
                    'engine': result.get('adaptive_metadata', {}).get('engine_selection', {}).get('selected_engine', 'adaptive'),
                    'quality_tier': result.get('adaptive_metadata', {}).get('quality_assessment', {}).get('tier'),
                    'preprocessing_applied': result.get('adaptive_metadata', {}).get('performance', {}).get('preprocessing_applied', [])
                }
            else:
                return {
                    'success': False,
                    'error': result.get('error', 'Unknown error'),
                    'confidence': 0.0,
                    'processing_time': 0.0,
                    'extracted_fields': {},
                    'accuracy_metrics': {'field_accuracy': 0.0, 'exact_matches': 0},
                    'engine': 'adaptive_failed'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'confidence': 0.0,
                'processing_time': 0.0,
                'extracted_fields': {},
                'accuracy_metrics': {'field_accuracy': 0.0, 'exact_matches': 0},
                'engine': 'adaptive_error'
            }
    
    def _calculate_accuracy(self, extracted_fields: Dict[str, str], image_name: str) -> Dict[str, float]:
        """Calculate field extraction accuracy against ground truth."""
        if image_name not in self.config.ground_truth_fields:
            return {'field_accuracy': 0.0, 'exact_matches': 0, 'total_fields': 0}
        
        ground_truth = self.config.ground_truth_fields[image_name]
        total_fields = len(ground_truth)
        exact_matches = 0
        partial_matches = 0
        
        for field, expected_value in ground_truth.items():
            extracted_value = str(extracted_fields.get(field, '')).strip()
            
            if extracted_value == expected_value:
                exact_matches += 1
            elif extracted_value and expected_value:
                # Check for partial matches (useful for numeric fields)
                try:
                    if field in ['electricity_kwh', 'carbon_kgco2e']:
                        extracted_num = float(extracted_value)
                        expected_num = float(expected_value)
                        # Within 5% is considered partial match
                        if abs(extracted_num - expected_num) / expected_num <= 0.05:
                            partial_matches += 0.5
                except:
                    pass
        
        field_accuracy = (exact_matches + partial_matches) / total_fields if total_fields > 0 else 0.0
        
        return {
            'field_accuracy': field_accuracy,
            'exact_matches': exact_matches,
            'partial_matches': partial_matches,
            'total_fields': total_fields
        }
    
    def run_comprehensive_evaluation(self, image_paths: List[str]) -> Dict[str, any]:
        """Run comprehensive robustness evaluation on multiple images."""
        self.logger.info(f"Starting comprehensive evaluation on {len(image_paths)} images")
        
        all_results = []
        
        # Process each image
        for image_path in image_paths:
            if not Path(image_path).exists():
                self.logger.warning(f"Image not found: {image_path}")
                continue
            
            image_results = self.evaluate_image_robustness(image_path)
            all_results.append(image_results)
        
        # Compile comprehensive analysis
        analysis = self._compile_comprehensive_analysis(all_results)
        
        # Save detailed results
        results_file = self.output_dir / "comprehensive_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'config': self.config.__dict__,
                'detailed_results': all_results,
                'analysis': analysis
            }, f, indent=2)
        
        self.logger.info(f"Detailed results saved to {results_file}")
        
        # Generate visualizations
        if self.config.generate_plots:
            self._generate_robustness_plots(analysis)
        
        return analysis
    
    def _compile_comprehensive_analysis(self, all_results: List[Dict]) -> Dict[str, any]:
        """Compile comprehensive analysis from all results."""
        analysis = {
            'summary': {},
            'degradation_analysis': {},
            'improvement_metrics': {},
            'confidence_analysis': {},
            'performance_analysis': {}
        }
        
        # Initialize data structures
        baseline_scores = {deg: {sev: [] for sev in self.config.degradation_levels} for deg in self.config.degradation_types}
        adaptive_scores = {deg: {sev: [] for sev in self.config.degradation_levels} for deg in self.config.degradation_types}
        
        baseline_confidences = {deg: {sev: [] for sev in self.config.degradation_levels} for deg in self.config.degradation_types}
        adaptive_confidences = {deg: {sev: [] for sev in self.config.degradation_levels} for deg in self.config.degradation_types}
        
        # Collect data
        for result in all_results:
            for deg_type in self.config.degradation_types:
                for severity in self.config.degradation_levels:
                    key = f"{deg_type}_{severity}"
                    
                    if key in result['degraded_results']['baseline']:
                        variants = result['degraded_results']['baseline'][key]
                        
                        for variant_data in variants:
                            baseline_result = variant_data['baseline']
                            adaptive_result = variant_data['adaptive']
                            
                            # Accuracy scores
                            baseline_acc = baseline_result['accuracy_metrics']['field_accuracy']
                            adaptive_acc = adaptive_result['accuracy_metrics']['field_accuracy']
                            
                            baseline_scores[deg_type][severity].append(baseline_acc)
                            adaptive_scores[deg_type][severity].append(adaptive_acc)
                            
                            # Confidence scores
                            baseline_confidences[deg_type][severity].append(baseline_result['confidence'])
                            adaptive_confidences[deg_type][severity].append(adaptive_result['confidence'])
        
        # Calculate summary statistics
        analysis['summary'] = self._calculate_summary_stats(baseline_scores, adaptive_scores)
        analysis['degradation_analysis'] = self._analyze_by_degradation(baseline_scores, adaptive_scores)
        analysis['improvement_metrics'] = self._calculate_improvements(baseline_scores, adaptive_scores)
        analysis['confidence_analysis'] = self._analyze_confidence_correlation(baseline_confidences, adaptive_confidences, baseline_scores, adaptive_scores)
        
        return analysis
    
    def _calculate_summary_stats(self, baseline_scores: Dict, adaptive_scores: Dict) -> Dict[str, float]:
        """Calculate overall summary statistics."""
        all_baseline = []
        all_adaptive = []
        
        for deg_type in baseline_scores:
            for severity in baseline_scores[deg_type]:
                all_baseline.extend(baseline_scores[deg_type][severity])
                all_adaptive.extend(adaptive_scores[deg_type][severity])
        
        return {
            'overall_baseline_accuracy': np.mean(all_baseline) if all_baseline else 0.0,
            'overall_adaptive_accuracy': np.mean(all_adaptive) if all_adaptive else 0.0,
            'overall_improvement': np.mean(all_adaptive) - np.mean(all_baseline) if all_baseline and all_adaptive else 0.0,
            'relative_improvement': (np.mean(all_adaptive) / np.mean(all_baseline) - 1) * 100 if all_baseline and np.mean(all_baseline) > 0 else 0.0,
            'total_samples': len(all_baseline)
        }
    
    def _analyze_by_degradation(self, baseline_scores: Dict, adaptive_scores: Dict) -> Dict[str, Dict]:
        """Analyze performance by degradation type and severity."""
        analysis = {}
        
        for deg_type in baseline_scores:
            analysis[deg_type] = {}
            
            for severity in baseline_scores[deg_type]:
                baseline_vals = baseline_scores[deg_type][severity]
                adaptive_vals = adaptive_scores[deg_type][severity]
                
                if baseline_vals and adaptive_vals:
                    analysis[deg_type][severity] = {
                        'baseline_mean': np.mean(baseline_vals),
                        'adaptive_mean': np.mean(adaptive_vals),
                        'improvement': np.mean(adaptive_vals) - np.mean(baseline_vals),
                        'baseline_std': np.std(baseline_vals),
                        'adaptive_std': np.std(adaptive_vals),
                        'sample_count': len(baseline_vals)
                    }
        
        return analysis
    
    def _calculate_improvements(self, baseline_scores: Dict, adaptive_scores: Dict) -> Dict[str, any]:
        """Calculate improvement metrics."""
        improvements = {
            'by_degradation_type': {},
            'by_severity': {},
            'significant_improvements': []
        }
        
        # By degradation type
        for deg_type in baseline_scores:
            all_baseline = []
            all_adaptive = []
            
            for severity in baseline_scores[deg_type]:
                all_baseline.extend(baseline_scores[deg_type][severity])
                all_adaptive.extend(adaptive_scores[deg_type][severity])
            
            if all_baseline and all_adaptive:
                improvement = np.mean(all_adaptive) - np.mean(all_baseline)
                improvements['by_degradation_type'][deg_type] = {
                    'absolute_improvement': improvement,
                    'relative_improvement': (improvement / np.mean(all_baseline)) * 100 if np.mean(all_baseline) > 0 else 0
                }
                
                # Flag significant improvements (>20% relative)
                if improvement / np.mean(all_baseline) > 0.2:
                    improvements['significant_improvements'].append(deg_type)
        
        # By severity
        for severity in self.config.degradation_levels:
            all_baseline = []
            all_adaptive = []
            
            for deg_type in baseline_scores:
                all_baseline.extend(baseline_scores[deg_type][severity])
                all_adaptive.extend(adaptive_scores[deg_type][severity])
            
            if all_baseline and all_adaptive:
                improvement = np.mean(all_adaptive) - np.mean(all_baseline)
                improvements['by_severity'][severity] = {
                    'absolute_improvement': improvement,
                    'relative_improvement': (improvement / np.mean(all_baseline)) * 100 if np.mean(all_baseline) > 0 else 0
                }
        
        return improvements
    
    def _analyze_confidence_correlation(self, baseline_conf: Dict, adaptive_conf: Dict, 
                                      baseline_acc: Dict, adaptive_acc: Dict) -> Dict[str, any]:
        """Analyze correlation between confidence and accuracy."""
        # Flatten data for correlation analysis
        baseline_conf_flat = []
        baseline_acc_flat = []
        adaptive_conf_flat = []
        adaptive_acc_flat = []
        
        for deg_type in baseline_conf:
            for severity in baseline_conf[deg_type]:
                baseline_conf_flat.extend(baseline_conf[deg_type][severity])
                baseline_acc_flat.extend(baseline_acc[deg_type][severity])
                adaptive_conf_flat.extend(adaptive_conf[deg_type][severity])
                adaptive_acc_flat.extend(adaptive_acc[deg_type][severity])
        
        correlations = {}
        
        if len(baseline_conf_flat) > 1:
            correlations['baseline_correlation'] = np.corrcoef(baseline_conf_flat, baseline_acc_flat)[0, 1]
        if len(adaptive_conf_flat) > 1:
            correlations['adaptive_correlation'] = np.corrcoef(adaptive_conf_flat, adaptive_acc_flat)[0, 1]
        
        return correlations
    
    def _generate_robustness_plots(self, analysis: Dict[str, any]):
        """Generate robustness visualization plots."""
        self.logger.info("Generating robustness plots")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('OCR Robustness Evaluation Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Overall improvement by degradation type
        deg_types = list(analysis['improvement_metrics']['by_degradation_type'].keys())
        improvements = [analysis['improvement_metrics']['by_degradation_type'][dt]['relative_improvement'] 
                       for dt in deg_types]
        
        axes[0, 0].bar(deg_types, improvements, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Relative Improvement by Degradation Type')
        axes[0, 0].set_ylabel('Improvement (%)')
        axes[0, 0].set_xlabel('Degradation Type')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Improvement by severity
        severities = list(analysis['improvement_metrics']['by_severity'].keys())
        sev_improvements = [analysis['improvement_metrics']['by_severity'][sv]['relative_improvement'] 
                          for sv in severities]
        
        axes[0, 1].bar(severities, sev_improvements, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Relative Improvement by Severity Level')
        axes[0, 1].set_ylabel('Improvement (%)')
        axes[0, 1].set_xlabel('Severity Level')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Accuracy comparison heatmap
        if 'degradation_analysis' in analysis:
            deg_analysis = analysis['degradation_analysis']
            
            # Create matrices for heatmap
            baseline_matrix = []
            adaptive_matrix = []
            
            for deg_type in deg_types:
                if deg_type in deg_analysis:
                    baseline_row = []
                    adaptive_row = []
                    for severity in severities:
                        if severity in deg_analysis[deg_type]:
                            baseline_row.append(deg_analysis[deg_type][severity]['baseline_mean'])
                            adaptive_row.append(deg_analysis[deg_type][severity]['adaptive_mean'])
                        else:
                            baseline_row.append(0)
                            adaptive_row.append(0)
                    baseline_matrix.append(baseline_row)
                    adaptive_matrix.append(adaptive_row)
            
            if baseline_matrix:
                baseline_df = pd.DataFrame(baseline_matrix, index=deg_types, columns=severities)
                im = axes[1, 0].imshow(baseline_df.values, cmap='Reds', aspect='auto')
                axes[1, 0].set_title('Baseline Accuracy Heatmap')
                axes[1, 0].set_xlabel('Severity')
                axes[1, 0].set_ylabel('Degradation Type')
                axes[1, 0].set_xticks(range(len(severities)))
                axes[1, 0].set_xticklabels(severities)
                axes[1, 0].set_yticks(range(len(deg_types)))
                axes[1, 0].set_yticklabels(deg_types)
                plt.colorbar(im, ax=axes[1, 0])
        
        # Plot 4: Summary statistics
        summary = analysis['summary']
        metrics = ['Overall Baseline', 'Overall Adaptive', 'Improvement']
        values = [summary['overall_baseline_accuracy'], 
                 summary['overall_adaptive_accuracy'],
                 summary['overall_improvement']]
        
        colors = ['lightblue', 'lightgreen', 'gold']
        bars = axes[1, 1].bar(metrics, values, color=colors, alpha=0.7)
        axes[1, 1].set_title('Overall Performance Summary')
        axes[1, 1].set_ylabel('Accuracy / Improvement')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "robustness_evaluation_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Plots saved to {plot_path}")
    
    def generate_report(self, analysis: Dict[str, any]) -> str:
        """Generate a comprehensive text report."""
        report_lines = [
            "# OCR Robustness Evaluation Report",
            "=" * 50,
            "",
            "## Summary",
            f"Overall baseline accuracy: {analysis['summary']['overall_baseline_accuracy']:.3f}",
            f"Overall adaptive accuracy: {analysis['summary']['overall_adaptive_accuracy']:.3f}",
            f"Absolute improvement: {analysis['summary']['overall_improvement']:.3f}",
            f"Relative improvement: {analysis['summary']['relative_improvement']:.1f}%",
            f"Total samples evaluated: {analysis['summary']['total_samples']}",
            "",
            "## Key Findings",
        ]
        
        # Significant improvements
        if analysis['improvement_metrics']['significant_improvements']:
            report_lines.extend([
                "### Significant Improvements (>20% relative):",
                ""
            ])
            for deg_type in analysis['improvement_metrics']['significant_improvements']:
                improvement = analysis['improvement_metrics']['by_degradation_type'][deg_type]['relative_improvement']
                report_lines.append(f"- {deg_type}: {improvement:.1f}% improvement")
            report_lines.append("")
        
        # Best performing degradation types
        if 'by_degradation_type' in analysis['improvement_metrics']:
            improvements = analysis['improvement_metrics']['by_degradation_type']
            sorted_improvements = sorted(improvements.items(), 
                                       key=lambda x: x[1]['relative_improvement'], 
                                       reverse=True)
            
            report_lines.extend([
                "### Top 3 Improvements by Degradation Type:",
                ""
            ])
            
            for i, (deg_type, metrics) in enumerate(sorted_improvements[:3]):
                report_lines.append(f"{i+1}. {deg_type}: {metrics['relative_improvement']:.1f}% improvement")
            report_lines.append("")
        
        # Confidence analysis
        if 'baseline_correlation' in analysis.get('confidence_analysis', {}):
            baseline_corr = analysis['confidence_analysis']['baseline_correlation']
            adaptive_corr = analysis['confidence_analysis'].get('adaptive_correlation', 0)
            
            report_lines.extend([
                "## Confidence-Accuracy Correlation",
                f"Baseline correlation: {baseline_corr:.3f}",
                f"Adaptive correlation: {adaptive_corr:.3f}",
                ""
            ])
        
        report = "\n".join(report_lines)
        
        # Save report
        report_path = self.output_dir / "robustness_evaluation_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Report saved to {report_path}")
        return report


# Example usage
if __name__ == "__main__":
    # Initialize evaluator
    config = EvaluationConfig(
        degradation_levels=['low', 'medium', 'high'],
        num_variants_per_level=2,  # Reduced for testing
        generate_plots=True,
        save_degraded_images=False
    )
    
    evaluator = RobustnessEvaluator(config)
    
    # Test images
    test_images = []
    for img_name in ["DEWA.png", "SEWA.png"]:
        if Path(img_name).exists():
            test_images.append(img_name)
    
    if test_images:
        print(f"Running robustness evaluation on {len(test_images)} images")
        print(f"Testing {len(config.degradation_types)} degradation types at {len(config.degradation_levels)} severity levels")
        print(f"Total test conditions: {len(test_images) * len(config.degradation_types) * len(config.degradation_levels) * config.num_variants_per_level}")
        
        # Run evaluation
        analysis = evaluator.run_comprehensive_evaluation(test_images)
        
        # Generate report
        report = evaluator.generate_report(analysis)
        print("\n" + "="*50)
        print("ROBUSTNESS EVALUATION REPORT")
        print("="*50)
        print(report)
        
    else:
        print("No test images found (DEWA.png, SEWA.png)")
        print("Please ensure test images are available in the current directory")