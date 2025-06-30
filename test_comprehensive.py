#!/usr/bin/env python3
"""
Comprehensive Test Suite for OCR Pipeline
Includes downscaling tests and character/word/field accuracy metrics
Focus on electricity (299 kWh) and carbon footprint (120 kg CO2e) fields
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Tuple
from difflib import SequenceMatcher
from PIL import Image
import numpy as np

# Import pipeline functions
from pipeline import run_ocr, extract_fields

class AccuracyMetrics:
    """Calculate character, word, and field level accuracy"""
    
    @staticmethod
    def character_accuracy(predicted: str, ground_truth: str) -> float:
        """Calculate character-level accuracy using sequence matching"""
        if not ground_truth:
            return 100.0 if not predicted else 0.0
        
        matcher = SequenceMatcher(None, ground_truth.lower(), predicted.lower())
        return matcher.ratio() * 100
    
    @staticmethod
    def word_accuracy(predicted: str, ground_truth_words: List[str]) -> Tuple[float, List[str]]:
        """Calculate word-level accuracy"""
        pred_words = re.findall(r'\b\w+\b', predicted.lower())
        gt_words_lower = [w.lower() for w in ground_truth_words]
        
        correct = sum(1 for word in gt_words_lower if word in pred_words)
        accuracy = (correct / len(ground_truth_words) * 100) if ground_truth_words else 100.0
        
        missing = [w for w in gt_words_lower if w not in pred_words]
        return accuracy, missing
    
    @staticmethod
    def field_accuracy(extracted_fields: Dict, ground_truth_fields: Dict) -> Tuple[float, Dict]:
        """Calculate field-level accuracy"""
        correct = 0
        total = len(ground_truth_fields)
        details = {}
        
        for field, expected in ground_truth_fields.items():
            extracted = str(extracted_fields.get(field, "")).strip()
            expected = str(expected).strip()
            
            is_correct = extracted == expected
            if is_correct:
                correct += 1
            
            details[field] = {
                "expected": expected,
                "extracted": extracted,
                "correct": is_correct
            }
        
        accuracy = (correct / total * 100) if total > 0 else 0.0
        return accuracy, details

class ComprehensiveOCRTester:
    """Comprehensive testing including downscaling and accuracy metrics"""
    
    def __init__(self):
        # Ground truth for DEWA bill
        self.ground_truth = {
            "electricity_kwh": "299",
            "carbon_kgco2e": "120"
        }
        
        # Critical words to check
        self.critical_words = ["electricity", "299", "kwh", "carbon", "footprint", "120", "co2e"]
        
        # Test results storage
        self.results = []
    
    def create_downscaled_image(self, input_path: str, scale: float) -> str:
        """Create a downscaled version of the image"""
        img = Image.open(input_path)
        
        # Convert RGBA to RGB for JPEG
        if img.mode == 'RGBA':
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
            img = rgb_img
        
        # Calculate new dimensions
        new_width = int(img.width * scale)
        new_height = int(img.height * scale)
        
        # Resize
        downscaled = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Save
        output_path = f"test_downscaled_{scale}.jpg"
        downscaled.save(output_path, quality=90)
        
        return output_path
    
    def test_image(self, image_path: str, scale: float = 1.0) -> Dict:
        """Test OCR on a single image with accuracy metrics"""
        print(f"\nTesting: {image_path} (scale: {scale*100}%)")
        print("-" * 60)
        
        # Run OCR
        start_time = time.time()
        ocr_result = run_ocr(Path(image_path))
        ocr_time = time.time() - start_time
        
        # Extract fields
        fields = extract_fields(ocr_result.text, ocr_result=ocr_result)
        field_confidences = fields.get("_field_confidences", {})
        
        # Calculate accuracy metrics
        char_acc = AccuracyMetrics.character_accuracy(
            " ".join([fields.get("electricity_kwh", ""), fields.get("carbon_kgco2e", "")]),
            " ".join(self.ground_truth.values())
        )
        
        word_acc, missing_words = AccuracyMetrics.word_accuracy(
            ocr_result.text,
            self.critical_words
        )
        
        field_acc, field_details = AccuracyMetrics.field_accuracy(
            fields,
            self.ground_truth
        )
        
        # Calculate average confidence
        avg_confidence = sum(ocr_result.confidences) / len(ocr_result.confidences) if ocr_result.confidences else 0
        
        # Prepare result
        result = {
            "image": image_path,
            "scale": scale,
            "resolution": f"{int(img.width*scale)}x{int(img.height*scale)}" if 'img' in locals() else "unknown",
            "ocr_engine": ocr_result.engine,
            "ocr_time": ocr_time,
            "character_accuracy": char_acc,
            "word_accuracy": word_acc,
            "field_accuracy": field_acc,
            "overall_confidence": avg_confidence,
            "field_confidence": ocr_result.field_confidence,
            "electricity_kwh": {
                "extracted": fields.get("electricity_kwh", ""),
                "expected": self.ground_truth["electricity_kwh"],
                "correct": fields.get("electricity_kwh") == self.ground_truth["electricity_kwh"],
                "confidence": field_confidences.get("electricity_kwh", 0)
            },
            "carbon_kgco2e": {
                "extracted": fields.get("carbon_kgco2e", ""),
                "expected": self.ground_truth["carbon_kgco2e"],
                "correct": fields.get("carbon_kgco2e") == self.ground_truth["carbon_kgco2e"],
                "confidence": field_confidences.get("carbon_kgco2e", 0)
            },
            "missing_words": missing_words,
            "field_details": field_details
        }
        
        # Print results
        print(f"OCR Engine: {result['ocr_engine']}")
        print(f"Processing Time: {result['ocr_time']:.2f}s")
        print(f"Overall Confidence: {result['overall_confidence']:.2%}")
        print(f"Field Confidence: {result['field_confidence']:.2%}")
        
        print(f"\nAccuracy Metrics:")
        print(f"  Character Accuracy: {result['character_accuracy']:.1f}%")
        print(f"  Word Accuracy: {result['word_accuracy']:.1f}%")
        print(f"  Field Accuracy: {result['field_accuracy']:.1f}%")
        
        print(f"\nCritical Fields:")
        elec = result['electricity_kwh']
        print(f"  Electricity: {elec['extracted']} {'✅' if elec['correct'] else '❌'} "
              f"(expected: {elec['expected']}, conf: {elec['confidence']:.2f})")
        
        carbon = result['carbon_kgco2e']
        print(f"  Carbon: {carbon['extracted']} {'✅' if carbon['correct'] else '❌'} "
              f"(expected: {carbon['expected']}, conf: {carbon['confidence']:.2f})")
        
        if missing_words:
            print(f"\nMissing words: {', '.join(missing_words)}")
        
        self.results.append(result)
        return result
    
    def run_comprehensive_tests(self):
        """Run tests on original and downscaled images"""
        print("="*70)
        print("COMPREHENSIVE OCR PIPELINE TESTING")
        print("="*70)
        
        # Test scales
        scales = [1.0, 0.5, 0.25]
        
        # Original image
        original_image = "ActualBill.png"
        
        if not Path(original_image).exists():
            print(f"Error: {original_image} not found!")
            return
        
        # Test each scale
        for scale in scales:
            if scale == 1.0:
                self.test_image(original_image, scale)
            else:
                # Create downscaled version
                downscaled_path = self.create_downscaled_image(original_image, scale)
                self.test_image(downscaled_path, scale)
                # Clean up
                os.remove(downscaled_path)
        
        # Generate summary report
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*70)
        print("SUMMARY REPORT")
        print("="*70)
        
        # Overall statistics
        total_tests = len(self.results)
        electricity_correct = sum(1 for r in self.results if r['electricity_kwh']['correct'])
        carbon_correct = sum(1 for r in self.results if r['carbon_kgco2e']['correct'])
        
        print(f"\nTotal tests run: {total_tests}")
        print(f"Electricity field accuracy: {electricity_correct}/{total_tests} ({electricity_correct/total_tests*100:.0f}%)")
        print(f"Carbon field accuracy: {carbon_correct}/{total_tests} ({carbon_correct/total_tests*100:.0f}%)")
        
        # Detailed table
        print("\n" + "="*100)
        print(f"{'Scale':<10} {'Char Acc':<10} {'Word Acc':<10} {'Field Acc':<10} {'Confidence':<12} "
              f"{'Electricity':<15} {'Carbon':<15} {'Time':<8}")
        print("="*100)
        
        for result in self.results:
            scale_str = f"{result['scale']*100:.0f}%"
            char_acc = f"{result['character_accuracy']:.1f}%"
            word_acc = f"{result['word_accuracy']:.1f}%"
            field_acc = f"{result['field_accuracy']:.1f}%"
            conf = f"{result['field_confidence']:.3f}"
            
            elec_status = "✅ " + result['electricity_kwh']['extracted'] if result['electricity_kwh']['correct'] else "❌ " + (result['electricity_kwh']['extracted'] or "NOT FOUND")
            carbon_status = "✅ " + result['carbon_kgco2e']['extracted'] if result['carbon_kgco2e']['correct'] else "❌ " + (result['carbon_kgco2e']['extracted'] or "NOT FOUND")
            
            time_str = f"{result['ocr_time']:.2f}s"
            
            print(f"{scale_str:<10} {char_acc:<10} {word_acc:<10} {field_acc:<10} {conf:<12} "
                  f"{elec_status:<15} {carbon_status:<15} {time_str:<8}")
        
        # Key findings
        print("\n" + "="*70)
        print("KEY FINDINGS")
        print("="*70)
        
        # Check 90% accuracy target
        field_accuracies = [r['field_accuracy'] for r in self.results]
        avg_field_accuracy = sum(field_accuracies) / len(field_accuracies)
        
        print(f"\n1. Average field accuracy: {avg_field_accuracy:.1f}%")
        print(f"   {'✅' if avg_field_accuracy >= 90 else '❌'} Meets 90% accuracy target: {'YES' if avg_field_accuracy >= 90 else 'NO'}")
        
        # Confidence correlation
        print(f"\n2. Confidence vs Accuracy Correlation:")
        for result in self.results:
            scale = f"{result['scale']*100:.0f}%"
            conf = result['field_confidence']
            acc = result['field_accuracy']
            print(f"   Scale {scale}: Confidence={conf:.3f}, Accuracy={acc:.0f}%")
        
        # Performance at different scales
        print(f"\n3. Performance vs Image Quality:")
        print(f"   - Original (100%): Maintains {self.results[0]['field_accuracy']:.0f}% field accuracy")
        if len(self.results) > 1:
            print(f"   - Downscaled 50%: Maintains {self.results[1]['field_accuracy']:.0f}% field accuracy")
        if len(self.results) > 2:
            print(f"   - Downscaled 25%: Drops to {self.results[2]['field_accuracy']:.0f}% field accuracy")
        
        # Save detailed results
        with open("comprehensive_test_results.json", "w") as f:
            json.dump({
                "summary": {
                    "total_tests": total_tests,
                    "electricity_accuracy": electricity_correct/total_tests*100,
                    "carbon_accuracy": carbon_correct/total_tests*100,
                    "average_field_accuracy": avg_field_accuracy,
                    "meets_90_percent_target": avg_field_accuracy >= 90
                },
                "detailed_results": self.results
            }, f, indent=2)
        
        print(f"\nDetailed results saved to: comprehensive_test_results.json")
        
        # Final verdict
        print("\n" + "="*70)
        print("FINAL VERDICT")
        print("="*70)
        
        if avg_field_accuracy >= 90 and electricity_correct >= total_tests - 1:
            print("✅ Pipeline PASSES comprehensive testing")
            print("   - Achieves 90%+ accuracy on critical fields")
            print("   - Correctly extracts electricity consumption (299 kWh)")
            print("   - Correctly extracts carbon footprint (120 kg CO2e)")
            print("   - Maintains high accuracy down to 50% image scale")
        else:
            print("❌ Pipeline needs improvement")
            print(f"   - Current accuracy: {avg_field_accuracy:.1f}%")
            print(f"   - Target accuracy: 90%")


def main():
    """Run comprehensive tests"""
    tester = ComprehensiveOCRTester()
    tester.run_comprehensive_tests()


if __name__ == "__main__":
    main()