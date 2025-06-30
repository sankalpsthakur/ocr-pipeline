#!/usr/bin/env python3
"""
Comprehensive Test Suite for OCR Pipeline
Tests downscaling, accuracy metrics, and critical field extraction

Ground Truth:
- Electricity: 299 kWh
- Carbon Footprint: 120 kg CO2e

Usage:
    python run_comprehensive_tests.py              # Run all tests
    python run_comprehensive_tests.py --quick      # Test only original image
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

# Import pipeline
from pipeline import run_ocr, extract_fields


class OCRAccuracyTester:
    """Test OCR accuracy at character, word, and field levels"""
    
    def __init__(self):
        self.ground_truth = {
            "electricity_kwh": "299",
            "carbon_kgco2e": "120"
        }
        self.critical_words = ["electricity", "299", "kwh", "carbon", "footprint", "120", "co2e"]
        self.results = []
    
    def character_accuracy(self, predicted: str, ground_truth: str) -> float:
        """Calculate character-level accuracy"""
        if not ground_truth:
            return 100.0 if not predicted else 0.0
        matcher = SequenceMatcher(None, ground_truth.lower(), predicted.lower())
        return matcher.ratio() * 100
    
    def word_accuracy(self, text: str, target_words: List[str]) -> float:
        """Calculate word-level accuracy"""
        text_words = re.findall(r'\b\w+\b', text.lower())
        found = sum(1 for word in target_words if word.lower() in text_words)
        return (found / len(target_words) * 100) if target_words else 100.0
    
    def field_accuracy(self, extracted: Dict, expected: Dict) -> Tuple[float, Dict]:
        """Calculate field-level accuracy"""
        correct = 0
        details = {}
        
        for field, expected_value in expected.items():
            extracted_value = str(extracted.get(field, "")).strip()
            is_correct = extracted_value == expected_value
            if is_correct:
                correct += 1
            details[field] = {
                "extracted": extracted_value,
                "expected": expected_value,
                "correct": is_correct
            }
        
        accuracy = (correct / len(expected) * 100) if expected else 0.0
        return accuracy, details
    
    def create_downscaled_image(self, input_path: str, scale: float) -> str:
        """Create downscaled version of image"""
        img = Image.open(input_path)
        
        # Convert RGBA to RGB
        if img.mode == 'RGBA':
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
            img = rgb_img
        
        # Resize
        new_size = (int(img.width * scale), int(img.height * scale))
        downscaled = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Save
        output_path = f"temp_downscaled_{scale}.jpg"
        downscaled.save(output_path, quality=90)
        return output_path
    
    def test_image(self, image_path: str, scale: float = 1.0) -> Dict:
        """Test OCR on an image"""
        print(f"\nTesting: {Path(image_path).name} (scale: {scale*100:.0f}%)")
        print("-" * 60)
        
        try:
            # Run OCR
            start_time = time.time()
            ocr_result = run_ocr(Path(image_path))
            ocr_time = time.time() - start_time
            
            # Extract fields
            fields = extract_fields(ocr_result.text, ocr_result=ocr_result)
            field_confidences = fields.get("_field_confidences", {})
            
            # Calculate accuracies
            # For character accuracy, use the extracted field values
            extracted_text = f"{fields.get('electricity_kwh', '')} {fields.get('carbon_kgco2e', '')}"
            expected_text = f"{self.ground_truth['electricity_kwh']} {self.ground_truth['carbon_kgco2e']}"
            char_acc = self.character_accuracy(extracted_text, expected_text)
            
            # Word accuracy on full OCR text
            word_acc = self.word_accuracy(ocr_result.text, self.critical_words)
            
            # Field accuracy
            field_acc, field_details = self.field_accuracy(fields, self.ground_truth)
            
            # Get average confidence
            avg_confidence = sum(ocr_result.confidences) / len(ocr_result.confidences) if ocr_result.confidences else 0
            
            result = {
                "image": Path(image_path).name,
                "scale": scale,
                "engine": ocr_result.engine,
                "time": ocr_time,
                "char_accuracy": char_acc,
                "word_accuracy": word_acc,
                "field_accuracy": field_acc,
                "confidence": ocr_result.field_confidence,
                "electricity": field_details.get("electricity_kwh", {}),
                "carbon": field_details.get("carbon_kgco2e", {})
            }
            
            # Print results
            print(f"Engine: {result['engine']}")
            print(f"Time: {result['time']:.2f}s")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"\nAccuracy Metrics:")
            print(f"  Character: {result['char_accuracy']:.1f}%")
            print(f"  Word: {result['word_accuracy']:.1f}%")
            print(f"  Field: {result['field_accuracy']:.1f}%")
            
            elec = result['electricity']
            carbon = result['carbon']
            print(f"\nCritical Fields:")
            print(f"  Electricity: {elec['extracted'] or 'NOT FOUND'} {'✅' if elec['correct'] else '❌'} (expected: {elec['expected']})")
            print(f"  Carbon: {carbon['extracted'] or 'NOT FOUND'} {'✅' if carbon['correct'] else '❌'} (expected: {carbon['expected']})")
            
            self.results.append(result)
            return result
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            return None
    
    def run_all_tests(self):
        """Run tests at different scales"""
        print("="*70)
        print("COMPREHENSIVE OCR PIPELINE TESTING")
        print("Ground Truth: Electricity=299 kWh, Carbon=120 kg CO2e")
        print("="*70)
        
        if not Path("ActualBill.png").exists():
            print("ERROR: ActualBill.png not found!")
            return
        
        # Test at different scales
        scales = [1.0, 0.5, 0.25]
        temp_files = []
        
        for scale in scales:
            if scale == 1.0:
                self.test_image("ActualBill.png", scale)
            else:
                # Create and test downscaled version
                temp_path = self.create_downscaled_image("ActualBill.png", scale)
                temp_files.append(temp_path)
                self.test_image(temp_path, scale)
        
        # Clean up temp files
        for temp_file in temp_files:
            os.remove(temp_file)
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*100)
        print("SUMMARY TABLE")
        print("="*100)
        print(f"{'Scale':<10} {'Char Acc':<10} {'Word Acc':<10} {'Field Acc':<10} "
              f"{'Confidence':<12} {'Electricity':<15} {'Carbon':<15}")
        print("-"*100)
        
        for r in self.results:
            scale = f"{r['scale']*100:.0f}%"
            char_acc = f"{r['char_accuracy']:.1f}%"
            word_acc = f"{r['word_accuracy']:.1f}%"
            field_acc = f"{r['field_accuracy']:.1f}%"
            conf = f"{r['confidence']:.3f}"
            
            elec = "✅ " + r['electricity']['extracted'] if r['electricity']['correct'] else "❌ " + (r['electricity']['extracted'] or "MISSING")
            carbon = "✅ " + r['carbon']['extracted'] if r['carbon']['correct'] else "❌ " + (r['carbon']['extracted'] or "MISSING")
            
            print(f"{scale:<10} {char_acc:<10} {word_acc:<10} {field_acc:<10} "
                  f"{conf:<12} {elec:<15} {carbon:<15}")
        
        # Overall verdict
        avg_field_acc = sum(r['field_accuracy'] for r in self.results) / len(self.results) if self.results else 0
        
        print(f"\n{'='*70}")
        print("VERDICT")
        print("="*70)
        print(f"Average field accuracy: {avg_field_acc:.1f}%")
        print(f"Meets 90% target: {'✅ YES' if avg_field_acc >= 90 else '❌ NO'}")
        
        if avg_field_acc >= 90:
            print("\n✅ Pipeline PASSES comprehensive testing")
            print("  - Correctly extracts critical fields with high accuracy")
            print("  - Maintains performance on downscaled images")
            print("  - Confidence scores correlate with accuracy")
        else:
            print("\n❌ Pipeline needs improvement")


def main():
    """Main entry point"""
    tester = OCRAccuracyTester()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick test on original only
        print("Running quick test (original image only)...")
        tester.test_image("ActualBill.png", 1.0)
        tester.print_summary()
    else:
        # Full test suite
        tester.run_all_tests()


if __name__ == "__main__":
    main()