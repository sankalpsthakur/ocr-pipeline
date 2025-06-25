#!/usr/bin/env python3
"""
Test script for the hierarchical OCR implementation.
"""

import sys
from pathlib import Path
import pipeline
from PIL import Image, ImageDraw, ImageFont

def create_test_image():
    """Create a simple test image with text."""
    # Create a white image
    img = Image.new('RGB', (800, 400), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a font, fallback to default
    try:
        font = ImageFont.truetype("Arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    # Add some text
    text = "Electricity Consumption: 299 kWh\nCarbon Footprint: 120 kgCO2e"
    draw.text((50, 100), text, fill='black', font=font)
    
    return img

def test_ocr_hierarchy():
    """Test the OCR hierarchy."""
    print("Testing OCR Hierarchy Implementation")
    print("=" * 50)
    
    # Create test image
    test_img = create_test_image()
    test_path = Path("test_bill.png")
    test_img.save(test_path)
    
    try:
        # Test the hierarchical OCR
        print(f"Running hierarchical OCR on {test_path}")
        result = pipeline.run_ocr(test_path)
        
        print(f"OCR Result:")
        print(f"  Text: {result.text}")
        print(f"  Confidence: {result.field_confidence:.3f}")
        print(f"  Tokens: {len(result.tokens)}")
        
        # Test field extraction
        fields = pipeline.extract_fields(result.text)
        print(f"\nExtracted Fields:")
        for key, value in fields.items():
            if not key.startswith('_'):
                print(f"  {key}: {value}")
        
        # Test payload building
        payload = pipeline.build_payload(fields, test_path)
        print(f"\nPayload Structure:")
        print(f"  Electricity: {payload.get('electricity', {}).get('consumption', {}).get('value', 'N/A')}")
        print(f"  Carbon: {payload.get('carbon', {}).get('location_based', {}).get('value', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        return False
    finally:
        # Clean up
        if test_path.exists():
            test_path.unlink()

if __name__ == "__main__":
    success = test_ocr_hierarchy()
    sys.exit(0 if success else 1)