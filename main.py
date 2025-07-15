#!/usr/bin/env python3
"""FastAPI Web Service for OCR Pipeline

Simple REST API for utility bill OCR processing with Railway deployment support.
"""

import os
import tempfile
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import sys

# Add compatibility shim for imghdr (removed in Python 3.13)
try:
    import imghdr
except ImportError:
    import imghdr_compat as imghdr
    sys.modules['imghdr'] = imghdr

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Import available OCR components
PIPELINE_AVAILABLE = False
MOBILE_PIPELINE_AVAILABLE = False
PADDLEOCR_AVAILABLE = False

# Try PyTorch mobile pipeline first (best performance)
try:
    import sys
    sys.path.append('./pytorch_mobile')
    from pytorch_mobile.ocr_pipeline import run_ocr_with_utility_schema
    PYTORCH_MOBILE_AVAILABLE = True
    logging.info("PyTorch mobile pipeline loaded successfully")
except ImportError as e:
    logging.warning(f"PyTorch mobile pipeline import failed: {e}")
    PYTORCH_MOBILE_AVAILABLE = False

# Try main pipeline basic functions (without utility bill payload)
try:
    from pipeline import run_ocr, extract_fields
    PIPELINE_AVAILABLE = True
    logging.info("Main pipeline (basic functions) loaded successfully")
except ImportError as e:
    logging.warning(f"Main pipeline import failed: {e}")

# Try PaddleOCR directly
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
    logging.info("PaddleOCR loaded successfully")
except ImportError as e:
    logging.warning(f"PaddleOCR import failed: {e}")

# Try Tesseract as fallback
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    logging.info("Tesseract loaded successfully") 
except ImportError as e:
    logging.warning(f"Tesseract import failed: {e}")
    TESSERACT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="OCR Pipeline API",
    description="Utility bill OCR processing with field extraction",
    version="1.0.0"
)

# Railway port configuration
PORT = int(os.environ.get("PORT", 8000))

# Initialize OCR engines
ocr_engine = None
if PADDLEOCR_AVAILABLE:
    try:
        ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        logger.info("PaddleOCR engine initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize PaddleOCR: {e}")

def simple_extract_fields(text: str) -> Dict[str, Any]:
    """Simple field extraction for utility bills using regex patterns."""
    fields = {}
    
    # Electricity patterns
    electricity_patterns = [
        r"(?:electricity|kilowatt|kwh)[:\s]*(\d+(?:\.\d+)?)\s*(?:kwh)?",
        r"total\s*consumption[:\s]*(\d+(?:\.\d+)?)\s*kwh",
        r"(\d+(?:\.\d+)?)\s*kwh"
    ]
    
    # Carbon footprint patterns  
    carbon_patterns = [
        r"carbon\s*footprint[:\s]*(\d+(?:\.\d+)?)\s*(?:kg\s*co2e?)?",
        r"(\d+(?:\.\d+)?)\s*kg\s*co2e?",
        r"emissions[:\s]*(\d+(?:\.\d+)?)\s*kg"
    ]
    
    # Water patterns
    water_patterns = [
        r"water[:\s]*(\d+(?:\.\d+)?)\s*(?:m3|cubic|liters?|gallons?)",
        r"(\d+(?:\.\d+)?)\s*(?:m3|cubic\s*meters?)"
    ]
    
    # Extract fields using patterns
    text_lower = text.lower()
    
    for pattern in electricity_patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match and not fields.get('electricity_kwh'):
            value = float(match.group(1))
            if 1 <= value <= 10000:  # Reasonable range
                fields['electricity_kwh'] = str(int(value))
                break
    
    for pattern in carbon_patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE) 
        if match and not fields.get('carbon_kgco2e'):
            value = float(match.group(1))
            if 1 <= value <= 10000:  # Reasonable range
                fields['carbon_kgco2e'] = str(int(value))
                break
                
    for pattern in water_patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match and not fields.get('water_m3'):
            value = float(match.group(1))
            if 0.1 <= value <= 10000:  # Reasonable range
                fields['water_m3'] = str(value)
                break
    
    return fields

def simple_run_ocr(image_path: Path) -> Dict[str, Any]:
    """Simple OCR using available engines."""
    if not ocr_engine and not TESSERACT_AVAILABLE:
        raise Exception("No OCR engine available")
    
    try:
        if ocr_engine:
            # Use PaddleOCR
            result = ocr_engine.ocr(str(image_path), cls=True)
            
            # Extract text from PaddleOCR result
            text_lines = []
            confidence_scores = []
            
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        text_lines.append(line[1][0])
                        confidence_scores.append(line[1][1])
            
            full_text = ' '.join(text_lines)
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            
            return {
                'text': full_text,
                'confidence': avg_confidence,
                'engine': 'paddleocr'
            }
            
        elif TESSERACT_AVAILABLE:
            # Use Tesseract
            from PIL import Image
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            
            return {
                'text': text,
                'confidence': 0.8,  # Default confidence for Tesseract
                'engine': 'tesseract'
            }
            
    except Exception as e:
        raise Exception(f"OCR processing failed: {e}")

def build_simple_utility_bill_payload(fields: Dict[str, Any], image_path: Path) -> Dict[str, Any]:
    """Build a simple utility bill payload."""
    return {
        "success": True,
        "document_type": "utility_bill",
        "extracted_fields": fields,
        "validation": {
            "confidence": 0.85,  # Conservative estimate
            "fields_extracted": len(fields),
            "processing_engine": "simplified_ocr"
        },
        "metadata": {
            "file_name": image_path.name,
            "processing_time": "< 1s",
            "extraction_method": "regex_patterns"
        }
    }

@app.get("/")
async def root():
    """Root endpoint with API documentation."""
    return HTMLResponse(content="""
    <html>
        <head>
            <title>OCR Pipeline API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
                .status { color: green; font-weight: bold; }
                .warning { color: orange; }
            </style>
        </head>
        <body>
            <h1>OCR Pipeline API</h1>
            <p class="status">✅ Service is running</p>
            
            <h2>Available Endpoints:</h2>
            
            <div class="endpoint">
                <h3>POST /ocr</h3>
                <p>Upload an image file for OCR processing</p>
                <p><strong>Parameters:</strong></p>
                <ul>
                    <li><code>file</code>: Image file (PNG, JPEG, PDF)</li>
                    <li><code>format</code>: Output format ("utility_bill" or "basic")</li>
                    <li><code>model</code>: OCR model ("pytorch_mobile", "main_pipeline", "simplified_ocr", "auto")</li>
                </ul>
            </div>
            
            <div class="endpoint">
                <h3>GET /health</h3>
                <p>Health check endpoint</p>
            </div>
            
            <div class="endpoint">
                <h3>GET /status</h3>
                <p>System status and capabilities</p>
            </div>
            
            <h2>System Status:</h2>
            <ul>
                <li>PyTorch Mobile: """ + ("✅ Available" if PYTORCH_MOBILE_AVAILABLE else "❌ Not Available") + """</li>
                <li>Main Pipeline: """ + ("✅ Available" if PIPELINE_AVAILABLE else "❌ Not Available") + """</li>
                <li>PaddleOCR: """ + ("✅ Available" if PADDLEOCR_AVAILABLE else "❌ Not Available") + """</li>
                <li>Tesseract: """ + ("✅ Available" if TESSERACT_AVAILABLE else "❌ Not Available") + """</li>
            </ul>
            
            <h2>Usage Examples:</h2>
            <pre>
# Auto model selection (recommended)
curl -X POST "http://localhost:8000/ocr" \\
     -F "file=@bill.png" \\
     -F "format=utility_bill" \\
     -F "model=auto"

# Force PyTorch mobile (production-ready)
curl -X POST "http://localhost:8000/ocr" \\
     -F "file=@bill.png" \\
     -F "format=utility_bill" \\
     -F "model=pytorch_mobile"
            </pre>
        </body>
    </html>
    """)

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway."""
    return {"status": "healthy", "service": "ocr-pipeline"}

@app.get("/status")
async def get_status():
    """System status and capabilities."""
    return {
        "service": "OCR Pipeline API",
        "version": "1.0.0",
        "capabilities": {
            "pytorch_mobile": PYTORCH_MOBILE_AVAILABLE,
            "main_pipeline": PIPELINE_AVAILABLE,
            "paddleocr": PADDLEOCR_AVAILABLE,
            "tesseract": TESSERACT_AVAILABLE,
            "ocr_engine_active": ocr_engine is not None or PYTORCH_MOBILE_AVAILABLE,
            "supported_formats": ["PNG", "JPEG", "PDF"],
            "output_formats": ["utility_bill", "basic"],
            "available_models": ["pytorch_mobile", "main_pipeline", "simplified_ocr", "auto"],
            "recommended_model": "pytorch_mobile" if PYTORCH_MOBILE_AVAILABLE else "auto"
        },
        "environment": {
            "port": PORT,
            "python_version": "3.13+",
            "platform": "Railway"
        }
    }

@app.post("/ocr")
async def process_ocr(
    file: UploadFile = File(...),
    format: str = Form(default="utility_bill"),
    model: str = Form(default="auto")
):
    """
    Process uploaded image with OCR and extract fields.
    
    Args:
        file: Uploaded image file (PNG, JPEG, PDF)
        format: Output format ("utility_bill" or "basic")
        model: OCR model to use ("pytorch_mobile", "main_pipeline", "simplified_ocr", "auto")
    
    Returns:
        JSON with extracted fields and metadata
    """
    
    # Validate file type
    if not file.content_type or not any(
        file.content_type.startswith(mime) 
        for mime in ["image/", "application/pdf"]
    ):
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file type. Please upload PNG, JPEG, or PDF."
        )
    
    # Validate model parameter
    valid_models = ["pytorch_mobile", "main_pipeline", "simplified_ocr", "auto"]
    if model not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{model}'. Must be one of: {', '.join(valid_models)}"
        )
    
    # Check if requested model is available
    if model == "pytorch_mobile" and not PYTORCH_MOBILE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="PyTorch mobile pipeline is not available. Try 'auto' for fallback."
        )
    elif model == "main_pipeline" and not PIPELINE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Main pipeline is not available. Try 'auto' for fallback."
        )
    elif model == "simplified_ocr" and not (PADDLEOCR_AVAILABLE or TESSERACT_AVAILABLE):
        raise HTTPException(
            status_code=503,
            detail="Simplified OCR engines are not available. Try 'auto' for fallback."
        )
    
    # Check if any OCR engine is available for auto mode
    if model == "auto" and not PYTORCH_MOBILE_AVAILABLE and not PIPELINE_AVAILABLE and not PADDLEOCR_AVAILABLE and not TESSERACT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="OCR services are currently unavailable. Please try again later."
        )
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
            # Write uploaded content
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = Path(tmp_file.name)
        
        logger.info(f"Processing file: {file.filename} ({len(content)} bytes)")
        
        # Model selection logic based on user preference
        pipeline_used = "unknown"
        
        # Force specific model if requested
        if model == "pytorch_mobile":
            # Use PyTorch mobile pipeline (production-ready)
            if format == "utility_bill":
                result = run_ocr_with_utility_schema(tmp_file_path)
                pipeline_used = "pytorch_mobile"
            else:
                # For basic format, extract text from the utility schema result
                utility_result = run_ocr_with_utility_schema(tmp_file_path)
                result = {
                    "success": True,
                    "extracted_fields": utility_result.get("extractedData", {}).get("consumptionData", {}),
                    "raw_text": f"Provider: {utility_result.get('extractedData', {}).get('billInfo', {}).get('providerName', '')}, "
                               f"Account: {utility_result.get('extractedData', {}).get('billInfo', {}).get('accountNumber', '')}, "
                               f"Electricity: {utility_result.get('extractedData', {}).get('consumptionData', {}).get('electricity', {}).get('value', 0)} kWh",
                    "confidence": utility_result.get("validation", {}).get("confidence", 0.0)
                }
                pipeline_used = "pytorch_mobile_basic"
        
        elif model == "main_pipeline":
            # Use main pipeline
            ocr_result = run_ocr(tmp_file_path)
            fields = extract_fields(ocr_result.text)
            
            if format == "utility_bill":
                result = build_simple_utility_bill_payload(fields, tmp_file_path)
            else:
                result = {
                    "success": True,
                    "extracted_fields": fields,
                    "raw_text": ocr_result.text,
                    "confidence": getattr(ocr_result, 'confidence', 0.0)
                }
            
            pipeline_used = "main"
        
        elif model == "simplified_ocr":
            # Use simplified OCR pipeline
            ocr_result = simple_run_ocr(tmp_file_path)
            fields = simple_extract_fields(ocr_result['text'])
            
            if format == "utility_bill":
                result = build_simple_utility_bill_payload(fields, tmp_file_path)
            else:
                result = {
                    "success": True,
                    "extracted_fields": fields,
                    "raw_text": ocr_result['text'],
                    "confidence": ocr_result['confidence']
                }
            
            pipeline_used = f"simplified_{ocr_result['engine']}"
        
        elif model == "auto" and PYTORCH_MOBILE_AVAILABLE:
            try:
                # Use PyTorch mobile pipeline (production-ready)
                if format == "utility_bill":
                    result = run_ocr_with_utility_schema(tmp_file_path)
                    pipeline_used = "pytorch_mobile"
                else:
                    # For basic format, extract text from the utility schema result
                    utility_result = run_ocr_with_utility_schema(tmp_file_path)
                    result = {
                        "success": True,
                        "extracted_fields": utility_result.get("extractedData", {}).get("consumptionData", {}),
                        "raw_text": f"Provider: {utility_result.get('extractedData', {}).get('billInfo', {}).get('providerName', '')}, "
                                   f"Account: {utility_result.get('extractedData', {}).get('billInfo', {}).get('accountNumber', '')}, "
                                   f"Electricity: {utility_result.get('extractedData', {}).get('consumptionData', {}).get('electricity', {}).get('value', 0)} kWh",
                        "confidence": utility_result.get("validation", {}).get("confidence", 0.0)
                    }
                    pipeline_used = "pytorch_mobile_basic"
                    
            except Exception as e:
                logger.warning(f"PyTorch mobile pipeline failed: {e}, falling back to main pipeline")
                # Fall back to main pipeline
                if PIPELINE_AVAILABLE:
                    try:
                        # Use main pipeline
                        ocr_result = run_ocr(tmp_file_path)
                        fields = extract_fields(ocr_result.text)
                        
                        if format == "utility_bill":
                            result = build_simple_utility_bill_payload(fields, tmp_file_path)
                        else:
                            result = {
                                "success": True,
                                "extracted_fields": fields,
                                "raw_text": ocr_result.text,
                                "confidence": getattr(ocr_result, 'confidence', 0.0)
                            }
                        
                        pipeline_used = "main"
                        
                    except Exception as e2:
                        logger.warning(f"Main pipeline failed: {e2}, falling back to simplified OCR")
                        # Fall back to simplified OCR
                        ocr_result = simple_run_ocr(tmp_file_path)
                        fields = simple_extract_fields(ocr_result['text'])
                        
                        if format == "utility_bill":
                            result = build_simple_utility_bill_payload(fields, tmp_file_path)
                        else:
                            result = {
                                "success": True,
                                "extracted_fields": fields,
                                "raw_text": ocr_result['text'],
                                "confidence": ocr_result['confidence']
                            }
                        
                        pipeline_used = f"fallback_{ocr_result['engine']}"
                else:
                    raise e
        
        elif PIPELINE_AVAILABLE:
            try:
                # Use main pipeline
                ocr_result = run_ocr(tmp_file_path)
                fields = extract_fields(ocr_result.text)
                
                if format == "utility_bill":
                    result = build_simple_utility_bill_payload(fields, tmp_file_path)
                else:
                    result = {
                        "success": True,
                        "extracted_fields": fields,
                        "raw_text": ocr_result.text,
                        "confidence": getattr(ocr_result, 'confidence', 0.0)
                    }
                
                pipeline_used = "main"
                
            except Exception as e:
                logger.warning(f"Main pipeline failed: {e}, falling back to simplified OCR")
                # Fall back to simplified OCR
                ocr_result = simple_run_ocr(tmp_file_path)
                fields = simple_extract_fields(ocr_result['text'])
                
                if format == "utility_bill":
                    result = build_simple_utility_bill_payload(fields, tmp_file_path)
                else:
                    result = {
                        "success": True,
                        "extracted_fields": fields,
                        "raw_text": ocr_result['text'],
                        "confidence": ocr_result['confidence']
                    }
                
                pipeline_used = f"fallback_{ocr_result['engine']}"
                
        else:
            # Use simplified OCR pipeline
            ocr_result = simple_run_ocr(tmp_file_path)
            fields = simple_extract_fields(ocr_result['text'])
            
            if format == "utility_bill":
                result = build_simple_utility_bill_payload(fields, tmp_file_path)
            else:
                result = {
                    "success": True,
                    "extracted_fields": fields,
                    "raw_text": ocr_result['text'],
                    "confidence": ocr_result['confidence']
                }
            
            pipeline_used = f"simplified_{ocr_result['engine']}"
        
        # Add processing metadata
        result["processing_metadata"] = {
            "pipeline_used": pipeline_used,
            "model_requested": model,
            "file_name": file.filename,
            "file_size": len(content),
            "format_requested": format
        }
        
        logger.info(f"Successfully processed {file.filename} using {pipeline_used} pipeline")
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        try:
            if tmp_file_path.exists():
                tmp_file_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to clean up temp file: {e}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "detail": "Check /docs for available endpoints"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "Please try again later"}
    )

if __name__ == "__main__":
    # For local development
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
        log_level="info"
    )