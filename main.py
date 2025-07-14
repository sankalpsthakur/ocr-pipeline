#!/usr/bin/env python3
"""FastAPI Web Service for OCR Pipeline

Simple REST API for utility bill OCR processing with Railway deployment support.
"""

import os
import tempfile
import json
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Import the core OCR pipeline
try:
    from pipeline import run_ocr, extract_fields, build_utility_bill_payload
    PIPELINE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Main pipeline import failed: {e}")
    PIPELINE_AVAILABLE = False

# Try PyTorch mobile pipeline as fallback
try:
    from pytorch_mobile.ocr_pipeline import run_ocr_with_tesseract, extract_fields as extract_fields_mobile, build_utility_bill_payload as build_payload_mobile
    MOBILE_PIPELINE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Mobile pipeline import failed: {e}")
    MOBILE_PIPELINE_AVAILABLE = False

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
                <li>Main Pipeline: """ + ("✅ Available" if PIPELINE_AVAILABLE else "❌ Not Available") + """</li>
                <li>Mobile Pipeline: """ + ("✅ Available" if MOBILE_PIPELINE_AVAILABLE else "❌ Not Available") + """</li>
            </ul>
            
            <h2>Usage Example:</h2>
            <pre>
curl -X POST "http://localhost:8000/ocr" \\
     -F "file=@bill.png" \\
     -F "format=utility_bill"
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
            "main_pipeline": PIPELINE_AVAILABLE,
            "mobile_pipeline": MOBILE_PIPELINE_AVAILABLE,
            "supported_formats": ["PNG", "JPEG", "PDF"],
            "output_formats": ["utility_bill", "basic"]
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
    format: str = Form(default="utility_bill")
):
    """
    Process uploaded image with OCR and extract fields.
    
    Args:
        file: Uploaded image file (PNG, JPEG, PDF)
        format: Output format ("utility_bill" or "basic")
    
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
    
    # Check if any pipeline is available
    if not PIPELINE_AVAILABLE and not MOBILE_PIPELINE_AVAILABLE:
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
        
        # Try main pipeline first
        if PIPELINE_AVAILABLE:
            try:
                # Use main pipeline
                ocr_result = run_ocr(tmp_file_path)
                fields = extract_fields(ocr_result.text)
                
                if format == "utility_bill":
                    result = build_utility_bill_payload(fields, tmp_file_path)
                else:
                    result = {
                        "success": True,
                        "extracted_fields": fields,
                        "raw_text": ocr_result.text,
                        "confidence": getattr(ocr_result, 'confidence', 0.0)
                    }
                
                pipeline_used = "main"
                
            except Exception as e:
                logger.warning(f"Main pipeline failed: {e}")
                if not MOBILE_PIPELINE_AVAILABLE:
                    raise
                # Fall back to mobile pipeline
                raise e
                
        else:
            # Use mobile pipeline
            result = run_ocr_with_tesseract(tmp_file_path)
            fields = extract_fields_mobile(result.get('_full_text', ''))
            
            if format == "utility_bill":
                result = build_payload_mobile(fields, tmp_file_path)
            else:
                result = {
                    "success": True,
                    "extracted_fields": fields,
                    "raw_text": result.get('_full_text', ''),
                    "confidence": result.get('_ocr_confidence', 0.0)
                }
            
            pipeline_used = "mobile"
        
        # Add processing metadata
        result["processing_metadata"] = {
            "pipeline_used": pipeline_used,
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