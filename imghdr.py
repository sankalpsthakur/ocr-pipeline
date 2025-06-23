"""
Compatibility shim for imghdr module (removed in Python 3.13)
This provides basic image format detection for PaddleOCR compatibility.
"""

import mimetypes

def what(filename):
    """Determine image format from filename"""
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type:
        if mime_type.startswith('image/'):
            return mime_type.split('/')[-1]
    return None

# Add to sys.modules for PaddleOCR
import sys
sys.modules['imghdr'] = sys.modules[__name__]