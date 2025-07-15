"""
Compatibility shim for imghdr module removed in Python 3.13
This provides basic image type detection functionality for PaddleOCR.
"""

import struct
from pathlib import Path

def what(filename, h=None):
    """Detect image format from file or file-like object."""
    if hasattr(filename, 'read'):
        # File-like object
        f = filename
        filename = getattr(f, 'name', '')
    else:
        # Filename string
        try:
            f = open(filename, 'rb')
        except (OSError, TypeError):
            return None
    
    try:
        if h is None:
            h = f.read(32)
        
        # PNG
        if h.startswith(b'\x89PNG\r\n\x1a\n'):
            return 'png'
        
        # JPEG
        if h.startswith(b'\xff\xd8\xff'):
            return 'jpeg'
        
        # GIF
        if h.startswith((b'GIF87a', b'GIF89a')):
            return 'gif'
        
        # BMP
        if h.startswith(b'BM'):
            return 'bmp'
        
        # TIFF
        if h.startswith((b'II*\x00', b'MM\x00*')):
            return 'tiff'
        
        # WebP
        if h.startswith(b'RIFF') and h[8:12] == b'WEBP':
            return 'webp'
        
        return None
        
    except Exception:
        return None
    finally:
        if hasattr(filename, 'read'):
            pass  # Don't close file-like objects we didn't open
        else:
            try:
                f.close()
            except:
                pass

# Legacy function names for compatibility
test_jpeg = lambda h, f: h.startswith(b'\xff\xd8\xff')
test_png = lambda h, f: h.startswith(b'\x89PNG\r\n\x1a\n')
test_gif = lambda h, f: h.startswith((b'GIF87a', b'GIF89a'))
test_bmp = lambda h, f: h.startswith(b'BM')
test_tiff = lambda h, f: h.startswith((b'II*\x00', b'MM\x00*'))
test_webp = lambda h, f: h.startswith(b'RIFF') and len(h) >= 12 and h[8:12] == b'WEBP'