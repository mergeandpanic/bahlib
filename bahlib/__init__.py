"""
Bahlib - Biometric Anonymizer Hub Library

A high-performance, lightweight Python library for 100% local face anonymization.
Automatically detects human faces in images and applies Gaussian blur for privacy.

Example:
    >>> from bahlib import Bahlib
    >>> bh = Bahlib()
    >>> result = bh.anonymize("photo.jpg")
    >>> cv2.imwrite("anonymized.jpg", result)
"""

__version__ = "0.1.0"
__author__ = "Bahlib Contributors"

from .core import Bahlib
from .video import anonymize_video, anonymize_webcam, process_frame
from .batch import anonymize_directory, anonymize_files, get_supported_formats

__all__ = [
    "Bahlib",
    "anonymize_video",
    "anonymize_webcam",
    "process_frame",
    "anonymize_directory",
    "anonymize_files",
    "get_supported_formats",
    "__version__",
]

