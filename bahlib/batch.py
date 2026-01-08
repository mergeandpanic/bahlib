"""
Bahlib Batch Module - Bulk Image Processing

Provides functions for processing multiple images at once.
"""

import os
import cv2
from pathlib import Path
from typing import List, Optional, Callable, Union
from .core import Bahlib


# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}


def anonymize_directory(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    blur_strength: int = 51,
    feather_amount: float = 0.3,
    model_selection: int = 1,
    min_detection_confidence: float = 0.5,
    recursive: bool = False,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    preserve_structure: bool = True,
    scale_factor: Optional[float] = None,
    multi_scale: bool = False,
    tiled: bool = False,
    tile_size: int = 640,
    method: str = 'blur',
    pixelate_blocks: int = 10
) -> dict:
    """
    Process all images in a directory and save anonymized versions.
    
    Args:
        input_dir: Path to input directory containing images.
        output_dir: Path to output directory for anonymized images.
        blur_strength: Kernel size for Gaussian blur (must be odd). Default is 51.
        feather_amount: Edge feather amount (0.0-1.0) for smooth blending. Default is 0.3.
        model_selection: 0 for short-range, 1 for full-range face detection.
        min_detection_confidence: Minimum confidence threshold [0.0, 1.0].
        recursive: If True, process subdirectories recursively.
        progress_callback: Optional callback function(filename, current, total).
        preserve_structure: If True, preserve subdirectory structure in output.
        scale_factor: Upscale factor for small face detection (e.g., 1.5, 2.0).
        multi_scale: If True, detect faces at multiple scales for better accuracy.
        tiled: If True, use tiled detection for small faces in large images.
        tile_size: Size of tiles when tiled=True. Default is 640.
        method: Anonymization method: 'blur', 'pixelate', 'blackbar', or 'all'.
        pixelate_blocks: Number of pixel blocks for pixelation. Default is 10.
    
    Returns:
        dict: Processing statistics with keys:
              - 'processed': List of successfully processed file paths
              - 'failed': List of (filepath, error_message) tuples
              - 'skipped': List of skipped file paths (non-image files)
              - 'total': Total number of image files found
    
    Raises:
        FileNotFoundError: If input directory does not exist.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    if not input_path.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all image files
    image_files = []
    if recursive:
        for ext in SUPPORTED_FORMATS:
            image_files.extend(input_path.rglob(f"*{ext}"))
            image_files.extend(input_path.rglob(f"*{ext.upper()}"))
    else:
        for ext in SUPPORTED_FORMATS:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    # Remove duplicates and sort
    image_files = sorted(set(image_files))
    total = len(image_files)
    
    results = {
        'processed': [],
        'failed': [],
        'skipped': [],
        'total': total
    }
    
    if total == 0:
        return results
    
    # Initialize Bahlib
    with Bahlib(
        model_selection=model_selection,
        min_detection_confidence=min_detection_confidence
    ) as bh:
        for idx, image_file in enumerate(image_files, 1):
            try:
                # Determine output path
                if preserve_structure and recursive:
                    relative_path = image_file.relative_to(input_path)
                    out_file = output_path / relative_path
                    out_file.parent.mkdir(parents=True, exist_ok=True)
                else:
                    out_file = output_path / image_file.name
                
                # Process image
                anonymized = bh.anonymize(
                    str(image_file), 
                    blur_strength=blur_strength,
                    feather_amount=feather_amount,
                    scale_factor=scale_factor,
                    multi_scale=multi_scale,
                    tiled=tiled,
                    tile_size=tile_size,
                    method=method,
                    pixelate_blocks=pixelate_blocks
                )
                
                # Save result
                cv2.imwrite(str(out_file), anonymized)
                results['processed'].append(str(out_file))
                
                # Progress callback
                if progress_callback:
                    progress_callback(image_file.name, idx, total)
                    
            except Exception as e:
                results['failed'].append((str(image_file), str(e)))
    
    return results


def anonymize_files(
    file_paths: List[Union[str, Path]],
    output_dir: Union[str, Path],
    blur_strength: int = 51,
    feather_amount: float = 0.3,
    model_selection: int = 1,
    min_detection_confidence: float = 0.5,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    scale_factor: Optional[float] = None,
    multi_scale: bool = False
) -> dict:
    """
    Process a list of specific image files.
    
    Args:
        file_paths: List of paths to image files.
        output_dir: Path to output directory for anonymized images.
        blur_strength: Kernel size for Gaussian blur (must be odd). Default is 51.
        feather_amount: Edge feather amount (0.0-1.0) for smooth blending. Default is 0.3.
        model_selection: 0 for short-range, 1 for full-range face detection.
        min_detection_confidence: Minimum confidence threshold [0.0, 1.0].
        progress_callback: Optional callback function(filename, current, total).
        scale_factor: Upscale factor for small face detection (e.g., 1.5, 2.0).
        multi_scale: If True, detect faces at multiple scales for better accuracy.
    
    Returns:
        dict: Processing statistics with keys:
              - 'processed': List of successfully processed file paths
              - 'failed': List of (filepath, error_message) tuples
              - 'total': Total number of files in input list
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    total = len(file_paths)
    results = {
        'processed': [],
        'failed': [],
        'total': total
    }
    
    if total == 0:
        return results
    
    with Bahlib(
        model_selection=model_selection,
        min_detection_confidence=min_detection_confidence
    ) as bh:
        for idx, file_path in enumerate(file_paths, 1):
            file_path = Path(file_path)
            
            try:
                # Process image
                anonymized = bh.anonymize(
                    str(file_path), 
                    blur_strength=blur_strength,
                    feather_amount=feather_amount,
                    scale_factor=scale_factor,
                    multi_scale=multi_scale
                )
                
                # Save result
                out_file = output_path / file_path.name
                cv2.imwrite(str(out_file), anonymized)
                results['processed'].append(str(out_file))
                
                # Progress callback
                if progress_callback:
                    progress_callback(file_path.name, idx, total)
                    
            except Exception as e:
                results['failed'].append((str(file_path), str(e)))
    
    return results


def get_supported_formats() -> set:
    """
    Get the set of supported image file extensions.
    
    Returns:
        set: Set of supported extensions (lowercase, with leading dot).
    """
    return SUPPORTED_FORMATS.copy()

