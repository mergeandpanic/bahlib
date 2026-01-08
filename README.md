# Bahlib 🕶️

**Biometric Anonymizer Hub Library** — A fast, lightweight Python library for **100% local** face anonymization.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-17%20passed-brightgreen.svg)](#testing)

Bahlib detects human faces in images and videos, then applies smooth Gaussian blur with feathered edges. **Your data never leaves your machine.**

📖 **[Documentation](https://mergeandpanic.github.io/bahlib)**

## Features

- **100% Offline** — No cloud, no API keys, no data leaks
- **Multiple Methods** — Blur, pixelate, black bar, or all combined
- **Smooth Blending** — Elliptical masks with feathered edges for natural results
- **Group Photo Support** — Multi-scale and tiled detection for small faces
- **Real-time Video** — Process video files or live webcam feeds
- **Batch Processing** — Anonymize entire directories at once
- **CLI Tool** — Use directly from the command line
- **Lightweight** — Powered by MediaPipe TFLite models

## Installation

```bash
pip install bahlib
```

Or install from source:

```bash
git clone https://github.com/mergeandpanic/bahlib.git
cd bahlib
pip install -e .
```

**Requirements:** Python 3.8+, OpenCV, MediaPipe, NumPy

## Quick Start

### Python API

```python
from bahlib import Bahlib
import cv2

# Basic usage
with Bahlib() as bh:
    result = bh.anonymize("photo.jpg")
    cv2.imwrite("photo_blurred.jpg", result)

# Custom blur strength and feathering
with Bahlib(min_detection_confidence=0.6) as bh:
    result = bh.anonymize(
        "photo.jpg",
        blur_strength=75,      # Higher = stronger blur
        feather_amount=0.4     # Higher = softer edges (0.0-1.0)
    )
    cv2.imwrite("output.jpg", result)

# Group photos with small faces (RECOMMENDED)
with Bahlib(min_detection_confidence=0.3) as bh:
    result = bh.anonymize(
        "group.jpg",
        tiled=True,            # Best for many small faces
        tile_size=320          # Smaller = detects smaller faces
    )
    cv2.imwrite("group_blurred.jpg", result)

# Get face locations without blurring
with Bahlib() as bh:
    faces = bh.detect_faces("photo.jpg")
    for face in faces:
        print(f"Face at ({face['x']}, {face['y']}) - {face['confidence']:.0%} confidence")

# Different anonymization methods
with Bahlib() as bh:
    # Pixelation (mosaic effect)
    result = bh.anonymize("photo.jpg", method='pixelate', pixelate_blocks=10)
    
    # Black bar over eyes only
    result = bh.anonymize("photo.jpg", method='blackbar')
    
    # Pixelation + black bar (like news/crime photos)
    result = bh.anonymize("photo.jpg", method='all')
```

### Video Processing

```python
from bahlib import anonymize_video, anonymize_webcam

# Process a video file
stats = anonymize_video(
    "input.mp4",
    "output.mp4",
    blur_strength=51,
    feather_amount=0.3
)
print(f"Processed {stats['processed_frames']} frames")

# Live webcam anonymization (press ESC to quit)
anonymize_webcam(blur_strength=51)
```

### Batch Processing

```python
from bahlib import anonymize_directory

# Process all images in a directory
stats = anonymize_directory(
    "./photos",
    "./anonymized",
    blur_strength=51,
    recursive=True,
    tiled=True,           # Enable for group photos
    tile_size=320
)
print(f"Processed: {len(stats['processed'])}")
print(f"Failed: {len(stats['failed'])}")
```

### Command Line

```bash
# Single image
bahlib image photo.jpg -o blurred.jpg --blur 51 --feather 0.3

# Group photo with tiled detection (best for small faces)
bahlib image group.jpg -o output.jpg --tiled --tile-size 320 --confidence 0.3

# Multi-scale detection (alternative for group photos)
bahlib image group.jpg -o output.jpg --multi-scale --confidence 0.4

# Different anonymization methods
bahlib image photo.jpg -o output.jpg --method pixelate --pixelate-blocks 10
bahlib image photo.jpg -o output.jpg --method blackbar
bahlib image photo.jpg -o output.jpg --method all  # pixelate + blackbar

# Video file
bahlib video input.mp4 -o output.mp4 --blur 51

# Live webcam
bahlib webcam --blur 51 --feather 0.4

# Batch directory
bahlib batch ./photos ./output --recursive --blur 51 --tiled
```

## Anonymization Methods

| Method | Description | CLI Flag |
|--------|-------------|----------|
| **blur** | Gaussian blur with soft edges (default) | `--method blur` |
| **pixelate** | Mosaic/pixelation effect | `--method pixelate` |
| **blackbar** | Black bar over eyes only | `--method blackbar` |
| **all** | Pixelation + black bar combined | `--method all` |

**Tip:** Use `--method all` for the classic "news/crime photo" look (pixelate + black bar).

## Detection Modes

| Mode | Best For | CLI Flag |
|------|----------|----------|
| **Standard** | Single portraits, large faces | (default) |
| **Multi-scale** | Medium group photos | `--multi-scale` |
| **Tiled** | Large groups, small faces | `--tiled --tile-size 320` |

**Tip:** For group photos, start with `--tiled --tile-size 320 --confidence 0.3`

## API Reference

### `Bahlib` Class

```python
Bahlib(model_selection=1, min_detection_confidence=0.5)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_selection` | int | 1 | `0` = short-range (≤2m), `1` = full-range (≤5m) |
| `min_detection_confidence` | float | 0.5 | Detection threshold (0.0 to 1.0) |

#### Methods

**`anonymize(image, blur_strength=51, feather_amount=0.3, scale_factor=None, multi_scale=False, tiled=False, tile_size=640, method='blur', pixelate_blocks=10)`**

Detect and anonymize all faces in an image.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | str \| ndarray | — | File path or BGR numpy array |
| `blur_strength` | int | 51 | Gaussian blur kernel size (odd number) |
| `feather_amount` | float | 0.3 | Edge softness (0.0 = hard, 1.0 = very soft) |
| `scale_factor` | float | None | Upscale image before detection (e.g., 2.0) |
| `multi_scale` | bool | False | Detect at multiple scales (1x, 1.5x, 2x) |
| `tiled` | bool | False | Use tiled detection for small faces |
| `tile_size` | int | 640 | Tile size when `tiled=True` (smaller = more sensitive) |
| `method` | str | 'blur' | Anonymization method: 'blur', 'pixelate', 'blackbar', 'all' |
| `pixelate_blocks` | int | 10 | Pixel blocks for mosaic (lower = more pixelated) |

Returns: BGR numpy array

**`detect_faces(image, scale_factor=None, multi_scale=False, tiled=False, tile_size=640)`**

Get face bounding boxes without blurring.

Returns: List of dicts with keys `x`, `y`, `width`, `height`, `confidence`

**`close()`**

Release MediaPipe resources. Called automatically when using `with` statement.

### Video Functions

```python
anonymize_video(input_path, output_path, blur_strength=51, feather_amount=0.3, ...)
anonymize_webcam(camera_id=0, blur_strength=51, feather_amount=0.3, ...)
process_frame(frame, bahlib_instance, blur_strength=51, feather_amount=0.3)
```

### Batch Functions

```python
anonymize_directory(input_dir, output_dir, blur_strength=51, feather_amount=0.3, 
                    recursive=False, tiled=False, tile_size=640, ...)
anonymize_files(file_paths, output_dir, blur_strength=51, feather_amount=0.3, ...)
get_supported_formats()  # Returns: {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
```

## How It Works

1. **Detection** — MediaPipe TFLite model locates faces in the image
2. **Masking** — Creates elliptical mask with smooth feathered edges
3. **Blurring** — Applies Gaussian blur to face regions
4. **Blending** — Seamlessly merges blurred faces using cosine interpolation

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## Why Bahlib?

- **Privacy-first** — GDPR/CCPA compliant by design
- **No dependencies on external services** — Works in air-gapped environments
- **Production-ready** — Battle-tested with comprehensive test suite
- **Smooth results** — Feathered elliptical masks look natural, not like rectangles
- **Group photo support** — Tiled detection finds even the smallest faces

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*What happens on your hardware, stays on your hardware.* 🔒
