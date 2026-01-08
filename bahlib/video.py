"""
Bahlib Video Module - Real-time Video and Webcam Face Anonymization

Provides functions for processing video files and live webcam feeds.
"""

import cv2
import numpy as np
from typing import Optional, Callable
from .core import Bahlib


def anonymize_video(
    input_path: str,
    output_path: str,
    blur_strength: int = 51,
    feather_amount: float = 0.3,
    model_selection: int = 1,
    min_detection_confidence: float = 0.5,
    codec: str = "mp4v",
    progress_callback: Optional[Callable[[int, int], None]] = None,
    scale_factor: Optional[float] = None
) -> dict:
    """
    Process a video file and anonymize all detected faces.
    
    Args:
        input_path: Path to input video file.
        output_path: Path to save anonymized video.
        blur_strength: Kernel size for Gaussian blur (must be odd). Default is 51.
        feather_amount: Edge feather amount (0.0-1.0) for smooth blending. Default is 0.3.
        model_selection: 0 for short-range, 1 for full-range face detection.
        min_detection_confidence: Minimum confidence threshold [0.0, 1.0].
        codec: FourCC codec code (e.g., 'mp4v', 'XVID', 'MJPG'). Default is 'mp4v'.
        progress_callback: Optional callback function(current_frame, total_frames).
        scale_factor: Upscale factor for small face detection (e.g., 1.5, 2.0).
    
    Returns:
        dict: Processing statistics with keys:
              - 'total_frames': Total frames in video
              - 'processed_frames': Successfully processed frames
              - 'fps': Original video FPS
              - 'resolution': Tuple of (width, height)
    
    Raises:
        FileNotFoundError: If input video cannot be opened.
        ValueError: If output path is invalid.
    """
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        cap.release()
        raise ValueError(f"Could not create output video: {output_path}")
    
    # Initialize Bahlib
    with Bahlib(
        model_selection=model_selection,
        min_detection_confidence=min_detection_confidence
    ) as bh:
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Anonymize the frame
            anonymized_frame = bh.anonymize(
                frame, 
                blur_strength=blur_strength,
                feather_amount=feather_amount,
                scale_factor=scale_factor
            )
            out.write(anonymized_frame)
            
            processed_frames += 1
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(processed_frames, total_frames)
    
    # Cleanup
    cap.release()
    out.release()
    
    return {
        'total_frames': total_frames,
        'processed_frames': processed_frames,
        'fps': fps,
        'resolution': (width, height)
    }


def anonymize_webcam(
    camera_id: int = 0,
    blur_strength: int = 51,
    feather_amount: float = 0.3,
    model_selection: int = 1,
    min_detection_confidence: float = 0.5,
    window_name: str = "Bahlib - Press ESC to quit",
    mirror: bool = True
) -> None:
    """
    Run live face anonymization on webcam feed.
    
    Press ESC key to quit the webcam window.
    
    Args:
        camera_id: Camera device ID. Default is 0 (primary webcam).
        blur_strength: Kernel size for Gaussian blur (must be odd). Default is 51.
        feather_amount: Edge feather amount (0.0-1.0) for smooth blending. Default is 0.3.
        model_selection: 0 for short-range, 1 for full-range face detection.
        min_detection_confidence: Minimum confidence threshold [0.0, 1.0].
        window_name: Name of the display window.
        mirror: If True, flip the image horizontally for mirror effect.
    
    Raises:
        RuntimeError: If webcam cannot be opened.
    """
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam with ID: {camera_id}")
    
    with Bahlib(
        model_selection=model_selection,
        min_detection_confidence=min_detection_confidence
    ) as bh:
        print(f"Webcam started. Press ESC to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from webcam.")
                break
            
            # Mirror the frame if requested
            if mirror:
                frame = cv2.flip(frame, 1)
            
            # Anonymize the frame
            anonymized_frame = bh.anonymize(
                frame, 
                blur_strength=blur_strength,
                feather_amount=feather_amount
            )
            
            # Display the result
            cv2.imshow(window_name, anonymized_frame)
            
            # Check for ESC key (27)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("ESC pressed. Exiting...")
                break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


def process_frame(
    frame: np.ndarray,
    bahlib_instance: Bahlib,
    blur_strength: int = 51,
    feather_amount: float = 0.3,
    scale_factor: Optional[float] = None,
    multi_scale: bool = False
) -> np.ndarray:
    """
    Process a single frame - utility function for custom video pipelines.
    
    Args:
        frame: Input frame as numpy array (BGR format).
        bahlib_instance: An initialized Bahlib instance.
        blur_strength: Kernel size for Gaussian blur.
        feather_amount: Edge feather amount (0.0-1.0) for smooth blending.
        scale_factor: Upscale factor for small face detection.
        multi_scale: If True, detect faces at multiple scales.
    
    Returns:
        numpy.ndarray: Anonymized frame.
    """
    return bahlib_instance.anonymize(
        frame, 
        blur_strength=blur_strength,
        feather_amount=feather_amount,
        scale_factor=scale_factor,
        multi_scale=multi_scale
    )

