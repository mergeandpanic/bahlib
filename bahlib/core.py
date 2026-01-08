"""
Bahlib Core Module - Face Anonymization Engine

Provides the main Bahlib class for detecting and anonymizing faces in images.
Works 100% locally using MediaPipe's TFLite models.

Supports multiple anonymization methods:
- blur: Gaussian blur with feathered edges (default)
- pixelate: Mosaic/pixelation effect
- blackbar: Black bar over eyes only
"""

import cv2
import numpy as np
from typing import Union, Optional, List

# Try to import MediaPipe - support both legacy and new APIs
_USE_LEGACY_API = False
try:
    import mediapipe as mp
    # Check if legacy solutions API is available
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_detection'):
        _USE_LEGACY_API = True
    else:
        # Try new Tasks API
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        import urllib.request
        import os
except ImportError as e:
    raise ImportError(
        "MediaPipe is required. Install with: pip install mediapipe"
    ) from e


# Model URL for new Tasks API
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
_MODEL_PATH = None


def _get_model_path() -> str:
    """Download and cache the face detection model for new API."""
    global _MODEL_PATH
    if _MODEL_PATH and os.path.exists(_MODEL_PATH):
        return _MODEL_PATH
    
    # Save to user's cache directory
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "bahlib")
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, "blaze_face_short_range.tflite")
    
    if not os.path.exists(model_path):
        print(f"Downloading face detection model to {model_path}...")
        urllib.request.urlretrieve(_MODEL_URL, model_path)
        print("Download complete.")
    
    _MODEL_PATH = model_path
    return model_path


def _compute_iou(box1: dict, box2: dict) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    x1 = max(box1['x'], box2['x'])
    y1 = max(box1['y'], box2['y'])
    x2 = min(box1['x'] + box1['width'], box2['x'] + box2['width'])
    y2 = min(box1['y'] + box1['height'], box2['y'] + box2['height'])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = box1['width'] * box1['height']
    area2 = box2['width'] * box2['height']
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def _non_max_suppression(detections: List[dict], iou_threshold: float = 0.5) -> List[dict]:
    """Apply Non-Maximum Suppression to remove overlapping detections."""
    if not detections:
        return []
    
    # Sort by confidence (highest first)
    sorted_dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    while sorted_dets:
        best = sorted_dets.pop(0)
        keep.append(best)
        
        # Remove boxes with high IoU with the best box
        sorted_dets = [
            det for det in sorted_dets 
            if _compute_iou(best, det) < iou_threshold
        ]
    
    return keep


class Bahlib:
    """
    Local face anonymization engine using MediaPipe face detection.
    
    Supports multi-scale detection for finding small faces in group photos.
    
    Attributes:
        model_selection: 0 for short-range (within 2m), 1 for full-range (within 5m)
        min_detection_confidence: Minimum confidence threshold for face detection
    
    Example:
        >>> bh = Bahlib()
        >>> result = bh.anonymize("photo.jpg")
        >>> cv2.imwrite("anonymized.jpg", result)
        
        # For group photos with small faces
        >>> with Bahlib(min_detection_confidence=0.3) as bh:
        ...     result = bh.anonymize("group.jpg", scale_factor=2.0)
    """
    
    def __init__(
        self, 
        model_selection: int = 1, 
        min_detection_confidence: float = 0.5
    ):
        """
        Initialize the Bahlib face anonymizer.
        
        Args:
            model_selection: 0 for short-range model (within 2m), 
                           1 for full-range model (within 5m). Default is 1.
            min_detection_confidence: Minimum confidence value [0.0, 1.0] for 
                                    face detection to be considered successful.
                                    Lower values detect more faces but may have
                                    more false positives. Default is 0.5.
        """
        self.model_selection = model_selection
        self.min_detection_confidence = min_detection_confidence
        self._face_detection = None
        self._use_legacy = _USE_LEGACY_API
        
        if self._use_legacy:
            self._init_legacy_api()
        else:
            self._init_tasks_api()
    
    def _init_legacy_api(self):
        """Initialize using legacy mp.solutions API."""
        self._mp_face_detection = mp.solutions.face_detection
        self._face_detection = self._mp_face_detection.FaceDetection(
            model_selection=self.model_selection,
            min_detection_confidence=self.min_detection_confidence
        )
    
    def _init_tasks_api(self):
        """Initialize using new MediaPipe Tasks API."""
        model_path = _get_model_path()
        
        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=self.min_detection_confidence
        )
        self._face_detection = mp_vision.FaceDetector.create_from_options(options)
    
    def _detect_legacy(self, rgb_img: np.ndarray) -> list:
        """Detect faces using legacy API."""
        results = self._face_detection.process(rgb_img)
        
        detections = []
        if results.detections:
            h, w = rgb_img.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                det = {
                    'x': int(bbox.xmin * w),
                    'y': int(bbox.ymin * h),
                    'width': int(bbox.width * w),
                    'height': int(bbox.height * h),
                    'confidence': detection.score[0]
                }
                
                # Extract eye keypoints if available
                # MediaPipe keypoints: 0=right_eye, 1=left_eye, 2=nose_tip, 
                #                      3=mouth_center, 4=right_ear, 5=left_ear
                keypoints = detection.location_data.relative_keypoints
                if keypoints and len(keypoints) >= 2:
                    det['right_eye'] = (int(keypoints[0].x * w), int(keypoints[0].y * h))
                    det['left_eye'] = (int(keypoints[1].x * w), int(keypoints[1].y * h))
                
                detections.append(det)
        return detections
    
    def _detect_tasks(self, rgb_img: np.ndarray) -> list:
        """Detect faces using new Tasks API."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        results = self._face_detection.detect(mp_image)
        
        detections = []
        h, w = rgb_img.shape[:2]
        for detection in results.detections:
            bbox = detection.bounding_box
            det = {
                'x': bbox.origin_x,
                'y': bbox.origin_y,
                'width': bbox.width,
                'height': bbox.height,
                'confidence': detection.categories[0].score if detection.categories else 0.0
            }
            
            # Extract eye keypoints if available
            # Tasks API keypoints: 0=right_eye, 1=left_eye, 2=nose_tip, etc.
            if detection.keypoints and len(detection.keypoints) >= 2:
                det['right_eye'] = (int(detection.keypoints[0].x * w), 
                                    int(detection.keypoints[0].y * h))
                det['left_eye'] = (int(detection.keypoints[1].x * w), 
                                   int(detection.keypoints[1].y * h))
            
            detections.append(det)
        return detections
    
    def _detect_at_scale(self, rgb_img: np.ndarray, scale: float) -> list:
        """
        Detect faces at a specific scale.
        
        Args:
            rgb_img: RGB image as numpy array.
            scale: Scale factor (1.0 = original, 2.0 = 2x upscale).
        
        Returns:
            list: Detections with coordinates scaled back to original image.
        """
        h, w = rgb_img.shape[:2]
        
        if scale != 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            scaled_img = cv2.resize(rgb_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            scaled_img = rgb_img
        
        # Detect faces
        if self._use_legacy:
            detections = self._detect_legacy(scaled_img)
        else:
            detections = self._detect_tasks(scaled_img)
        
        # Scale coordinates back to original image size
        if scale != 1.0:
            for det in detections:
                det['x'] = int(det['x'] / scale)
                det['y'] = int(det['y'] / scale)
                det['width'] = int(det['width'] / scale)
                det['height'] = int(det['height'] / scale)
        
        return detections
    
    def _multi_scale_detect(
        self, 
        rgb_img: np.ndarray, 
        scales: List[float]
    ) -> list:
        """
        Detect faces at multiple scales and merge results.
        
        Args:
            rgb_img: RGB image as numpy array.
            scales: List of scale factors to try.
        
        Returns:
            list: Merged detections after NMS.
        """
        all_detections = []
        
        for scale in scales:
            detections = self._detect_at_scale(rgb_img, scale)
            all_detections.extend(detections)
        
        # Apply non-maximum suppression to remove duplicates
        return _non_max_suppression(all_detections, iou_threshold=0.5)
    
    def _tiled_detect(
        self, 
        rgb_img: np.ndarray, 
        tile_size: int = 640,
        overlap: float = 0.25,
        min_face_size: int = 20,
        min_confidence_boost: float = 0.15
    ) -> list:
        """
        Detect faces using sliding window tiles for small face detection.
        
        This approach processes the image in overlapping tiles, which
        effectively makes small faces larger relative to the detector.
        
        Args:
            rgb_img: RGB image as numpy array.
            tile_size: Size of each tile (square). Default is 640.
            overlap: Overlap between tiles (0.0-0.5). Default is 0.25.
            min_face_size: Minimum face size in pixels to accept. Default is 20.
            min_confidence_boost: Extra confidence required for tile detections
                                  vs full image detections. Default is 0.15.
        
        Returns:
            list: Merged detections after NMS.
        """
        h, w = rgb_img.shape[:2]
        all_detections = []
        
        # If image is smaller than tile size, just detect normally
        if h <= tile_size and w <= tile_size:
            if self._use_legacy:
                return self._detect_legacy(rgb_img)
            else:
                return self._detect_tasks(rgb_img)
        
        # Calculate step size based on overlap
        step = int(tile_size * (1 - overlap))
        
        # Minimum confidence for tile detections (higher to reduce false positives)
        tile_min_conf = self.min_detection_confidence + min_confidence_boost
        
        # Process tiles
        for y in range(0, h, step):
            for x in range(0, w, step):
                # Calculate tile bounds
                x1 = x
                y1 = y
                x2 = min(x + tile_size, w)
                y2 = min(y + tile_size, h)
                
                # Skip if tile is too small
                if (x2 - x1) < tile_size * 0.5 or (y2 - y1) < tile_size * 0.5:
                    continue
                
                # Extract tile
                tile = rgb_img[y1:y2, x1:x2]
                
                # Detect faces in tile
                if self._use_legacy:
                    tile_detections = self._detect_legacy(tile)
                else:
                    tile_detections = self._detect_tasks(tile)
                
                # Filter and adjust coordinates to original image space
                for det in tile_detections:
                    # Apply stricter confidence filter for tile detections
                    if det['confidence'] < tile_min_conf:
                        continue
                    
                    # Filter by minimum face size
                    if det['width'] < min_face_size or det['height'] < min_face_size:
                        continue
                    
                    # Filter by aspect ratio (faces should be roughly square)
                    aspect_ratio = det['width'] / max(det['height'], 1)
                    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                        continue
                    
                    det['x'] += x1
                    det['y'] += y1
                    all_detections.append(det)
        
        # Also detect on the full image for larger faces (original confidence)
        if self._use_legacy:
            full_detections = self._detect_legacy(rgb_img)
        else:
            full_detections = self._detect_tasks(rgb_img)
        
        # Filter full detections by minimum size
        for det in full_detections:
            if det['width'] >= min_face_size and det['height'] >= min_face_size:
                all_detections.append(det)
        
        # Apply NMS to remove duplicates
        return _non_max_suppression(all_detections, iou_threshold=0.4)
    
    def anonymize(
        self, 
        image: Union[str, np.ndarray], 
        blur_strength: int = 51,
        feather_amount: float = 0.3,
        scale_factor: Optional[float] = None,
        multi_scale: bool = False,
        tiled: bool = False,
        tile_size: int = 640,
        method: str = 'blur',
        pixelate_blocks: int = 10
    ) -> np.ndarray:
        """
        Detect and anonymize all faces in an image.
        
        Args:
            image: Either a file path (str) or a numpy array (BGR format).
            blur_strength: Kernel size for Gaussian blur. Must be odd and positive.
                          Higher values = stronger blur. Default is 51.
            feather_amount: How much to feather the edges (0.0-1.0). Higher values
                           create softer transitions. Default is 0.3.
            scale_factor: Upscale factor for detection. Use 1.5-2.0 for images
                         with small faces (e.g., group photos). Default is None
                         (no scaling, or auto-scale if multi_scale=True).
            multi_scale: If True, detect faces at multiple scales (1.0, 1.5, 2.0)
                        for better detection of small faces. Slower but more
                        accurate for group photos. Default is False.
            tiled: If True, use sliding window tiles for detecting small faces
                  in large images. Best for group photos. Default is False.
            tile_size: Size of tiles when tiled=True. Default is 640.
            method: Anonymization method. Options:
                   - 'blur': Gaussian blur with feathered edges (default)
                   - 'pixelate': Mosaic/pixelation effect
                   - 'blackbar': Black bar over eyes only
                   - 'all': Pixelation + black bar combined
            pixelate_blocks: Number of pixel blocks for pixelation (default 10).
                            Lower = more pixelated.
        
        Returns:
            numpy.ndarray: The anonymized image in BGR format.
            
        Raises:
            ValueError: If image cannot be loaded or parameters are invalid.
            FileNotFoundError: If image path does not exist.
        """
        # Validate method
        valid_methods = ('blur', 'pixelate', 'blackbar', 'all')
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Must be one of: {valid_methods}")
        
        # Ensure blur_strength is odd (required by OpenCV GaussianBlur)
        if blur_strength % 2 == 0:
            blur_strength += 1
        if blur_strength < 1:
            blur_strength = 1
        
        # Load image if path provided
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise FileNotFoundError(f"Could not load image: {image}")
        else:
            img = image.copy()  # Don't modify original array
        
        # Convert BGR to RGB for MediaPipe
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        if tiled:
            # Tiled detection for group photos with many small faces
            faces = self._tiled_detect(rgb_img, tile_size=tile_size)
        elif multi_scale:
            # Multi-scale detection for group photos
            scales = [1.0, 1.5, 2.0]
            faces = self._multi_scale_detect(rgb_img, scales)
        elif scale_factor is not None and scale_factor != 1.0:
            # Single scale with upscaling
            faces = self._detect_at_scale(rgb_img, scale_factor)
        else:
            # Standard single-scale detection
            if self._use_legacy:
                faces = self._detect_legacy(rgb_img)
            else:
                faces = self._detect_tasks(rgb_img)
        
        # If no faces detected, return original image
        if not faces:
            return img
        
        h, w = img.shape[:2]
        
        for face in faces:
            if method == 'blur':
                img = self._apply_blur(img, face, blur_strength, feather_amount)
            elif method == 'pixelate':
                img = self._apply_pixelate(img, face, pixelate_blocks, feather_amount)
            elif method == 'blackbar':
                img = self._apply_blackbar(img, face)
            elif method == 'all':
                # Apply pixelation first, then black bar
                img = self._apply_pixelate(img, face, pixelate_blocks, feather_amount)
                img = self._apply_blackbar(img, face)
        
        return img
    
    def _apply_blur(
        self,
        img: np.ndarray,
        face: dict,
        blur_strength: int,
        feather_amount: float
    ) -> np.ndarray:
        """Apply Gaussian blur to a face region with feathered edges."""
        h, w = img.shape[:2]
        x = face['x']
        y = face['y']
        bw = face['width']
        bh = face['height']
        
        # Expand the region slightly for better coverage
        padding = int(max(bw, bh) * 0.15)
        x = max(0, x - padding)
        y = max(0, y - padding)
        x2 = min(w, x + bw + 2 * padding)
        y2 = min(h, y + bh + 2 * padding)
        
        # Skip if ROI is invalid
        if x2 <= x or y2 <= y:
            return img
        
        roi_w = x2 - x
        roi_h = y2 - y
        
        # Create elliptical mask with feathered edges
        mask = self._create_feathered_ellipse_mask(roi_w, roi_h, feather_amount)
        
        # Extract face region and blur it
        face_roi = img[y:y2, x:x2].copy()
        blurred_face = cv2.GaussianBlur(
            face_roi, 
            (blur_strength, blur_strength), 
            0
        )
        
        # Blend blurred and original using the mask
        mask_3ch = np.stack([mask] * 3, axis=-1)
        blended = (blurred_face * mask_3ch + face_roi * (1 - mask_3ch)).astype(np.uint8)
        
        img[y:y2, x:x2] = blended
        return img
    
    def _apply_pixelate(
        self,
        img: np.ndarray,
        face: dict,
        blocks: int = 10,
        feather_amount: float = 0.3
    ) -> np.ndarray:
        """Apply pixelation/mosaic effect to a face region."""
        h, w = img.shape[:2]
        x = face['x']
        y = face['y']
        bw = face['width']
        bh = face['height']
        
        # Expand the region slightly for better coverage
        padding = int(max(bw, bh) * 0.15)
        x = max(0, x - padding)
        y = max(0, y - padding)
        x2 = min(w, x + bw + 2 * padding)
        y2 = min(h, y + bh + 2 * padding)
        
        # Skip if ROI is invalid
        if x2 <= x or y2 <= y:
            return img
        
        roi_w = x2 - x
        roi_h = y2 - y
        
        # Extract face region
        face_roi = img[y:y2, x:x2].copy()
        
        # Calculate block size based on face dimensions
        block_w = max(1, roi_w // blocks)
        block_h = max(1, roi_h // blocks)
        
        # Pixelate: shrink then enlarge with nearest neighbor
        small = cv2.resize(face_roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
        
        # Create elliptical mask with feathered edges
        mask = self._create_feathered_ellipse_mask(roi_w, roi_h, feather_amount)
        
        # Blend pixelated and original using the mask
        mask_3ch = np.stack([mask] * 3, axis=-1)
        blended = (pixelated * mask_3ch + face_roi * (1 - mask_3ch)).astype(np.uint8)
        
        img[y:y2, x:x2] = blended
        return img
    
    def _apply_blackbar(
        self,
        img: np.ndarray,
        face: dict
    ) -> np.ndarray:
        """Apply black bar over eyes."""
        h, w = img.shape[:2]
        
        # Try to use eye keypoints if available
        if 'right_eye' in face and 'left_eye' in face:
            right_eye = face['right_eye']
            left_eye = face['left_eye']
            
            # Calculate bar dimensions based on eye positions
            eye_distance = abs(left_eye[0] - right_eye[0])
            eye_center_y = (right_eye[1] + left_eye[1]) // 2
            
            # Bar width: extend beyond eyes
            bar_width_padding = int(eye_distance * 0.4)
            bar_x1 = max(0, min(right_eye[0], left_eye[0]) - bar_width_padding)
            bar_x2 = min(w, max(right_eye[0], left_eye[0]) + bar_width_padding)
            
            # Bar height: proportional to eye distance
            bar_height = max(int(eye_distance * 0.35), 10)
            bar_y1 = max(0, eye_center_y - bar_height // 2)
            bar_y2 = min(h, eye_center_y + bar_height // 2)
        else:
            # Fallback: estimate eye position from face bounding box
            # Eyes are typically in the upper third of the face
            x = face['x']
            y = face['y']
            bw = face['width']
            bh = face['height']
            
            # Eyes are roughly 1/3 down from top of face
            eye_y = y + int(bh * 0.35)
            bar_height = max(int(bh * 0.15), 10)
            
            bar_x1 = max(0, x + int(bw * 0.1))
            bar_x2 = min(w, x + int(bw * 0.9))
            bar_y1 = max(0, eye_y - bar_height // 2)
            bar_y2 = min(h, eye_y + bar_height // 2)
        
        # Draw black bar
        cv2.rectangle(img, (bar_x1, bar_y1), (bar_x2, bar_y2), (0, 0, 0), -1)
        
        return img
    
    def _create_feathered_ellipse_mask(
        self, 
        width: int, 
        height: int, 
        feather_amount: float = 0.3
    ) -> np.ndarray:
        """
        Create an elliptical mask with soft feathered edges.
        
        Args:
            width: Width of the mask.
            height: Height of the mask.
            feather_amount: How much to feather (0.0-1.0).
        
        Returns:
            numpy.ndarray: Float mask with values 0.0-1.0.
        """
        # Create coordinate grids
        y_coords, x_coords = np.ogrid[:height, :width]
        
        # Center of the ellipse
        cx, cy = width / 2, height / 2
        
        # Semi-axes (slightly smaller than half to ensure face is covered)
        a = width / 2 * 0.95  # horizontal semi-axis
        b = height / 2 * 0.95  # vertical semi-axis
        
        # Calculate normalized distance from center (ellipse equation)
        # Values <= 1 are inside the ellipse
        dist = ((x_coords - cx) / a) ** 2 + ((y_coords - cy) / b) ** 2
        
        # Create smooth falloff using sigmoid-like function
        # Inner region (dist < 1-feather) is fully blurred
        # Transition region smoothly blends
        inner_threshold = (1 - feather_amount) ** 2
        outer_threshold = (1 + feather_amount * 0.5) ** 2
        
        mask = np.zeros((height, width), dtype=np.float32)
        
        # Fully inside
        mask[dist <= inner_threshold] = 1.0
        
        # Transition zone - smooth gradient
        transition = (dist > inner_threshold) & (dist <= outer_threshold)
        if np.any(transition):
            # Smooth cosine interpolation for natural falloff
            t = (dist[transition] - inner_threshold) / (outer_threshold - inner_threshold)
            mask[transition] = 0.5 * (1 + np.cos(np.pi * t))
        
        return mask
    
    def detect_faces(
        self, 
        image: Union[str, np.ndarray],
        scale_factor: Optional[float] = None,
        multi_scale: bool = False,
        tiled: bool = False,
        tile_size: int = 640
    ) -> list:
        """
        Detect faces and return their bounding boxes without blurring.
        
        Args:
            image: Either a file path (str) or a numpy array (BGR format).
            scale_factor: Upscale factor for detection. Use 1.5-2.0 for small faces.
            multi_scale: If True, detect at multiple scales for better accuracy.
            tiled: If True, use sliding window tiles for small face detection.
            tile_size: Size of tiles when tiled=True. Default is 640.
        
        Returns:
            list: List of dictionaries containing face bounding boxes.
                  Each dict has keys: 'x', 'y', 'width', 'height', 'confidence'
        """
        # Load image if path provided
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise FileNotFoundError(f"Could not load image: {image}")
        else:
            img = image
        
        # Convert BGR to RGB for MediaPipe
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect using appropriate method
        if tiled:
            return self._tiled_detect(rgb_img, tile_size=tile_size)
        elif multi_scale:
            scales = [1.0, 1.5, 2.0]
            return self._multi_scale_detect(rgb_img, scales)
        elif scale_factor is not None and scale_factor != 1.0:
            return self._detect_at_scale(rgb_img, scale_factor)
        else:
            if self._use_legacy:
                return self._detect_legacy(rgb_img)
            else:
                return self._detect_tasks(rgb_img)
    
    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, '_face_detection') and self._face_detection is not None:
            if self._use_legacy:
                self._face_detection.close()
            else:
                self._face_detection.close()
            self._face_detection = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.close()
        return False
    
    def __del__(self):
        """Destructor to ensure resources are released."""
        if hasattr(self, '_face_detection'):
            self.close()
