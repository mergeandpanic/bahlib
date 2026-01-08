"""
Unit tests for Bahlib core functionality.

Run with: pytest tests/test_core.py -v
"""

import pytest
import numpy as np
import cv2
import os
import tempfile
from pathlib import Path

from bahlib import Bahlib
from bahlib.batch import get_supported_formats, SUPPORTED_FORMATS


class TestBahlibCore:
    """Tests for the main Bahlib class."""
    
    def test_initialization_default(self):
        """Test default initialization parameters."""
        bh = Bahlib()
        assert bh.model_selection == 1
        assert bh.min_detection_confidence == 0.5
        bh.close()
    
    def test_initialization_custom(self):
        """Test custom initialization parameters."""
        bh = Bahlib(model_selection=0, min_detection_confidence=0.7)
        assert bh.model_selection == 0
        assert bh.min_detection_confidence == 0.7
        bh.close()
    
    def test_context_manager(self):
        """Test context manager functionality."""
        with Bahlib() as bh:
            assert bh._face_detection is not None
        # After context exit, resources should be released
        assert bh._face_detection is None
    
    def test_anonymize_numpy_array(self):
        """Test anonymizing a numpy array image."""
        # Create a simple test image (blank white image)
        test_img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        with Bahlib() as bh:
            result = bh.anonymize(test_img, blur_strength=51)
        
        # Result should be same shape as input
        assert result.shape == test_img.shape
        assert result.dtype == test_img.dtype
    
    def test_anonymize_does_not_modify_original(self):
        """Test that anonymize doesn't modify the original array."""
        test_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        original_copy = test_img.copy()
        
        with Bahlib() as bh:
            _ = bh.anonymize(test_img, blur_strength=51)
        
        # Original should be unchanged
        np.testing.assert_array_equal(test_img, original_copy)
    
    def test_anonymize_file_not_found(self):
        """Test that FileNotFoundError is raised for missing files."""
        with Bahlib() as bh:
            with pytest.raises(FileNotFoundError):
                bh.anonymize("nonexistent_image.jpg")
    
    def test_blur_strength_odd_enforcement(self):
        """Test that even blur_strength is converted to odd."""
        test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        with Bahlib() as bh:
            # Even number should work (converted internally to odd)
            result = bh.anonymize(test_img, blur_strength=50)
            assert result is not None
    
    def test_detect_faces_empty_image(self):
        """Test face detection on an image with no faces."""
        # Create blank image with no faces
        test_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        with Bahlib() as bh:
            faces = bh.detect_faces(test_img)
        
        # Should return empty list for no faces
        assert isinstance(faces, list)
        assert len(faces) == 0
    
    def test_detect_faces_returns_correct_format(self):
        """Test that detect_faces returns correct dictionary format."""
        # This test uses a blank image, but verifies the return structure
        test_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        with Bahlib() as bh:
            faces = bh.detect_faces(test_img)
        
        assert isinstance(faces, list)
        # If there were faces, each would have these keys
        expected_keys = {'x', 'y', 'width', 'height', 'confidence'}
        for face in faces:
            assert set(face.keys()) == expected_keys


class TestBahlibWithTempFile:
    """Tests that use temporary files."""
    
    def test_anonymize_from_file(self):
        """Test anonymizing from a file path."""
        # Create a temporary image file
        test_img = np.ones((480, 640, 3), dtype=np.uint8) * 200
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            temp_path = f.name
        
        try:
            cv2.imwrite(temp_path, test_img)
            
            with Bahlib() as bh:
                result = bh.anonymize(temp_path, blur_strength=51)
            
            assert result.shape == test_img.shape
        finally:
            os.unlink(temp_path)
    
    def test_detect_faces_from_file(self):
        """Test face detection from a file path."""
        test_img = np.ones((480, 640, 3), dtype=np.uint8) * 200
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
        
        try:
            cv2.imwrite(temp_path, test_img)
            
            with Bahlib() as bh:
                faces = bh.detect_faces(temp_path)
            
            assert isinstance(faces, list)
        finally:
            os.unlink(temp_path)


class TestBatchUtilities:
    """Tests for batch module utilities."""
    
    def test_get_supported_formats(self):
        """Test that supported formats are returned correctly."""
        formats = get_supported_formats()
        
        assert isinstance(formats, set)
        assert '.jpg' in formats
        assert '.png' in formats
        assert '.jpeg' in formats
        assert '.bmp' in formats
        assert '.webp' in formats
    
    def test_supported_formats_immutable(self):
        """Test that get_supported_formats returns a copy."""
        formats1 = get_supported_formats()
        formats1.add('.xyz')
        formats2 = get_supported_formats()
        
        # The added format shouldn't appear in the new copy
        assert '.xyz' not in formats2


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_very_small_image(self):
        """Test with a very small image."""
        test_img = np.ones((10, 10, 3), dtype=np.uint8) * 128
        
        with Bahlib() as bh:
            result = bh.anonymize(test_img, blur_strength=3)
        
        assert result.shape == test_img.shape
    
    def test_grayscale_image_handling(self):
        """Test behavior with grayscale images."""
        # Note: MediaPipe expects 3-channel images, but cv2.cvtColor 
        # may handle this differently depending on the version
        test_img = np.ones((100, 100), dtype=np.uint8) * 128
        
        with Bahlib() as bh:
            # Either raises an error OR processes gracefully
            # We just ensure it doesn't crash unexpectedly
            try:
                result = bh.anonymize(test_img)
                # If it succeeds, result should be an array
                assert isinstance(result, np.ndarray)
            except (cv2.error, ValueError):
                # Expected error for grayscale input is acceptable
                pass
    
    def test_minimum_blur_strength(self):
        """Test with minimum blur strength."""
        test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        with Bahlib() as bh:
            result = bh.anonymize(test_img, blur_strength=1)
        
        assert result is not None
    
    def test_large_blur_strength(self):
        """Test with large blur strength."""
        test_img = np.ones((200, 200, 3), dtype=np.uint8) * 128
        
        with Bahlib() as bh:
            result = bh.anonymize(test_img, blur_strength=99)
        
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

