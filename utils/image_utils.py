"""
Image processing utilities for medical images.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

def match_shape(mask: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    Match mask shape to volume shape by transposing if necessary.
    Handles common orientation mismatches between masks and volumes.
    """
    if mask.shape == volume.shape:
        return mask
    elif mask.shape[::-1] == volume.shape:
        return np.transpose(mask, (2, 1, 0))
    elif (mask.shape[2], mask.shape[1], mask.shape[0]) == volume.shape:
        return np.transpose(mask, (2, 1, 0))
    else:
        logger.warning(f"Shape mismatch: mask {mask.shape}, volume {volume.shape}")
        return mask

def reorient_volume(volume: np.ndarray, detected_plane: str) -> np.ndarray:
    """
    Reorient volume to a standard orientation for consistent viewing.
    
    The goal is to make all volumes have the same orientation regardless of 
    how they were originally acquired, so that the slice extraction logic
    in the viewer works consistently.
    
    Standard orientation: volume[slice, height, width] where:
    - slice dimension (axis 0) has the most slices
    - height and width are the other two dimensions
    
    Args:
        volume: Input volume array
        detected_plane: The acquisition plane ("Axial", "Coronal", "Sagittal")
        
    Returns:
        Reoriented volume in standard format
    """
    shape = volume.shape
    original_slice_dim = np.argmax(shape)
    
    # Map anatomical plane to expected slice dimension
    # For DICOM: detected_plane tells us the anatomical acquisition plane
    # We need to determine if the volume needs reorientation based on shape
    
    # If the volume is already in standard orientation (slice dim = 0), return as-is
    if original_slice_dim == 0:
        return volume
    
    # Reorient to standard orientation (slice dimension = 0)
    if original_slice_dim == 1:
        # Move axis 1 (slice) to axis 0: (H, S, W) -> (S, H, W)
        return np.transpose(volume, (1, 0, 2))
    elif original_slice_dim == 2:
        # Move axis 2 (slice) to axis 0: (H, W, S) -> (S, H, W)
        return np.transpose(volume, (2, 0, 1))
    
    return volume

def detect_main_plane_dicom(first_dcm_path: str) -> str:
    """
    Detect the main acquisition plane from DICOM ImageOrientationPatient tag.
    Returns "Axial", "Coronal", or "Sagittal".
    """
    try:
        import pydicom
        ds = pydicom.dcmread(first_dcm_path, stop_before_pixels=True)
        
        # Check if ImageOrientationPatient exists
        if not hasattr(ds, 'ImageOrientationPatient'):
            logger.warning("ImageOrientationPatient tag not found, defaulting to Axial")
            return "Axial"
            
        iop = ds.ImageOrientationPatient
        
        if len(iop) != 6:
            logger.warning(f"Invalid ImageOrientationPatient length: {len(iop)}, defaulting to Axial")
            return "Axial"
            
        row = np.array(iop[0:3], dtype=float)
        col = np.array(iop[3:6], dtype=float)
        
        normal = np.cross(row, col)
        
        # Check for valid normal vector
        if np.allclose(normal, 0):
            logger.warning("Zero normal vector detected, defaulting to Axial")
            return "Axial"
            
        idx = int(np.argmax(np.abs(normal)))
        
        # Ensure index is valid
        if idx < 0 or idx >= 3:
            logger.warning(f"Invalid plane index: {idx}, defaulting to Axial")
            return "Axial"
            
        # Original mapping that was working
        planes = ["Axial", "Coronal", "Sagittal"]
        detected_plane = planes[idx]
        
        logger.info(f"DICOM orientation detected: {detected_plane}")
        return detected_plane
        
    except Exception as e:
        logger.warning(f"DICOM orientation detection failed: {e}")
        return "Axial"
