"""
DICOM loading functionality.
"""

import os
import numpy as np
import SimpleITK as sitk
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def load_dicom_series(folder: str) -> Tuple[np.ndarray, str, str]:
    """
    Load a DICOM series from a folder.
    
    Args:
        folder: Path to DICOM folder
        
    Returns:
        Tuple of (volume_array, detected_plane, body_part_examined)
    """
    try:
        reader = sitk.ImageSeriesReader()
        files = reader.GetGDCMSeriesFileNames(folder)
        if not files:
            raise ValueError("No DICOM files found")
            
        import pydicom
        ds = pydicom.dcmread(files[0], stop_before_pixels=True)
        
        body_part = getattr(ds, "BodyPartExamined", None)
        modality = getattr(ds, "Modality", None)
        body_part_examined = f"{body_part.lower()} {modality.lower() if modality else ''}".strip() if body_part else "Unknown"
        
        reader.SetFileNames(files)
        image = reader.Execute()
        arr = sitk.GetArrayFromImage(image).astype(np.float32)
        
        # Detect orientation with better error handling
        try:
            from utils.image_utils import detect_main_plane_dicom
            detected_plane = detect_main_plane_dicom(files[0])
            
            # Validate the detected plane
            valid_planes = ["Axial", "Coronal", "Sagittal"]
            if detected_plane not in valid_planes:
                logger.warning(f"Invalid plane detected: {detected_plane}, defaulting to Axial")
                detected_plane = "Axial"
                
        except Exception as e:
            logger.warning(f"Orientation detection failed: {e}, defaulting to Axial")
            detected_plane = "Axial"
        
        logger.info(f"Loaded {len(files)} DICOM slices. Shape: {arr.shape}, Plane: {detected_plane}")
        
        # Return volume without reorientation - keep original DICOM orientation
        return arr, detected_plane, body_part_examined
        
    except Exception as e:
        logger.error(f"DICOM loading error: {e}")
        raise
