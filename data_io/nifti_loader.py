"""
NIfTI loading functionality.
"""

import os
import numpy as np
import nibabel as nib
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

def load_nifti_file(file_path: str) -> Tuple[np.ndarray, str, str]:
    """
    Load a NIfTI file.
    
    Args:
        file_path: Path to NIfTI file
        
    Returns:
        Tuple of (volume_array, detected_plane, body_part_examined)
    """
    try:
        # Load NIfTI file using nibabel
        nifti_img = nib.load(file_path)
        arr = nifti_img.get_fdata().astype(np.float32)
        
        # Detect orientation using AI
        from ai.orientation import detect_orientation_ai
        detected_plane = detect_orientation_ai(arr)
        
        # Try to detect body part from filename
        filename = os.path.basename(file_path).lower()
        if any(keyword in filename for keyword in ['brain', 'head']):
            body_part_examined = "brain"
        elif any(keyword in filename for keyword in ['chest', 'lung', 'thorax']):
            body_part_examined = "chest"
        elif any(keyword in filename for keyword in ['abdomen', 'liver', 'kidney']):
            body_part_examined = "abdomen"
        elif any(keyword in filename for keyword in ['heart', 'cardiac']):
            body_part_examined = "heart"
        else:
            body_part_examined = "Unknown"
        
        logger.info(f"Loaded NIfTI file. Shape: {arr.shape}, Plane: {detected_plane}")
        
        # Reorient volume based on detected plane for consistent viewing
        from utils.image_utils import reorient_volume
        arr = reorient_volume(arr, detected_plane)
        
        return arr, detected_plane, body_part_examined
        
    except Exception as e:
        logger.error(f"NIfTI loading error: {e}")
        raise
