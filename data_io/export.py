"""
Export functionality for medical images.
"""

import os
import numpy as np
import nibabel as nib
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

def export_slices(volume: np.ndarray, start: int, end: int, filename: str, 
                 mask_volume: Optional[np.ndarray] = None, 
                 affine: Optional[np.ndarray] = None, 
                 header: Optional[object] = None) -> List[str]:
    """
    Export a range of slices to NIfTI files.
    
    Args:
        volume: 3D volume data
        start: Start slice index
        end: End slice index (exclusive)
        filename: Output filename
        mask_volume: Optional mask volume
        affine: Optional affine matrix
        header: Optional NIfTI header
        
    Returns:
        List of created filenames
    """
    try:
        img_data = volume[start:end, :, :]
        
        img_nifti = nib.Nifti1Image(
            img_data, 
            affine=affine if affine is not None else np.eye(4), 
            header=header
        )
        
        # Save image file
        nib.save(img_nifti, filename)
        files_created = [os.path.basename(filename)]
        
        # Save mask if available
        if mask_volume is not None:
            mask_data = mask_volume[start:end, :, :]
            mask_nifti = nib.Nifti1Image(
                mask_data, 
                affine=affine if affine is not None else np.eye(4), 
                header=header
            )
            
            # Create mask filename
            if filename.endswith('.nii.gz'):
                mask_filename = filename[:-7] + '_mask.nii.gz'
            elif filename.endswith('.nii'):
                mask_filename = filename[:-4] + '_mask.nii'
            else:
                mask_filename = filename + '_mask.nii.gz'
            
            nib.save(mask_nifti, mask_filename)
            files_created.append(os.path.basename(mask_filename))
        
        logger.info(f"Exported slices {start}-{end-1} ({end-start} slices)")
        return files_created
        
    except Exception as e:
        logger.error(f"Export error: {e}")
        raise
