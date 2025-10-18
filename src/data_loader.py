"""
Data loading functionality for the Medical MPR Viewer.
Handles DICOM series, NIfTI files, and external mask loading.
"""

import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from .utils import detect_main_plane_dicom, detect_orientation_ai, match_shape, TORCH_AVAILABLE
from .config import ORGAN_KEYWORDS


class DataLoader:
    """Handles loading of medical image data and masks."""
    
    def __init__(self):
        self.volume = None
        self.original_volume = None
        self.body_part_examined = "Unknown"
        self.detected_plane = "Axial"
        self.main_plane = "Axial"
        self.temp_nifti = None
        
        # External masks
        self.external_mask_volume = None
        self.external_color_masks = None
        self.using_external_masks = False
        self.external_masks_dir = None

    def load_dicom(self, folder_path):
        """
        Load a DICOM series from a folder.
        
        Args:
            folder_path: Path to the DICOM folder
            
        Returns:
            tuple: (success, error_message, volume_data, metadata)
        """
        try:
            reader = sitk.ImageSeriesReader()
            files = reader.GetGDCMSeriesFileNames(folder_path)
            if not files:
                return False, "No DICOM files found.", None, None
                
            import pydicom
            ds = pydicom.dcmread(files[0], stop_before_pixels=True)
            body_part = getattr(ds, "BodyPartExamined", None)
            modality = getattr(ds, "Modality", None)
            self.body_part_examined = f"{body_part.lower()} {modality.lower() if modality else ''}".strip() if body_part else "Unknown"
            
            reader.SetFileNames(files)
            image = reader.Execute()
            arr = sitk.GetArrayFromImage(image).astype(np.float32)
            
            self.original_volume = arr.copy()
            self.volume = arr
            self.detected_plane = detect_main_plane_dicom(files[0])
            self.main_plane = self.detected_plane
            self.temp_nifti = os.path.join(folder_path, "converted_for_seg.nii.gz")
            sitk.WriteImage(image, self.temp_nifti)
            
            metadata = {
                'num_slices': len(files),
                'shape': arr.shape,
                'main_plane': self.main_plane,
                'body_part': self.body_part_examined,
                'temp_nifti': self.temp_nifti
            }
            
            return True, None, arr, metadata
            
        except Exception as e:
            return False, str(e), None, None

    def load_nifti(self, file_path):
        """
        Load a NIfTI file (.nii or .nii.gz).
        
        Args:
            file_path: Path to the NIfTI file
            
        Returns:
            tuple: (success, error_message, volume_data, metadata)
        """
        try:
            # Load NIfTI file using nibabel
            nifti_img = nib.load(file_path)
            arr = nifti_img.get_fdata().astype(np.float32)

            # Store the original volume
            self.original_volume = arr.copy()
            self.volume = arr

            # AI-based orientation detection
            orientation, confidence, method = detect_orientation_ai(arr)
            self.detected_plane = orientation
            self.main_plane = orientation

            # Try to detect body part from filename
            filename = os.path.basename(file_path).lower()
            if any(keyword in filename for keyword in ['brain', 'head']):
                self.body_part_examined = "brain"
            elif any(keyword in filename for keyword in ['chest', 'lung', 'thorax']):
                self.body_part_examined = "chest"
            elif any(keyword in filename for keyword in ['abdomen', 'liver', 'kidney']):
                self.body_part_examined = "abdomen"
            elif any(keyword in filename for keyword in ['heart', 'cardiac']):
                self.body_part_examined = "heart"
            else:
                self.body_part_examined = "Unknown"

            # Store path for segmentation
            self.temp_nifti = file_path

            metadata = {
                'shape': arr.shape,
                'main_plane': self.main_plane,
                'body_part': self.body_part_examined,
                'temp_nifti': self.temp_nifti,
                'ai_method': method,
                'confidence': confidence
            }

            return True, None, arr, metadata
            
        except Exception as e:
            return False, str(e), None, None

    def load_external_masks(self, mask_files_to_load):
        """
        Load pre-existing segmentation masks from files.
        
        Args:
            mask_files_to_load: List of tuples (file_path, mask_name)
            
        Returns:
            tuple: (success, error_message, mask_data, metadata)
        """
        if self.volume is None:
            return False, "No volume loaded. Load a DICOM series or NIfTI file first.", None, None

        if not mask_files_to_load:
            return False, "No mask files selected.", None, None

        try:
            self.external_mask_volume = np.zeros_like(self.volume, dtype=np.uint8)
            self.external_color_masks = {}
            self.using_external_masks = True

            for mask_path, mask_name in mask_files_to_load:
                arr = nib.load(mask_path).get_fdata() > 0
                arr = match_shape(arr, self.volume)
                arr = arr.astype(np.uint8)
                self.external_mask_volume |= arr
                self.external_color_masks.setdefault(mask_name, np.zeros_like(self.volume, dtype=np.uint8))
                self.external_color_masks[mask_name] |= arr

            mask_names = [name for _, name in mask_files_to_load]

            # Detect main organ from loaded masks
            main_organ = self.detect_main_organ_from_masks(self.external_color_masks)

            metadata = {
                'mask_names': mask_names,
                'main_organ': main_organ,
                'num_masks': len(mask_names)
            }

            return True, None, self.external_color_masks, metadata

        except Exception as e:
            return False, str(e), None, None

    def detect_main_organ_from_masks(self, mask_dict):
        """Detect the main organ from already-loaded external masks by keyword matching."""
        organ_volumes = {}
        for organ, keywords in ORGAN_KEYWORDS.items():
            vol_count = 0
            for mask_name, mask_array in mask_dict.items():
                mask_name_lower = mask_name.lower()
                # Check if any keyword matches the mask name
                if any(keyword in mask_name_lower for keyword in keywords):
                    vol_count += np.sum(mask_array > 0)
            organ_volumes[organ] = vol_count

        # Return organ with largest volume
        if organ_volumes:
            main_organ = max(organ_volumes, key=organ_volumes.get)
            if organ_volumes[main_organ] > 0:
                return main_organ

        return "Unknown"

    def reset_masks(self):
        """Reset all mask data."""
        self.external_mask_volume = None
        self.external_color_masks = None
        self.using_external_masks = False
        self.external_masks_dir = None

    def get_volume_info(self):
        """Get current volume information."""
        if self.volume is None:
            return None
        
        return {
            'shape': self.volume.shape,
            'main_plane': self.main_plane,
            'detected_plane': self.detected_plane,
            'body_part': self.body_part_examined,
            'has_external_masks': self.using_external_masks,
            'temp_nifti': self.temp_nifti
        }
