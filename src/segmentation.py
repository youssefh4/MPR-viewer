"""
Segmentation functionality for the Medical MPR Viewer.
Handles TotalSegmentator integration and organ detection.
"""

import os
import numpy as np
import nibabel as nib
from totalsegmentator.python_api import totalsegmentator

from .config import ORGAN_GROUPS_SIMPLE, HEART_COLOR_GROUPS
from .utils import match_shape


class SegmentationManager:
    """Handles segmentation operations and organ detection."""
    
    def __init__(self, output_dir="totalsegmentator_output"):
        self.output_dir = output_dir
        self.mask_volume = None
        self.color_masks = None

    def run_totalsegmentator(self, input_path):
        """
        Run TotalSegmentator on the input file.
        
        Args:
            input_path: Path to the input NIfTI file
            
        Returns:
            tuple: (success, error_message, main_organ)
        """
        if not os.path.exists(input_path):
            return False, "Input file does not exist.", None

        # Use absolute path for output directory
        output_dir_abs = os.path.abspath(self.output_dir)
        os.makedirs(output_dir_abs, exist_ok=True)

        print(f"[Info] Running TotalSegmentator...")
        print(f"[Info] Input: {input_path}")
        print(f"[Info] Output: {output_dir_abs}")

        try:
            totalsegmentator(input=input_path, output=output_dir_abs)

            # Check if files were created
            if os.path.exists(output_dir_abs):
                created_files = [f for f in os.listdir(output_dir_abs) if f.endswith('.nii.gz')]
                print(f"[Info] TotalSegmentator created {len(created_files)} mask files")
            else:
                print(f"[Error] Output directory not found after segmentation!")
                return False, "Output directory not found after segmentation.", None

            main_organ = self.detect_main_organ(output_dir_abs)
            return True, None, main_organ

        except Exception as e:
            print(f"[Error] Segmentation failed: {e}")
            return False, str(e), None

    def detect_main_organ(self, output_dir_abs):
        """Detect the main organ from TotalSegmentator output by volume."""
        # Check if output directory exists and has files
        if not os.path.exists(output_dir_abs):
            print(f"[Warning] Output directory '{output_dir_abs}' does not exist")
            return "Unknown"

        mask_files = [f for f in os.listdir(output_dir_abs) if f.endswith('.nii.gz')]
        if not mask_files:
            print(f"[Warning] No mask files found in '{output_dir_abs}'")
            return "Unknown"

        print(f"[Info] Found {len(mask_files)} mask files in output directory")

        organ_volumes = {}
        for organ, names in ORGAN_GROUPS_SIMPLE.items():
            vol_count = 0
            for name in names:
                path = os.path.join(output_dir_abs, f"{name}.nii.gz")
                if os.path.exists(path):
                    try:
                        arr = nib.load(path).get_fdata() > 0
                        vol_count += np.sum(arr)
                        if vol_count > 0:
                            print(f"[Info] Found {name}: {np.sum(arr)} voxels")
                    except Exception as e:
                        print(f"[Warning] Error loading {name}: {e}")
                        continue
            organ_volumes[organ] = vol_count

        print(f"[Info] Organ volumes: {organ_volumes}")
        main_organ = max(organ_volumes, key=organ_volumes.get)
        volume = organ_volumes[main_organ]
        if volume > 0:
            return main_organ
        else:
            return "Unknown"

    def prepare_masks(self, volume, organ, using_external_masks=False, external_color_masks=None):
        """
        Prepare organ masks for display overlay.
        
        Args:
            volume: The 3D volume array
            organ: Selected organ name
            using_external_masks: Whether using external masks
            external_color_masks: External mask data if available
            
        Returns:
            tuple: (mask_volume, color_masks)
        """
        self.mask_volume = None
        self.color_masks = None

        if using_external_masks and external_color_masks is not None:
            mask_name = organ
            mask = external_color_masks.get(mask_name)
            if mask is not None:
                self.mask_volume = mask
                self.color_masks = {mask_name: mask}
            else:
                self.mask_volume = np.zeros_like(volume, dtype=np.uint8)
                self.color_masks = {}
            return self.mask_volume, self.color_masks

        if volume is None or organ == "None":
            return self.mask_volume, self.color_masks

        # Use absolute path for output directory
        output_dir_abs = os.path.abspath(self.output_dir)

        try:
            self.mask_volume = np.zeros_like(volume, dtype=np.uint8)
            self.color_masks = {}
            
            if organ == "Heart":
                for key, names in HEART_COLOR_GROUPS.items():
                    for name in names:
                        path = os.path.join(output_dir_abs, f"{name}.nii.gz")
                        if os.path.exists(path):
                            m = match_shape(nib.load(path).get_fdata() > 0, volume)
                            self.mask_volume |= m
                            k = "heart_main" if key == "main" else "heart_vessels"
                            self.color_masks.setdefault(k, np.zeros_like(volume, dtype=np.uint8))
                            self.color_masks[k] |= m
            else:
                for name in ORGAN_GROUPS_SIMPLE[organ]:
                    path = os.path.join(output_dir_abs, f"{name}.nii.gz")
                    if os.path.exists(path):
                        m = match_shape(nib.load(path).get_fdata() > 0, volume)
                        self.mask_volume |= m
                        k = organ.lower()
                        self.color_masks.setdefault(k, np.zeros_like(volume, dtype=np.uint8))
                        self.color_masks[k] |= m
        except Exception as e:
            print(f"[Error] Mask preparation failed: {e}")
            self.mask_volume = np.zeros_like(volume, dtype=np.uint8)
            self.color_masks = {}

        return self.mask_volume, self.color_masks

    def get_mask_slice(self, slice_idx, plane, volume_shape):
        """
        Get mask slice for a specific plane and slice index.
        
        Args:
            slice_idx: Slice index
            plane: Plane name ("Axial", "Coronal", "Sagittal")
            volume_shape: Shape of the volume
            
        Returns:
            numpy array: Mask slice or None
        """
        if self.mask_volume is None:
            return None
            
        shape = self.mask_volume.shape
        if plane == "Axial":
            safe_k = int(np.clip(slice_idx, 0, shape[0] - 1))
            return self.mask_volume[safe_k, :, :]
        elif plane == "Coronal":
            safe_k = int(np.clip(slice_idx, 0, shape[1] - 1))
            return self.mask_volume[:, safe_k, :]
        else:  # Sagittal
            safe_k = int(np.clip(slice_idx, 0, shape[2] - 1))
            return self.mask_volume[:, :, safe_k]

    def get_color_mask_slice(self, slice_idx, plane, volume_shape):
        """
        Get color mask slice for a specific plane and slice index.
        
        Args:
            slice_idx: Slice index
            plane: Plane name ("Axial", "Coronal", "Sagittal")
            volume_shape: Shape of the volume
            
        Returns:
            dict: Color mask slices or None
        """
        if not self.color_masks:
            return None
            
        out = {}
        for name, arr in self.color_masks.items():
            shape = arr.shape
            if plane == "Axial":
                safe_k = int(np.clip(slice_idx, 0, shape[0] - 1))
                out[name] = arr[safe_k, :, :]
            elif plane == "Coronal":
                safe_k = int(np.clip(slice_idx, 0, shape[1] - 1))
                out[name] = arr[:, safe_k, :]
            else:  # Sagittal
                safe_k = int(np.clip(slice_idx, 0, shape[2] - 1))
                out[name] = arr[:, :, safe_k]
        return out

    def reset_masks(self):
        """Reset all mask data."""
        self.mask_volume = None
        self.color_masks = None
