"""
Medical MPR Viewer - DICOM & NIfTI Multi-Planar Reconstruction Viewer

A comprehensive medical image viewer with:
- DICOM series and NIfTI file support
- Multi-planar reconstruction (Axial, Sagittal, Coronal views)
- Oblique slicing with 3D rotations
- TotalSegmentator integration for automatic organ segmentation
- External mask loading support
- Crosshair navigation across views
- ROI zoom functionality
- Automatic playback with play/pause, speed control, and direction control
- Collapsible sidebar sections for organized, clutter-free interface
- Professional GUI with modern controls
"""

from .mpr_viewer import DICOM_MPR_Viewer

__version__ = "1.0.0"
__all__ = ["DICOM_MPR_Viewer"]
