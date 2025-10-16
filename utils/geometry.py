"""
Geometric utilities for 3D medical image processing.
"""

import numpy as np
from scipy.ndimage import map_coordinates

def create_rotation_matrix(rotation_x: float, rotation_y: float, rotation_z: float) -> np.ndarray:
    """Create a 3D rotation matrix from euler angles in degrees."""
    rx = np.radians(rotation_x)
    ry = np.radians(rotation_y)
    rz = np.radians(rotation_z)
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    
    return Rz @ Ry @ Rx

def get_oblique_slice(volume: np.ndarray, slider_pos: int, rotation_x: float, 
                     rotation_y: float, rotation_z: float, reference_plane: str = "Axial") -> np.ndarray:
    """
    Extract an oblique slice from the volume using scipy interpolation.
    
    Args:
        volume: 3D numpy array of the medical image
        slider_pos: Position of the slice along the reference plane
        rotation_x, rotation_y, rotation_z: Rotation angles in degrees
        reference_plane: "Axial", "Coronal", or "Sagittal"
    
    Returns:
        2D numpy array representing the oblique slice
    """
    shape = np.array(volume.shape)
    
    # Clip slider_pos to valid range based on reference plane
    if reference_plane == "Axial":
        slider_pos = int(np.clip(slider_pos, 0, shape[0] - 1))
    elif reference_plane == "Coronal":
        slider_pos = int(np.clip(slider_pos, 0, shape[1] - 1))
    else:  # Sagittal
        slider_pos = int(np.clip(slider_pos, 0, shape[2] - 1))
    
    out_size = max(shape)
    center = shape / 2

    # Define reference vectors based on the reference plane
    if reference_plane == "Axial":
        normal = np.array([1,0,0])
        axis1 = np.array([0,1,0])
        axis2 = np.array([0,0,1])
    elif reference_plane == "Coronal":
        normal = np.array([0,1,0])
        axis1 = np.array([1,0,0])
        axis2 = np.array([0,0,1])
    else:  # Sagittal
        normal = np.array([0,0,1])
        axis1 = np.array([1,0,0])
        axis2 = np.array([0,1,0])

    # Apply rotations
    R = create_rotation_matrix(rotation_x, rotation_y, rotation_z)
    plane_normal = R @ normal
    axis1 = R @ axis1
    axis2 = R @ axis2

    # Create sampling grid
    half_range = out_size // 2
    offset = (slider_pos - half_range)
    plane_center = center + offset * plane_normal

    grid = np.arange(-half_range, half_range)
    xx, yy = np.meshgrid(grid, grid, indexing='ij')
    
    # Calculate sampling coordinates
    coords = (
        plane_center[0] + xx * axis1[0] + yy * axis2[0],
        plane_center[1] + xx * axis1[1] + yy * axis2[1],
        plane_center[2] + xx * axis1[2] + yy * axis2[2],
    )

    # Sample the volume using scipy interpolation
    oblique_slice = map_coordinates(volume, coords, order=1, mode='constant', cval=0)
    
    return oblique_slice
