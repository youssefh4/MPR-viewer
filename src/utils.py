"""
Utility functions for the Medical MPR Viewer.
Includes orientation detection, shape matching, and geometric operations.
"""

import numpy as np
import os

# Try to import PyTorch for ResNet18 AI model
try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms
    from torch.nn.functional import softmax
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def match_shape(mask, volume):
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
        print(f"[Warning] Shape mismatch: mask {mask.shape}, volume {volume.shape}")
        return mask


def detect_main_plane_dicom(first_dcm_path):
    """
    Detect the main acquisition plane from DICOM ImageOrientationPatient tag.
    Returns "Axial", "Coronal", or "Sagittal".
    """
    try:
        import pydicom
        ds = pydicom.dcmread(first_dcm_path, stop_before_pixels=True)
        iop = ds.ImageOrientationPatient
        row = np.array(iop[0:3], dtype=float)
        col = np.array(iop[3:6], dtype=float)
        normal = np.cross(row, col)
        idx = int(np.argmax(np.abs(normal)))
        return ["Sagittal", "Coronal", "Axial"][idx]
    except Exception:
        return "Axial"


class OrientationResNet18(nn.Module):
    """ResNet18-based orientation classifier for medical images."""

    def __init__(self, num_classes=3):
        super(OrientationResNet18, self).__init__()
        # Load pre-trained ResNet18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Modify first conv layer to accept grayscale input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify final layer for 3 classes (Axial, Coronal, Sagittal)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


def detect_orientation_resnet18(volume):
    """
    ResNet18-based AI orientation detection.
    Uses deep learning to classify orientation as Axial, Coronal, or Sagittal.
    """
    if not TORCH_AVAILABLE:
        print("[Info] PyTorch not available, falling back to heuristic detection")
        return detect_orientation_heuristic(volume)

    try:
        print("[ResNet18 Orientation Detection] Analyzing volume...")

        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = OrientationResNet18(num_classes=3)
        
        # Load your trained model weights
        model_path = r"X:\task 2 trials\Task 2\mpr_viewer\resnet18_orientation_finetuned.pth"
        if os.path.exists(model_path):
            try:
                # Try loading as state_dict first
                state_dict = torch.load(model_path, map_location=device)
                
                # Check if the state_dict keys match our model architecture
                model_keys = set(model.state_dict().keys())
                state_dict_keys = set(state_dict.keys())
                
                if model_keys == state_dict_keys:
                    # Perfect match - load directly
                    model.load_state_dict(state_dict)
                    print(f"[ResNet18] Loaded trained model from: {model_path}")
                elif all(key.startswith('resnet.') for key in state_dict_keys):
                    # State dict has 'resnet.' prefix but model doesn't expect it
                    # This shouldn't happen with current architecture, but handle it
                    model.load_state_dict(state_dict)
                    print(f"[ResNet18] Loaded trained model from: {model_path}")
                else:
                    # State dict doesn't have 'resnet.' prefix - need to add it
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        if not key.startswith('resnet.'):
                            new_key = f'resnet.{key}'
                        else:
                            new_key = key
                        new_state_dict[new_key] = value
                    
                    model.load_state_dict(new_state_dict)
                    print(f"[ResNet18] Loaded trained model from: {model_path} (added resnet. prefix)")
                    
            except Exception as e:
                print(f"[ResNet18] State dict loading failed: {e}")
                try:
                    # Try loading as full model
                    model = torch.load(model_path, map_location=device)
                    print(f"[ResNet18] Loaded full trained model from: {model_path}")
                except Exception as e2:
                    print(f"[ResNet18] Full model loading failed: {e2}")
                    print(f"[ResNet18] Using untrained model")
        else:
            print(f"[ResNet18] Model not found at {model_path}, using untrained model")
        
        # Move model to device (only if it's a proper model object)
        if hasattr(model, 'to'):
            model = model.to(device)
        model.eval()

        # Prepare middle slices from each orientation
        shape = volume.shape
        slices = [
            volume[shape[0] // 2, :, :],  # Axial
            volume[:, shape[1] // 2, :],  # Coronal
            volume[:, :, shape[2] // 2]  # Sagittal
        ]

        # Preprocess slices
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

        predictions = []
        with torch.no_grad():
            for slice_img in slices:
                # Normalize to 0-255 range
                slice_norm = ((slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8) * 255).astype(
                    np.uint8)

                # Convert to tensor
                img_tensor = transform(slice_norm).unsqueeze(0).to(device)

                # Get prediction
                output = model(img_tensor)
                probs = softmax(output, dim=1).cpu().numpy()[0]
                predictions.append(probs)

        # Aggregate predictions
        avg_probs = np.mean(predictions, axis=0)
        predicted_idx = np.argmax(avg_probs)
        orientations = ["Axial", "Coronal", "Sagittal"]
        confidence = avg_probs[predicted_idx] * 100

        print(f"[ResNet18 Results]")
        print(f"  Probabilities: Axial={avg_probs[0]:.3f}, Coronal={avg_probs[1]:.3f}, Sagittal={avg_probs[2]:.3f}")
        print(f"  Detected: {orientations[predicted_idx]} (Confidence: {confidence:.1f}%)")

        return orientations[predicted_idx], confidence

    except Exception as e:
        print(f"[ResNet18 Error] {e}")
        print("[Info] Falling back to heuristic detection")
        orientation = detect_orientation_heuristic(volume)
        return orientation, 0.0


def detect_orientation_heuristic(volume):
    """
    Heuristic-based orientation detection (fallback method).
    Uses image statistics and shape analysis.
    """
    try:
        shape = np.array(volume.shape)

        # Calculate variance along each axis
        variance_scores = []
        for axis in range(3):
            slices = np.split(volume, min(10, shape[axis]), axis=axis)
            variances = [np.var(s) for s in slices]
            variance_scores.append(np.std(variances))

        # Analyze middle slices
        mid_axial = volume[shape[0] // 2, :, :]
        mid_coronal = volume[:, shape[1] // 2, :]
        mid_sagittal = volume[:, :, shape[2] // 2]

        # Calculate edge density
        edge_scores = [
            np.std(np.gradient(mid_axial)),
            np.std(np.gradient(mid_coronal)),
            np.std(np.gradient(mid_sagittal))
        ]

        # Symmetry detection
        symmetry_scores = []
        for mid_slice in [mid_axial, mid_coronal, mid_sagittal]:
            left_half = mid_slice[:, :mid_slice.shape[1] // 2]
            right_half = np.fliplr(mid_slice[:, mid_slice.shape[1] // 2:])
            min_width = min(left_half.shape[1], right_half.shape[1])
            symmetry = np.corrcoef(left_half[:, :min_width].flatten(),
                                   right_half[:, :min_width].flatten())[0, 1]
            symmetry_scores.append(symmetry if not np.isnan(symmetry) else 0)

        # Weighted scoring
        scores = np.array([
            0.3 * variance_scores[0] + 0.3 * edge_scores[0] + 0.4 * symmetry_scores[0],
            0.3 * variance_scores[1] + 0.3 * edge_scores[1] + 0.4 * symmetry_scores[1],
            0.3 * variance_scores[2] + 0.3 * edge_scores[2] + 0.3 * symmetry_scores[2]
        ])

        # Shape heuristic
        if shape[0] > shape[1] * 1.5 and shape[0] > shape[2] * 1.5:
            scores[0] *= 1.3

        best_idx = np.argmax(scores)
        orientations = ["Axial", "Coronal", "Sagittal"]

        print(f"[Heuristic Orientation Detection]")
        print(f"  Variance: {variance_scores}")
        print(f"  Edge: {edge_scores}")
        print(f"  Symmetry: {symmetry_scores}")
        print(f"  Final scores: {scores}")
        print(f"  Detected: {orientations[best_idx]}")

        return orientations[best_idx]
    except Exception as e:
        print(f"[Heuristic Detection Error] {e}")
        return "Axial"


def detect_orientation_ai(volume):
    """
    Main AI orientation detection function.
    Tries ResNet18 first, falls back to heuristic if unavailable.
    Returns tuple: (orientation, confidence_percentage, method_used)
    """
    if TORCH_AVAILABLE:
        orientation, confidence = detect_orientation_resnet18(volume)
        return orientation, confidence, "ResNet18"
    else:
        orientation = detect_orientation_heuristic(volume)
        return orientation, 0.0, "Heuristic"


def create_rotation_matrix(rotation_x, rotation_y, rotation_z):
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


def get_oblique_slice(volume, slider_pos, rotation_x, rotation_y, rotation_z, reference_plane="Axial"):
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
    from scipy.ndimage import map_coordinates
    
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
        normal = np.array([1, 0, 0])
        axis1 = np.array([0, 1, 0])
        axis2 = np.array([0, 0, 1])
    elif reference_plane == "Coronal":
        normal = np.array([0, 1, 0])
        axis1 = np.array([1, 0, 0])
        axis2 = np.array([0, 0, 1])
    else:  # Sagittal
        normal = np.array([0, 0, 1])
        axis1 = np.array([1, 0, 0])
        axis2 = np.array([0, 1, 0])

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
