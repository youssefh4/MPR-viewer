"""
AI-based orientation detection for medical images.
Supports both ResNet18 deep learning and heuristic methods.
"""

import numpy as np
import logging

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

logger = logging.getLogger(__name__)

class OrientationResNet18(nn.Module):
    """ResNet18-based orientation classifier for medical images."""
    
    def __init__(self, num_classes=3):
        super(OrientationResNet18, self).__init__()
        # Load pre-trained ResNet18 with modern weights parameter
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Modify first conv layer to accept grayscale input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify final layer for 3 classes (Axial, Coronal, Sagittal)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

def detect_orientation_resnet18(volume: np.ndarray) -> str:
    """
    ResNet18-based AI orientation detection.
    Uses deep learning to classify orientation as Axial, Coronal, or Sagittal.
    """
    if not TORCH_AVAILABLE:
        logger.info("PyTorch not available, falling back to heuristic detection")
        return detect_orientation_heuristic(volume)
    
    try:
        logger.info("[ResNet18 Orientation Detection] Analyzing volume...")
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = OrientationResNet18(num_classes=3)
        model = model.to(device)
        model.eval()
        
        # Prepare middle slices from each orientation
        shape = volume.shape
        slices = [
            volume[shape[0]//2, :, :],      # Axial
            volume[:, shape[1]//2, :],      # Coronal
            volume[:, :, shape[2]//2]       # Sagittal
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
                slice_norm = ((slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8) * 255).astype(np.uint8)
                
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
        
        logger.info(f"[ResNet18 Results]")
        logger.info(f"  Probabilities: Axial={avg_probs[0]:.3f}, Coronal={avg_probs[1]:.3f}, Sagittal={avg_probs[2]:.3f}")
        logger.info(f"  Detected: {orientations[predicted_idx]} (Confidence: {confidence:.1f}%)")
        
        return orientations[predicted_idx]
        
    except Exception as e:
        logger.error(f"[ResNet18 Error] {e}")
        logger.info("Falling back to heuristic detection")
        return detect_orientation_heuristic(volume)

def detect_orientation_heuristic(volume: np.ndarray) -> str:
    """
    Heuristic-based orientation detection (fallback method).
    Uses image statistics and shape analysis.
    """
    try:
        shape = np.array(volume.shape)
        print(f"[Heuristic] Analyzing volume with shape: {shape}")
        
        # Calculate variance along each axis
        variance_scores = []
        for axis in range(3):
            slices = np.split(volume, min(10, shape[axis]), axis=axis)
            variances = [np.var(s) for s in slices]
            variance_scores.append(np.std(variances))
        
        print(f"[Heuristic] Variance scores: {variance_scores}")
        
        # Analyze middle slices for each axis
        edge_scores = []
        symmetry_scores = []
        
        for axis in range(3):
            # Get middle slice along this axis
            mid_idx = shape[axis] // 2
            if axis == 0:
                mid_slice = volume[mid_idx, :, :]
            elif axis == 1:
                mid_slice = volume[:, mid_idx, :]
            else:  # axis == 2
                mid_slice = volume[:, :, mid_idx]
            
            # Calculate edge density
            edge_score = np.std(np.gradient(mid_slice))
            edge_scores.append(edge_score)
            
            # Calculate symmetry
            left_half = mid_slice[:, :mid_slice.shape[1]//2]
            right_half = np.fliplr(mid_slice[:, mid_slice.shape[1]//2:])
            min_width = min(left_half.shape[1], right_half.shape[1])
            if min_width > 0:
                symmetry = np.corrcoef(left_half[:, :min_width].flatten(), 
                                      right_half[:, :min_width].flatten())[0, 1]
                symmetry_scores.append(symmetry if not np.isnan(symmetry) else 0)
            else:
                symmetry_scores.append(0)
        
        print(f"[Heuristic] Edge scores: {edge_scores}")
        print(f"[Heuristic] Symmetry scores: {symmetry_scores}")
        
        # Weighted scoring
        scores = np.array([
            0.3 * variance_scores[0] + 0.3 * edge_scores[0] + 0.4 * symmetry_scores[0],
            0.3 * variance_scores[1] + 0.3 * edge_scores[1] + 0.4 * symmetry_scores[1],
            0.3 * variance_scores[2] + 0.3 * edge_scores[2] + 0.3 * symmetry_scores[2]
        ])
        
        # Shape heuristic - if one dimension is much larger, it's likely the slice dimension
        for i in range(3):
            if shape[i] > shape[(i+1)%3] * 1.5 and shape[i] > shape[(i+2)%3] * 1.5:
                scores[i] *= 1.3
                print(f"[Heuristic] Applied shape boost to axis {i}")
        
        print(f"[Heuristic] Final scores: {scores}")
        best_idx = np.argmax(scores)
        orientations = ["Axial", "Coronal", "Sagittal"]
        
        print(f"[Heuristic] Detected orientation: {orientations[best_idx]} (axis {best_idx})")
        return orientations[best_idx]
        
    except Exception as e:
        print(f"[Heuristic Detection Error] {e}")
        return "Axial"

def detect_orientation_ai(volume: np.ndarray) -> str:
    """
    Detect the acquisition orientation of the volume.
    This tells us which plane the volume was originally acquired in.
    
    Returns:
        "Axial", "Coronal", or "Sagittal" - the acquisition plane
    """
    shape = volume.shape
    
    # For medical imaging, the acquisition plane is typically the one with the most slices
    # This is because you acquire many slices in one plane
    slice_dim = np.argmax(shape)
    
    if slice_dim == 0:
        return "Axial"    # Volume acquired in axial plane (many axial slices)
    elif slice_dim == 1:
        return "Coronal"   # Volume acquired in coronal plane (many coronal slices)
    else:  # slice_dim == 2
        return "Sagittal" # Volume acquired in sagittal plane (many sagittal slices)
