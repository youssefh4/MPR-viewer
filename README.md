# Medical MPR Viewer

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![PyQt5](https://img.shields.io/badge/PyQt5-GUI-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Stable-brightgreen.svg)

**A comprehensive medical image viewer with advanced visualization capabilities for DICOM and NIfTI files**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Documentation](#documentation) • [Contributing](#contributing)

</div>

---

## 🚀 Features

### Core Functionality
- **Multi-Planar Reconstruction (MPR)** - View medical images in axial, sagittal, and coronal planes
- **Oblique Slicing** - Advanced 3D rotation and oblique plane visualization
- **Crosshair Navigation** - Synchronized navigation across all views
- **ROI Zoom** - Region of interest zoom functionality
- **Automatic Playback** - Play/pause with speed and direction control

### File Format Support
- **DICOM Series** - Full support for DICOM medical imaging files
- **NIfTI Files** - Support for NIfTI-1 and NIfTI-2 formats
- **External Masks** - Load custom segmentation masks

### Advanced Segmentation
- **TotalSegmentator Integration** - Automatic organ segmentation using AI
- **Organ Detection** - Automatic detection of lungs, heart, brain, kidneys, liver, spleen, and spine
- **Color-coded Visualization** - Different colors for different organ types
- **Mask Management** - Load, save, and manipulate segmentation masks

### User Interface
- **Modern GUI** - Professional PyQt5-based interface
- **Collapsible Sidebar** - Organized, clutter-free interface
- **Responsive Design** - Adapts to different screen sizes
- **Intuitive Controls** - Easy-to-use medical imaging tools

#### Appearance Modes

**Dark Mode**
![Medical MPR Viewer - Dark Mode]
*Dark mode example: multi-planar views with sidebar in dark theme*

<img width="1920" height="1020" alt="Screenshot 2025-10-21 201045" src="https://github.com/user-attachments/assets/daffd774-53aa-4484-bfab-0d9f2523bd0a" />

**Light Mode**

![Medical MPR Viewer - Light Mode]
*Light mode example: multi-planar views with sidebar in light theme*

<img width="1920" height="1020" alt="Screenshot 2025-10-21 202331" src="https://github.com/user-attachments/assets/d6e38f6c-cb9d-4c34-9273-5af791d99ed4" />


## 📋 Requirements

### System Requirements
- **Python**: 3.7 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 4GB RAM (8GB recommended for large datasets)
- **GPU**: Recommended for TotalSegmentator (optional)

### Dependencies
- PyQt5 >= 5.15.0
- NumPy >= 1.21.0
- Nibabel >= 3.2.0
- SimpleITK >= 2.1.0
- TotalSegmentator >= 1.5.0
- PyTorch >= 1.9.0 (optional, for AI orientation detection)

## 🛠️ Installation

### Quick Start
```bash
# Clone the repository
git clone https://github.com/youssefh4/MPR-viewer.git
cd MPR-viewer

# Install dependencies
pip install -r requirements.txt

# Run the application
python run_viewer.py
```

### Alternative Installation
```bash
# Install specific dependencies
pip install PyQt5 numpy nibabel SimpleITK totalsegmentator

# Optional: Install PyTorch for enhanced features
pip install torch torchvision
```

## 🚀 Usage

### Basic Usage
1. **Launch the Application**:
   ```bash
   python run_viewer.py
   ```

2. **Load Medical Images**:
   - Click "Load DICOM Series" or "Load NIfTI File"
   - Navigate to your medical imaging data
   - Select the folder/file to load

3. **Navigate Images**:
   - Use mouse wheel to scroll through slices
   - Click and drag to pan
   - Use crosshairs to navigate between views

4. **Run Segmentation** (Optional):
   - Click "Run TotalSegmentator" in the sidebar
   - Wait for AI processing to complete
   - View segmented organs with color coding

### Advanced Features
- **Oblique Slicing**: Use rotation controls for 3D visualization
- **Playback**: Enable automatic slice playback with speed control
- **ROI Zoom**: Select regions of interest for detailed examination
- **External Masks**: Load custom segmentation results

## 📁 Project Structure

```
MPR-viewer/
├── src/                           # Source code
│   ├── mpr_viewer.py              # Main application
│   ├── data_loader.py             # Data loading utilities
│   ├── segmentation.py            # Segmentation management
│   ├── ui_components.py           # UI components
│   ├── utils.py                   # Utility functions
│   ├── config.py                  # Configuration settings
│   └── __init__.py                # Package initialization
├── models/                         # Model files
│   └── resnet18_orientation_finetuned.pth
├── data/                           # Data directory
│   └── totalsegmentator_output/   # Segmentation outputs
├── run_viewer.py                  # Main launcher script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore                     # Git ignore rules
```

## 🔧 Configuration

The application can be configured through `src/config.py`:

- **Organ Colors**: Customize colors for different organ types
- **Organ Groups**: Define organ groupings for segmentation
- **Default Settings**: Window size, playback speed, output directories

## 📖 Documentation

### API Reference
- `DICOM_MPR_Viewer`: Main application class
- `DataLoader`: Handles medical image loading
- `SegmentationManager`: Manages AI segmentation
- `CollapsibleBox`: UI component for organized interface

### Examples
```python
# Basic usage
from src.mpr_viewer import DICOM_MPR_Viewer
from PyQt5.QtWidgets import QApplication

app = QApplication([])
viewer = DICOM_MPR_Viewer()
viewer.show()
app.exec_()
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone the repository
git clone https://github.com/youssefh4/MPR-viewer.git
cd MPR-viewer

# Install development dependencies
pip install -r requirements.txt

# Run the application
python run_viewer.py
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TotalSegmentator** - For providing excellent organ segmentation capabilities
- **PyQt5** - For the robust GUI framework
- **SimpleITK** - For medical image processing
- **Nibabel** - For NIfTI file support

## 📞 Support

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/youssefh4/MPR-viewer/issues)
- **Discussions**: Join community discussions on [GitHub Discussions](https://github.com/youssefh4/MPR-viewer/discussions)

## 🔮 Roadmap

- [ ] Web-based viewer using Streamlit
- [ ] Additional segmentation models
- [ ] 3D volume rendering
- [ ] Plugin system for extensions
- [ ] Cloud deployment support
- [ ] Mobile app companion

---

<div align="center">

**Made with ❤️ for the medical imaging community**

[⭐ Star this repo](https://github.com/youssefh4/MPR-viewer) • [🐛 Report Bug](https://github.com/youssefh4/MPR-viewer/issues) • [💡 Request Feature](https://github.com/youssefh4/MPR-viewer/i[...]

</div>
